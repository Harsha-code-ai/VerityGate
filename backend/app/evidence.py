"""Sentence-level evidence units and claim-to-evidence relevance.

Why this exists: running NLI against a whole 180-word chunk mixes several facts into one premise, and taking the
maximum contradiction score over unrelated passages lets an irrelevant sentence overrule a relevant one.
The verifier therefore (1) splits retrieved passages into sentences, (2) decides which sentences are actually
ABOUT the claim, and (3) only lets relevant sentences vote.

Relevance is a mix of idf-weighted term coverage (lexical), entity anchors (proper nouns, acronyms, identifiers that
must appear) and - when an embedder is available - semantic similarity. All thresholds are configuration values."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from .numeric import extract_quantities
from .textutils import STARTS_WITH_REFERENCE, collapse_ws, content_terms, entity_anchors, split_sentences

DEFAULT_DF = 0.5  # weight basis when the document's term statistics are unknown


@dataclass
class EvidenceUnit:
    unit_id: str
    chunk_id: str
    page: int
    section: str | None
    text: str  # the sentence (or sentence window) shown to the user
    premise: str  # what the NLI model reads: the sentence, plus the previous one when it starts with "It/They/This..."
    context: str  # sentence + neighbours + heading; consulted only for years
    cited: bool = False
    terms: frozenset[str] = field(default_factory=frozenset)


def build_units(passages, cited_chunk_ids: set[str] | None = None) -> list[EvidenceUnit]:
    """Split passages into sentence units. Identical sentences from overlapping chunks on the same page are kept once."""
    cited_chunk_ids = cited_chunk_ids or set()
    units: list[EvidenceUnit] = []
    seen: dict[tuple[int, str], int] = {}
    for p in passages:
        sents = split_sentences(p.text)
        for i, s in enumerate(sents):
            if len(s.split()) < 2:
                continue
            prev_s = sents[i - 1] if i > 0 else ""
            next_s = sents[i + 1] if i + 1 < len(sents) else ""
            premise = f"{prev_s} {s}" if prev_s and STARTS_WITH_REFERENCE.match(s) else s
            context = collapse_ws(" ".join(x for x in (p.section or "", prev_s, s, next_s) if x))
            cited = p.chunk_id in cited_chunk_ids
            key = (p.page, collapse_ws(s).lower())
            if key in seen:
                if cited and not units[seen[key]].cited:  # prefer the copy the answer cited
                    units[seen[key]].cited, units[seen[key]].chunk_id = True, p.chunk_id
                continue
            seen[key] = len(units)
            units.append(EvidenceUnit(f"{p.chunk_id}#{i}", p.chunk_id, p.page, p.section, s, premise, context, cited,
                                      terms=frozenset(content_terms(f"{p.section or ''} {premise}"))))
    return units


@dataclass
class ClaimProfile:
    text: str
    terms: frozenset[str]
    anchors: tuple[str, ...]  # entity-like terms that must appear for a passage to be "about" the claim
    focus: frozenset[str]  # non-entity, non-ubiquitous terms (what is being said about the entities)
    weights: dict[str, float]
    identifiers: tuple[str, ...] = ()  # anchors made of letters AND digits (X200, A320): an exact entity, must be present to support


def profile_claim(text: str, df: Mapping[str, float] | None = None, anchor_df: float = 0.5, rare_df: float = 0.1) -> ClaimProfile:
    # Quantities (amounts, currencies, dates...) are compared by the typed checker. If their words counted here, a claim whose
    # only difference is "euros" instead of "dollars" would look UNRELATED to the sentence that contradicts it.
    df = df or None  # an empty mapping means 'unknown', same as None
    stripped = text
    for q in sorted(extract_quantities(text), key=lambda q: -q.start):
        if q.end > q.start:
            stripped = stripped[: q.start] + " " + stripped[q.end :]
    terms = frozenset(content_terms(stripped))
    all_anchors = entity_anchors(stripped, df, rare_df)
    anchors = tuple(a for a in all_anchors if df is None or df.get(a, 0.0) <= anchor_df)
    def weight(t: str) -> float:
        # Rarer terms carry more weight. A term found NOWHERE in the document is a paraphrase, not a discriminator: neutral weight.
        d = df.get(t, 0.0) if df is not None else DEFAULT_DF
        return (1.0 + (1.0 - d) if d > 0 else 1.0) + (1.0 if t in anchors else 0.0)

    weights = {t: weight(t) for t in terms}
    focus = frozenset(t for t in terms if t not in anchors and (df is None or df.get(t, 0.0) <= 0.5))
    # identifiers such as X200 / A320 discriminate by nature, so they are exempt from the document-frequency filter
    identifiers = tuple(a for a in all_anchors if any(c.isdigit() for c in a) and any(c.isalpha() for c in a))
    return ClaimProfile(text, terms, anchors, focus, weights, identifiers)


@dataclass
class Relevance:
    coverage: float  # idf-weighted share of the claim's terms found in the unit
    anchors_present: bool
    identifiers_present: bool
    focus_coverage: float
    similarity: float | None
    score: float  # ranking score in [0, 1]


def relevance(profile: ClaimProfile, unit: EvidenceUnit, similarity: float | None = None) -> Relevance:
    total = sum(profile.weights.values()) or 1.0
    found = sum(w for t, w in profile.weights.items() if t in unit.terms)
    coverage = found / total
    anchors_present = all(a in unit.terms for a in profile.anchors)
    identifiers_present = all(a in unit.terms for a in profile.identifiers)
    focus_cov = (sum(1 for t in profile.focus if t in unit.terms) / len(profile.focus)) if profile.focus else 1.0
    score = 0.6 * coverage + 0.4 * max(similarity, 0.0) if similarity is not None else coverage
    if profile.anchors and not anchors_present:
        score *= 0.5
    return Relevance(coverage, anchors_present, identifiers_present, focus_cov, similarity, min(score + (0.1 if unit.cited else 0.0), 1.0))


def question_terms_found(question: str, passages, df: Mapping[str, float] | None) -> bool:
    """False when a distinctive term of the question (a name, an acronym, a rare word) appears in none of the passages.

    Used to avoid wasting retries: if the question asks about 'Tokyo' and no retrieved text mentions it, widening the
    search will not help."""
    prof = profile_claim(question, df, anchor_df=0.3, rare_df=0.15)
    distinctive = set(prof.anchors) | {t for t in prof.terms if df is not None and df.get(t, 0.0) == 0.0}
    if not distinctive:
        return True
    text_terms: set[str] = set()
    for p in passages:
        text_terms |= set(content_terms(f"{p.section or ''} {p.text}"))
    return all(t in text_terms for t in distinctive)
