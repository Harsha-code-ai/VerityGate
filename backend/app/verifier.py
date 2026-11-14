"""Claim verification: sentence-level NLI + relevance gating + deterministic quantity checks.

NLI direction: premise = an evidence SENTENCE from the PDF, hypothesis = the claim. Never the reverse, never a whole chunk.

Decision procedure for one claim (all thresholds are settings, see config.py):

 1. Candidate evidence = sentences of the retrieved passages, ranked by relevance to the claim (the cited
    sentences get a small bonus). Only the best few go through the NLI model.
 2. A sentence SUPPORTS when it is related to the claim, the NLI entailment score is high and clearly above the
    contradiction score, AND every number/date/currency/percentage in the claim is found (typed) in that sentence.
 3. A sentence CONTRADICTS only when it is STRICTLY relevant (it names the same entities and talks about the same
    thing) and either the NLI contradiction score is high or a typed quantity/direction check shows a hard conflict.
    A contradiction score from a sentence that is about something else is ignored - never counted.
 4. SUPPORTED   : at least one supporting sentence and no genuinely relevant contradicting sentence.
    CONFLICT    : supporting AND genuinely contradicting sentences exist -> reported as NOT_ESTABLISHED with
                  conflict=True and both pieces of evidence (the document disagrees with itself).
    CONTRADICTED: no support, at least one genuinely relevant contradiction.
    NOT_ESTABLISHED otherwise (nothing relevant, only neutral evidence, or a number in the claim is not stated).

The NLI model is a baseline signal; a SUPPORTED label means "a passage entails the claim according to the model and the
deterministic checks", not "the claim is true in the world"."""
from __future__ import annotations

import logging
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

from .claims import ParsedClaim
from .evidence import EvidenceUnit, Relevance, build_units, profile_claim, relevance
from .numeric import QuantityCheck, compare_quantities
from .retrieval import Passage
from .textutils import direction_of

log = logging.getLogger(__name__)

SUPPORTED, CONTRADICTED, NOT_ESTABLISHED = "SUPPORTED", "CONTRADICTED", "NOT_ESTABLISHED"


@dataclass
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float
    truncated: bool = False  # the premise did not fit the model window: scores are unreliable and are ignored


class NLIModel(Protocol):
    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """pairs = (premise/evidence sentence, hypothesis/claim)"""


# ----------------------------------------------------------------------------- real NLI model
_KNOWN_ORDERS = [  # (substring of model name, label order of the logits) - used only when config labels are generic
    ("cross-encoder/nli", ["contradiction", "entailment", "neutral"]),
    ("moritzlaurer", ["entailment", "neutral", "contradiction"]),
    ("bart-large-mnli", ["contradiction", "neutral", "entailment"]),
    ("roberta-large-mnli", ["contradiction", "neutral", "entailment"]),
]
_LABEL_WORDS = {"entailment": "entail", "contradiction": "contradict", "neutral": "neutral"}


def resolve_nli_labels(id2label: Mapping[int, str], model_name: str, override: str = "") -> dict[str, int]:
    """Map 'entailment' / 'contradiction' / 'neutral' to logit indices, or raise when it cannot be done safely.

    Priority: explicit NLI_LABEL_ORDER override -> label names in the model config -> known model families.
    Two-class models (entailment / not_entailment) are accepted: 'neutral' then means 'not entailed' and contradictions
    can only come from the deterministic checks."""
    n = len(id2label)
    if override.strip():
        order = [x.strip().lower() for x in override.split(",")]
        if sorted(order) != sorted(_LABEL_WORDS) or n != 3:
            raise RuntimeError("NLI_LABEL_ORDER must list entailment, contradiction and neutral once each (comma separated).")
        return {name: i for i, name in enumerate(order)}
    names = {int(i): str(v).lower() for i, v in id2label.items()}
    found: dict[str, int] = {}
    for idx, name in names.items():
        for label, word in _LABEL_WORDS.items():
            if word in name and not name.startswith("not"):
                found[label] = idx
        if name.replace("-", "_") in {"not_entailment", "non_entailment"}:
            found["neutral"] = idx
    if "entailment" in found and ("contradiction" in found or n == 2):
        if n == 3 and "neutral" not in found:
            raise RuntimeError(f"NLI labels {sorted(names.values())} lack 'neutral'.")
        return found
    lowered = model_name.lower()
    for key, order in _KNOWN_ORDERS:
        if key in lowered and n == 3:
            return {name: i for i, name in enumerate(order)}
    raise RuntimeError(
        f"Cannot tell which output of NLI model '{model_name}' means entailment (labels: {sorted(names.values())}). "
        "Set NLI_LABEL_ORDER, e.g. NLI_LABEL_ORDER=contradiction,entailment,neutral."
    )


_PROBES = [  # (premise, hypothesis, expected label) - a strong, unambiguous pair per class
    ("The company opened its Lisbon office on 15 June 2019.", "The Lisbon office opened in 2019.", "entailment"),
    ("The company opened its Lisbon office on 15 June 2019.", "The company has never had an office in Lisbon.", "contradiction"),
]


class TransformersNLI:
    """cross-encoder NLI classifier (runs locally via transformers/torch)."""

    def __init__(self, model_name: str, label_order: str = "", max_length: int = 512):
        self.model_name, self.label_order, self.max_length = model_name, label_order, max_length
        self._tok = self._model = None
        self._labels: dict[str, int] = {}
        self._lock = threading.Lock()

    def _load(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch  # noqa: F401
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as exc:
                raise RuntimeError("transformers/torch are not installed (see requirements-ml.txt).") from exc
            log.info("Loading NLI model %s (first run downloads it)", self.model_name)
            tok = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name).eval()
            self._labels = resolve_nli_labels(model.config.id2label, self.model_name, self.label_order)
            self._tok, self._model = tok, model
            try:
                self.self_test()
            except RuntimeError:
                self._model = None  # fail closed: do not keep a model whose label mapping looks wrong
                raise

    def _probabilities(self, pairs: list[tuple[str, str]]) -> tuple[list[list[float]], list[bool]]:
        import torch

        probs: list[list[float]] = []
        truncated: list[bool] = []
        for i in range(0, len(pairs), 16):
            batch = pairs[i : i + 16]
            enc = self._tok([p for p, _ in batch], [h for _, h in batch], padding=True, truncation="only_first",
                            max_length=self.max_length, return_tensors="pt")
            full = self._tok([p for p, _ in batch], [h for _, h in batch], truncation=False)["input_ids"]
            truncated.extend(len(ids) > self.max_length for ids in full)
            with torch.inference_mode():
                probs.extend(torch.softmax(self._model(**enc).logits, dim=-1).tolist())
        return probs, truncated

    def _to_scores(self, row: list[float], truncated: bool) -> NLIScores:
        lab = self._labels
        return NLIScores(row[lab["entailment"]], row[lab["contradiction"]] if "contradiction" in lab else 0.0, row[lab["neutral"]], truncated)

    def self_test(self) -> None:
        """Verify the label mapping on two unambiguous pairs; a swapped mapping is exactly how 'supported' turns into 'contradicted'."""
        probs, _ = self._probabilities([(p, h) for p, h, _ in _PROBES])
        for (p, h, expected), row in zip(_PROBES, probs):
            if expected == "contradiction" and "contradiction" not in self._labels:
                continue
            best = max(self._labels, key=lambda name: row[self._labels[name]])
            if best != expected:
                raise RuntimeError(
                    f"NLI label self-test failed for model '{self.model_name}': expected '{expected}' for ({p!r}, {h!r}) but the "
                    f"model's strongest output maps to '{best}'. Check NLI_MODEL / NLI_LABEL_ORDER."
                )

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        self._load()
        probs, truncated = self._probabilities(pairs)
        return [self._to_scores(r, t) for r, t in zip(probs, truncated)]


# ----------------------------------------------------------------------------- results
@dataclass
class EvidenceItem:
    chunk_id: str
    page: int
    text: str
    entailment: float
    contradiction: float
    cited_by_model: bool
    role: str = "related"  # supporting | contradicting | related
    relevance: float = 0.0
    context: str = ""
    note: str = ""  # e.g. a quantity or direction conflict, or a rounding remark


@dataclass
class ClaimResult:
    text: str
    status: str
    reason: str
    cited_ids: list[str]
    evidence: list[EvidenceItem] = field(default_factory=list)  # supporting first, then contradicting, then related
    best_entailment: float = 0.0
    best_contradiction: float = 0.0
    citation_valid: bool = False  # did the model cite a passage that actually supports the claim?
    warnings: list[str] = field(default_factory=list)
    reason_code: str = ""
    conflict: bool = False  # the document itself contains supporting AND contradicting statements
    considered: int = 0  # how many evidence sentences were examined for this claim


@dataclass
class VerifierConfig:
    entail_thr: float = 0.80
    contra_thr: float = 0.85
    conflict_thr: float = 0.90  # contradiction score needed to call a document conflict when support also exists
    max_units: int = 6
    min_coverage: float = 0.50  # strict relevance (for contradictions)
    min_support_coverage: float = 0.30  # lenient relevance (for support; NLI entailment is itself strong evidence)
    min_similarity: float = 0.50  # embedding cosine that can replace lexical coverage
    min_focus: float = 0.50
    neutral_gate: float = 0.80  # a quantity conflict only counts when the NLI model does not call the pair clearly unrelated
    anchor_df: float = 0.50
    rare_df: float = 0.10


@dataclass
class _Scored:
    unit: EvidenceUnit
    rel: Relevance
    nli: NLIScores
    qc: QuantityCheck
    polarity_conflict: bool
    strict: bool
    related: bool


class ClaimVerifier:
    def __init__(self, nli: NLIModel, entail_thr: float = 0.80, contra_thr: float = 0.85, max_units: int = 6, *, embedder=None,
                 config: VerifierConfig | None = None):
        self.nli, self.embedder = nli, embedder
        self.cfg = config or VerifierConfig(entail_thr=entail_thr, contra_thr=contra_thr, max_units=max_units)

    # ------------------------------------------------------------------ public API
    def verify(self, claim: str, cited_ids: list[str], id_map: dict[str, Passage], passages: list[Passage],
               term_df: Mapping[str, float] | None = None) -> ClaimResult:
        return self.verify_claims([ParsedClaim(claim, cited_ids)], id_map, passages, term_df)[0]

    def verify_claims(self, claims: Sequence[ParsedClaim], id_map: dict[str, Passage], passages: list[Passage],
                      term_df: Mapping[str, float] | None = None) -> list[ClaimResult]:
        cfg = self.cfg
        # ---- 1. build candidate units once for the whole answer
        cited_all = {id_map[c].chunk_id for cl in claims for c in cl.cited if c in id_map}
        units = build_units(passages, cited_all)
        sims = self._similarities([c.text for c in claims], units)
        plans: list[list[tuple[EvidenceUnit, Relevance]]] = []
        profiles = []
        for ci, cl in enumerate(claims):
            prof = profile_claim(cl.text, term_df, cfg.anchor_df, cfg.rare_df)
            profiles.append(prof)
            cited_here = {id_map[c].chunk_id for c in cl.cited if c in id_map}
            scored = []
            for ui, u in enumerate(units):
                r = relevance(prof, u, sims[ci][ui] if sims else None)
                if u.chunk_id in cited_here:
                    r.score = min(r.score + 0.1, 1.0)
                scored.append((u, r))
            scored.sort(key=lambda t: t[1].score, reverse=True)
            plans.append([t for t in scored[: cfg.max_units] if t[1].score > 0.0])
        # ---- 2. one NLI call for everything
        pairs = [(u.premise, cl.text) for cl, plan in zip(claims, plans) if not cl.problem for u, _ in plan]
        scores = iter(self.nli.predict(pairs))
        results: list[ClaimResult] = []
        for cl, prof, plan in zip(claims, profiles, plans):
            warnings = self._citation_warnings(cl, id_map)
            if cl.problem:
                results.append(ClaimResult(cl.text, NOT_ESTABLISHED, cl.problem, cl.cited, warnings=warnings, reason_code="incomplete_claim"))
                continue
            scored_units = []
            for u, rel in plan:
                nli = next(scores)
                scored_units.append(self._score(cl, u, rel, nli, prof))
            results.append(self._decide(cl, scored_units, id_map, warnings, len(plan)))
        return results

    # ------------------------------------------------------------------ internals
    def _similarities(self, claim_texts: list[str], units: list[EvidenceUnit]) -> list[list[float]] | None:
        if self.embedder is None or not units or not claim_texts:
            return None
        try:
            vecs = self.embedder.embed(claim_texts + [u.text for u in units])
        except Exception as exc:  # similarity is an optional refinement; lexical relevance still works
            log.warning("Embedding for relevance failed (%s); using lexical relevance only", type(exc).__name__)
            return None
        cv, uv = vecs[: len(claim_texts)], vecs[len(claim_texts) :]
        return [[sum(a * b for a, b in zip(c, u)) for u in uv] for c in cv]

    def _score(self, cl: ParsedClaim, u: EvidenceUnit, rel: Relevance, nli: NLIScores, prof) -> _Scored:
        cfg = self.cfg
        if nli.truncated:
            nli = NLIScores(0.0, 0.0, 1.0, True)
        sim_ok = rel.similarity is not None and rel.similarity >= cfg.min_similarity
        related = rel.coverage >= cfg.min_support_coverage or sim_ok
        strict = rel.anchors_present and (rel.coverage >= cfg.min_coverage or sim_ok) and rel.focus_coverage >= cfg.min_focus
        qc = compare_quantities(cl.text, u.text, u.context)
        claim_dir, unit_dir = direction_of(cl.text), direction_of(u.text)
        polarity = len(claim_dir) == 1 and len(unit_dir) == 1 and claim_dir != unit_dir
        return _Scored(u, rel, nli, qc, polarity, strict, related)

    def _citation_warnings(self, cl: ParsedClaim, id_map: dict[str, Passage]) -> list[str]:
        warnings = []
        if cl.cited and any(c not in id_map for c in cl.cited):
            warnings.append("The answer cited an excerpt id that was not provided.")
        if not cl.cited:
            warnings.append("The claim carried no citation marker.")
        return warnings

    def _item(self, s: _Scored, role: str) -> EvidenceItem:
        note = ""
        if role == "contradicting" and s.qc.has_conflict:
            note = "Quantity check: " + s.qc.describe_conflict()
        elif role == "contradicting" and s.polarity_conflict:
            note = "The evidence describes the opposite direction of change."
        elif role == "supporting" and s.qc.notes:
            note = "; ".join(s.qc.notes)
        return EvidenceItem(s.unit.chunk_id, s.unit.page, s.unit.text, round(s.nli.entailment, 4), round(s.nli.contradiction, 4),
                            s.unit.cited, role, round(s.rel.score, 4), s.unit.context if s.unit.context != s.unit.text else "", note)

    def _decide(self, cl: ParsedClaim, scored: list[_Scored], id_map: dict[str, Passage], warnings: list[str], considered: int) -> ClaimResult:
        cfg = self.cfg
        def hard_conflict(s: _Scored) -> bool:
            """Typed quantity/direction conflict. Ignored when the NLI model finds the sentence clearly unrelated (neutral) or
            when the claim names another period than the sentence does."""
            same_amount_other_currency = any(why.startswith("currency differs") for _, _, why in s.qc.conflicts)  # strong typed signal
            return (s.qc.has_conflict or s.polarity_conflict) and (s.nli.neutral < cfg.neutral_gate or same_amount_other_currency)

        supporting = [s for s in scored if s.related and s.rel.identifiers_present and s.nli.entailment >= cfg.entail_thr and s.nli.entailment > s.nli.contradiction
                      and s.qc.ok and not s.polarity_conflict]
        contradicting = [s for s in scored if s.strict and not any(s is x for x in supporting) and not s.qc.period_mismatch
                         and ((s.nli.contradiction >= cfg.contra_thr and s.nli.contradiction > s.nli.entailment) or hard_conflict(s))]
        supporting.sort(key=lambda s: (s.unit.cited, s.nli.entailment), reverse=True)
        contradicting.sort(key=lambda s: (s.nli.contradiction, s.qc.has_conflict), reverse=True)

        evidence = [self._item(s, "supporting") for s in supporting[:3]] + [self._item(s, "contradicting") for s in contradicting[:3]]
        shown = {id(s) for s in supporting[:3] + contradicting[:3]}
        related_rest = sorted((s for s in scored if id(s) not in shown and s.related), key=lambda s: s.rel.score, reverse=True)
        evidence += [self._item(s, "related") for s in related_rest[:2]]
        best_ent = max((s.nli.entailment for s in scored), default=0.0)
        best_con = max((s.nli.contradiction for s in scored if s.strict), default=0.0)

        def result(status: str, reason: str, code: str, conflict: bool = False, citation_valid: bool = False) -> ClaimResult:
            return ClaimResult(cl.text, status, reason, cl.cited, evidence, best_ent, best_con, citation_valid, warnings, code, conflict, considered)

        if supporting:
            top = supporting[0]
            hard = [s for s in contradicting if s.nli.contradiction >= cfg.conflict_thr]
            if hard:
                bad = hard[0]
                return result(NOT_ESTABLISHED,
                              f"The document is inconsistent about this: page {top.unit.page} supports the claim, but page {bad.unit.page} "
                              f"appears to contradict it. Neither is treated as verified.", "conflict", conflict=True)
            valid = top.unit.cited
            if not valid:
                warnings.append("Supported by a passage the answer did not cite; shown page is the verified source.")
            if contradicting:
                warnings.append(f"A weaker contradiction signal on page {contradicting[0].unit.page} was ignored because stronger, "
                                f"directly relevant evidence supports the claim.")
            return result(SUPPORTED, f"Page {top.unit.page} states this (entailment {top.nli.entailment:.2f}).", "supported", citation_valid=valid)
        if contradicting:
            bad = contradicting[0]
            if bad.qc.has_conflict:
                why = f"Page {bad.unit.page}: {bad.qc.describe_conflict()}."
                return result(CONTRADICTED, why[0].upper() + why[1:], "contradicted_quantity")
            if bad.polarity_conflict and not bad.qc.has_conflict and bad.nli.contradiction < cfg.contra_thr:
                return result(CONTRADICTED, f"Page {bad.unit.page} describes the opposite direction of change.", "contradicted_direction")
            return result(CONTRADICTED, f"A passage on page {bad.unit.page} contradicts the claim (model score {bad.nli.contradiction:.2f}).",
                          "contradicted_nli")
        # ---- not established: explain why
        if not scored:
            return result(NOT_ESTABLISHED, "No evidence passages were available.", "no_relevant_evidence")
        related = [s for s in scored if s.related]
        if not related:
            return result(NOT_ESTABLISHED, "No retrieved passage is about what this claim says.", "no_relevant_evidence")
        entailing = [s for s in related if s.nli.entailment >= cfg.entail_thr and s.nli.entailment > s.nli.contradiction]
        if entailing:
            miss = next((s for s in entailing if s.qc.missing), None)
            if miss is not None:
                q = miss.qc.missing[0]
                return result(NOT_ESTABLISHED, f"A passage on page {miss.unit.page} is related, but it does not state '{q.raw}' from the claim.",
                              "quantity_not_stated")
        return result(NOT_ESTABLISHED, "Related passages exist, but none of them clearly states this claim.", "neutral_evidence")
