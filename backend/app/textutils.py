"""Shared text helpers: normalisation, sentence splitting, light stemming and term canonicalisation.

Everything here is deterministic, dependency-free and domain-neutral. The synonym groups are a small,
general-purpose list used for lexical matching (BM25 and evidence relevance); they are NOT used to decide
whether a claim is true - that is the job of the NLI model plus the deterministic quantity checks.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping

# --------------------------------------------------------------------------- normalisation
_LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi", "\ufb04": "ffl"}
_INVISIBLE = dict.fromkeys(map(ord, "\u00ad\u200b\u200c\u200d\u2060\ufeff"), None)  # soft hyphen, zero-width chars
_SPACES = re.compile(r"[ \t\r\f\v\u00a0\u2000-\u200a\u202f\u205f\u3000]+")


def normalize_text(text: str) -> str:
    """Unicode-normalise (NFC), expand ligatures, drop invisible characters, collapse runs of spaces.

    Newlines are kept: they carry paragraph structure for the sentence splitter."""
    text = unicodedata.normalize("NFC", text)
    for src, dst in _LIGATURES.items():
        text = text.replace(src, dst)
    text = text.translate(_INVISIBLE)
    text = _SPACES.sub(" ", text)
    text = re.sub(r" *\n *", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def collapse_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# --------------------------------------------------------------------------- sentence splitting
_ABBREVIATIONS = frozenset(
    "mr mrs ms dr prof sr jr st vs etc inc ltd co corp plc llc no nos fig figs eq eqs approx est dept dist vol vols "
    "ch sec cf al ed eds pp ca e.g i.e u.s u.k a.m p.m jan feb mar apr jun jul aug sep sept oct nov dec".split()
)
_CORPORATE = frozenset({"inc", "ltd", "co", "corp", "plc", "llc"})
_SENTENCE_STARTERS = frozenset(
    "the it its they their we our he she this that these those a an in on at as for from after before during since "
    "however also but and or if when while although there here".split()
)
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")
_LAST_WORD = re.compile(r"([A-Za-z][A-Za-z.]*)\.?[\"')\]]*$")


def _is_false_boundary(prev: str, nxt: str) -> bool:
    """True when the full stop after `prev` belongs to an abbreviation or an initial, not a sentence end."""
    if not prev.endswith("."):
        return False  # ! and ? are real boundaries
    m = _LAST_WORD.search(prev.rstrip(".") + ".")
    if not m:
        return False
    word = m.group(1).rstrip(".").lower()
    first_next = re.match(r"[A-Za-z]+", nxt)
    if word in _CORPORATE:  # "...Northstar Ltd. The company..." is a real boundary
        return not (first_next and first_next.group(0).lower() in _SENTENCE_STARTERS)
    if word in _ABBREVIATIONS:
        return True
    if len(word) == 1 and word.isalpha() and prev[-2:-1] != word.lower() and word.upper() == m.group(1).rstrip(".")[-1:]:
        return True  # single-letter initial such as "J. Smith"
    return False


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Newlines always end a sentence (they mark paragraph/heading/table-row units)."""
    out: list[str] = []
    for line in normalize_text(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        start = 0
        pieces: list[str] = []
        for m in _BOUNDARY.finditer(line):
            prev, nxt = line[start : m.start()], line[m.end() :]
            if _is_false_boundary(prev, nxt):
                continue
            pieces.append(line[start : m.start()].strip())
            start = m.end()
        pieces.append(line[start:].strip())
        out.extend(p for p in pieces if p)
    return out


# --------------------------------------------------------------------------- tokens, stems, canonical terms
STOPWORDS = frozenset(
    "a an the and or of to in on for with by is are was were be been being it its this that these those as at from "
    "what which who whom whose how when where why do does did can could should would will shall may might must "
    "has have had having not no than then there their they them he she his her we our you your i me my also about "
    "into over under per via such if so but because while during after before between among within without "
    "any all each both either neither some more most other another".split()
)
_TOKEN = re.compile(r"[a-z0-9]+")
_NUMERIC_TOKEN = re.compile(r"^\d+$")


def stem(word: str) -> str:
    """A tiny suffix stripper. It only has to be consistent between questions and documents."""
    w = word.lower()
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("sses"):
        w = w[:-2]
    elif w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ied") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith("ing") and len(w) > 5:
        w = w[:-3]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("ed") and len(w) > 4:
        w = w[:-2]
        if len(w) > 2 and w[-1] == w[-2] and w[-1] not in "lsz":
            w = w[:-1]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if len(w) > 4 and w.endswith("e"):
        w = w[:-1]
    return w


# Groups of words that usually mean the same thing in business, technical and academic prose.
_SYNONYM_GROUPS: list[tuple[str, ...]] = [
    ("open", "start", "begin", "began", "begun", "commence", "launch", "inaugurate", "establish", "found", "operate", "initiate"),
    ("close", "end", "finish", "terminate", "conclude", "cease", "shut"),
    ("increase", "rise", "rose", "risen", "grow", "grew", "grown", "gain", "climb", "expand", "surge", "improve", "higher", "jump", "soar"),
    ("decrease", "fall", "fell", "fallen", "drop", "decline", "reduce", "shrink", "shrank", "shrunk", "lower", "cut", "diminish", "sink", "sank", "sunk", "slump", "plunge"),
    ("revenue", "sales", "turnover"),
    ("cost", "expense", "expenditure", "spending"),
    ("profit", "earning", "income", "surplus"),
    ("employee", "staff", "worker", "personnel", "headcount", "workforce", "employ", "people", "person", "individual"),
    ("buy", "bought", "acquire", "purchase", "procure"),
    ("build", "built", "construct", "assemble", "manufacture", "produce"),
    ("locate", "base", "situate", "headquarter", "seat"),
    ("deadline", "due", "cutoff"),
    ("require", "need", "demand", "mandate"),
    ("show", "demonstrate", "indicate", "reveal", "display"),
    ("use", "utilize", "utilise", "employ", "apply"),
    ("large", "big", "huge", "substantial", "major"),
    ("small", "little", "minor", "tiny"),
    ("error", "mistake", "fault", "defect", "bug"),
    ("method", "approach", "technique", "procedure"),
    ("result", "outcome", "finding"),
    ("goal", "objective", "aim", "target"),
    ("risk", "hazard", "threat"),
    ("price", "fee", "charge", "tariff"),
    ("maintain", "maintenance", "service", "upkeep"),
    ("approve", "authorize", "authorise", "sanction", "permit"),
    ("mark", "grade", "score", "point"),
    ("submit", "hand", "deliver", "turn"),
    ("percent", "percentage", "pct"),
    ("year", "annual", "yearly", "annum"),
    ("company", "firm", "business", "corporation", "organisation", "organization"),
    ("office", "branch", "site"),
]
_CANON: dict[str, str] = {}
for _group in _SYNONYM_GROUPS:
    _head = stem(_group[0])
    for _word in _group:
        _CANON.setdefault(stem(_word), _head)

_DIRECTION_UP = frozenset(stem(w) for w in "increase rise rose risen grow grew grown gain climb surge improve higher up expand jump soar".split())
_DIRECTION_DOWN = frozenset(stem(w) for w in "decrease fall fell fallen drop decline reduce shrink shrank shrunk lower down cut diminish slump plunge sink sank sunk".split())


def canon(token: str) -> str:
    s = stem(token)
    return _CANON.get(s, s)


def tokenize(text: str) -> list[str]:
    """Lower-case alphanumeric tokens (possessive 's removed)."""
    return _TOKEN.findall(re.sub(r"['\u2019]s\b", "", text.lower()))


def content_terms(text: str, *, keep_numbers: bool = False) -> list[str]:
    """Canonical stems of the informative words in `text` (stop words, and by default numbers, removed)."""
    out = []
    for t in tokenize(text):
        if t in STOPWORDS:
            continue
        if _NUMERIC_TOKEN.match(t) and not keep_numbers:
            continue
        out.append(canon(t))
    return out


def direction_of(text: str) -> set[str]:
    """Which of {'up','down'} direction verbs appear in `text` (used to catch 'rose' vs 'fell' with equal numbers)."""
    stems = {stem(t) for t in tokenize(text)}
    found = set()
    if stems & _DIRECTION_UP:
        found.add("up")
    if stems & _DIRECTION_DOWN:
        found.add("down")
    return found


# --------------------------------------------------------------------------- entity anchors
_MONTHS = frozenset("january february march april may june july august september october november december "
                    "jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday".split())
_COMMON_STARTS = frozenset(
    "the a an this that these those in on at it its they their there here however also after before during since according "
    "as for from if when while although but and or we our he she his her what which who how why where".split()
)
_CAP_WORD = re.compile(r"\b([A-Z][A-Za-z0-9\-]*|[A-Z0-9]{2,}|[A-Za-z]+\d+[A-Za-z0-9]*)\b")


def entity_anchors(text: str, df: Mapping[str, float] | None = None, rare_df: float = 0.1) -> list[str]:
    """Proper-noun-like tokens (canonical) whose absence from a passage means the passage is about something else.

    A capitalised word counts when it is not sentence-initial, is an acronym/identifier, or - for a sentence-initial
    word - when `df` (document frequency per canonical term) shows it is rare in the document."""
    anchors: list[str] = []
    for sentence in split_sentences(text):
        for m in _CAP_WORD.finditer(sentence):
            word = m.group(1)
            low = word.lower()
            if low in STOPWORDS or low in _MONTHS or low in _COMMON_STARTS or word.isdigit():
                continue  # numbers are compared by the typed quantity checker, never used as entity anchors
            initial = m.start() == 0
            acronym = word.isupper() and len(word) >= 2
            has_digit = any(c.isdigit() for c in word)
            if re.fullmatch(r"Q[1-4]|H[12]|FY\d{2,4}", word):
                continue  # period labels, not entities
            if initial and not (acronym or has_digit):
                if df is None or df.get(canon(low), 0.0) > rare_df:
                    continue
            anchors.append(canon(low))
    return list(dict.fromkeys(anchors))


def term_document_frequency(texts: Iterable[str], min_docs: int = 6) -> dict[str, float]:
    """Fraction of `texts` (e.g. chunks) that contain each canonical term.

    Returns {} (= "unknown") when there are fewer than `min_docs` texts: in a tiny document every term appears in most
    chunks and the statistic would only mislead the relevance scoring."""
    counts: dict[str, int] = {}
    n = 0
    for text in texts:
        n += 1
        for term in set(content_terms(text)):
            counts[term] = counts.get(term, 0) + 1
    return {t: c / n for t, c in counts.items()} if n >= min_docs else {}


STARTS_WITH_REFERENCE = re.compile(
    r"^(it|its|they|their|them|this|these|those|he|she|his|her|the former|the latter|such|that|both)\b", re.IGNORECASE
)
