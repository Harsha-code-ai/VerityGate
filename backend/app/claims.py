"""Parse the model's answer into checkable claims and read its [S#] citations.

Preferred format is strict JSON (``{"insufficient": false, "claims": [{"text": ..., "sources": ["S1"]}]}``):
the model itself splits the answer into self-contained, atomic statements. If the reply is not valid JSON
(small local models often ignore format instructions) a sentence-based fallback is used.

A claim must be able to stand alone. Fragments such as "15 June 2019." or "45 million dollars." are flagged
as incomplete instead of being verified - they cannot be checked independently and would only create noise."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .numeric import extract_quantities
from .textutils import collapse_ws, content_terms, split_sentences, stem

INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
MAX_CLAIMS = 12
MAX_CLAIM_CHARS = 500

_CITE = re.compile(r"\[\s*(S\d+(?:\s*[,;]\s*S?\d+)*)\s*\]", re.IGNORECASE)
_SID = re.compile(r"^S?(\d+)$", re.IGNORECASE)

_AUX = frozenset("is are was were be been being am has have had having do does did will would can could should shall may might must "
                 "became become becomes began begun grew grown rose risen fell fallen led held built made took taken ran set cut paid sold bought won lost "
                 "saw seen got gave given kept left met sent spent stood told wrote written came went drove shows showed shown says said".split())
_VERB_STEMS = frozenset(stem(w) for w in (
    "employ open start begin launch operate establish found create produce build manufacture sell buy acquire purchase increase decrease "
    "rise fall grow decline reduce raise report state show indicate demonstrate require need use apply provide offer include contain cover "
    "support approve announce plan expect aim target improve achieve generate earn cost price charge pay owe lead manage own run hold "
    "locate base house weigh measure take make give keep submit deliver complete finish end close stop continue maintain store ship "
    "serve host hire cut save spend invest fund lose win suffer face reach exceed total average remain stay become depend allow prevent "
    "describe explain define list note mention recommend suggest conclude find observe propose compare test evaluate train score submit "
    "weigh produce house stand sit live work move transfer enter exit access store reside contribute represent account result mean").split())


@dataclass
class ParsedClaim:
    text: str
    cited: list[str] = field(default_factory=list)  # e.g. ['S1', 'S3']
    problem: str | None = None  # set when the claim cannot be verified as written (see completeness_problem)


@dataclass
class ParsedAnswer:
    claims: list[ParsedClaim]
    insufficient: bool
    fmt: str  # "json" | "text" | "empty"


def _norm_ids(values) -> list[str]:
    out: list[str] = []
    for v in values if isinstance(values, list) else [values]:
        for token in re.split(r"[,;\s]+", str(v)):
            m = _SID.match(token.strip())
            if m:
                out.append(f"S{int(m.group(1))}")
    return list(dict.fromkeys(out))


_PRONOUN_START = re.compile(r"^(it|its|they|their|them|he|she|his|her)\b", re.IGNORECASE)
_DEMONSTRATIVE_PRONOUN = re.compile(r"^(this|that|these|those)\s+(is|are|was|were|has|have|had|will|would|can|could|shows?|means?)\b", re.IGNORECASE)


def completeness_problem(text: str) -> str | None:
    """Why `text` is not a self-contained factual statement, or None when it looks fine (heuristic).

    A checkable claim names its subject (no leading pronoun), says something about it (a verb) and is more than a bare
    number, date or name."""
    words = re.findall(r"[A-Za-z0-9$€£%][\w$€£%.,'’\-]*", text)
    if len(words) < 4:
        return "The statement is too short to check on its own."
    if _PRONOUN_START.match(text.strip()) or _DEMONSTRATIVE_PRONOUN.match(text.strip()):
        return "The statement starts with a pronoun instead of naming its subject, so it cannot be checked on its own."
    stripped = text
    for q in sorted(extract_quantities(text), key=lambda q: -q.start):
        if q.end > q.start:
            stripped = stripped[: q.start] + " " + stripped[q.end :]
    terms = [t for t in content_terms(stripped) if t not in {"dollar", "euro", "pound", "million", "billion", "thousand", "percent", "usd", "eur"}]
    if not terms:
        return "The statement contains only numbers or dates and no subject, so it cannot be checked on its own."
    tokens = [w.lower().strip(".,;:'’") for w in re.findall(r"[A-Za-z][A-Za-z'’\-]*", text)]
    if not any(t in _AUX or stem(t) in _VERB_STEMS or t.endswith(("ed", "ing")) for t in tokens) and len(set(terms)) < 6:
        return "The statement has no verb, so it is a fragment rather than a claim."
    return None


def _strip_fence(raw: str) -> str:
    raw = raw.strip()
    m = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else raw


def _from_json(raw: str) -> ParsedAnswer | None:
    body = _strip_fence(raw)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(body[start : end + 1])
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict) or "claims" not in data:
        return None
    items = data.get("claims")
    if not isinstance(items, list):
        return None
    claims: list[ParsedClaim] = []
    for item in items[:MAX_CLAIMS]:
        if isinstance(item, str):
            item = {"text": item, "sources": []}
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            continue
        text = collapse_ws(_CITE.sub("", item["text"]))[:MAX_CLAIM_CHARS]
        if text:
            cited = _norm_ids(item.get("sources") or item.get("cited") or []) + [i for m in _CITE.finditer(item["text"]) for i in _norm_ids(m.group(1))]
            claims.append(ParsedClaim(text, list(dict.fromkeys(cited)), completeness_problem(text)))
    insufficient = bool(data.get("insufficient")) and not claims
    return ParsedAnswer(claims, insufficient, "json")


def split_claims(answer: str) -> list[ParsedClaim]:
    """Fallback for plain-text answers: one claim per sentence (and per semicolon clause), with [S#] markers attached."""
    claims: list[ParsedClaim] = []
    answer = re.sub(r"([.!?])\s*(\[\s*S\d[^\]]*\])", r"\2\1", answer)  # "Sentence. [S1]" -> "Sentence[S1]."
    for line in answer.splitlines():
        line = re.sub(r"^\s*(?:[-*\u2022]|\d+[.)])\s+", "", line).strip()
        if not line:
            continue
        for sentence in split_sentences(line):
            for part in re.split(r";\s+", sentence):
                cited = [i for m in _CITE.finditer(part) for i in _norm_ids(m.group(1))]
                text = collapse_ws(_CITE.sub("", part)).replace(" .", ".").replace(" ,", ",")
                if text and len(text.split()) >= 2:
                    claims.append(ParsedClaim(text[:MAX_CLAIM_CHARS], list(dict.fromkeys(cited)), completeness_problem(text)))
    return claims[:MAX_CLAIMS]


def parse_answer(raw: str) -> ParsedAnswer:
    raw = (raw or "").strip()
    if not raw:
        return ParsedAnswer([], False, "empty")
    parsed = _from_json(raw)
    if parsed is not None:
        return parsed
    if raw.strip().strip(".").strip() == INSUFFICIENT:
        return ParsedAnswer([], True, "text")
    claims = split_claims(re.sub(rf"\b{INSUFFICIENT}\b\.?", "", raw))
    return ParsedAnswer(claims, not claims and INSUFFICIENT in raw, "text")
