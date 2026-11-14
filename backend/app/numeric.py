"""Deterministic comparison of the quantities in a claim with those in an evidence passage.

A quantity is typed: percent, money (with currency), number (with optional unit/dimension), year or date.
Typing is what prevents the classic mistakes: "82 million euros" is not "82 million dollars", "50%" is not
"50 units", "12.5%" is not "125%", and a number from an unrelated sentence is never compared at all
(the caller only compares a claim with evidence sentences that are relevant to it).

Decision rules (conservative by design):
  match     the evidence contains an equal quantity of a compatible type (rounding of the claim is tolerated)
  conflict  the evidence contains exactly ONE comparable quantity and it differs (or has another currency/unit)
  missing   anything else - the claim states something the evidence does not state

Limits: digits and simple number words only; no arithmetic inference (a percentage is not derived from two
amounts); borderless tables are not understood here (the PDF layer linearises tables into sentences first).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from .textutils import STOPWORDS, canon, normalize_text

# ----------------------------------------------------------------------------- vocabulary
_CUR_SYMBOL = {"$": "USD", "us$": "USD", "c$": "CAD", "a$": "AUD", "s$": "SGD", "€": "EUR", "£": "GBP", "¥": "JPY", "₹": "INR"}
_CUR_CODES = {c.lower(): c for c in "USD EUR GBP JPY INR CHF CAD AUD CNY SGD AED SEK NOK DKK".split()} | {"rmb": "CNY"}
_CUR_WORDS = {"dollar": "USD", "euro": "EUR", "sterling": "GBP", "yen": "JPY", "rupee": "INR", "franc": "CHF", "yuan": "CNY", "renminbi": "CNY"}
_CUR_ADJ = {"us": "USD", "canadian": "CAD", "australian": "AUD", "singapore": "SGD", "hong kong": "HKD", "nz": "NZD", "new zealand": "NZD"}
_MAGNITUDE = {"hundred": 1e2, "thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9,
              "trillion": 1e12, "tn": 1e12, "lakh": 1e5, "crore": 1e7}
_SHORT_MAG = {"k", "m", "b", "mn", "bn", "tn"}
# unit -> (dimension, factor to the dimension's base unit)
_UNITS = {
    "g": ("mass", 1e-3), "gram": ("mass", 1e-3), "kg": ("mass", 1.0), "kilogram": ("mass", 1.0), "tonne": ("mass", 1000.0),
    "lb": ("mass", 0.45359237), "pound": ("mass", 0.45359237), "oz": ("mass", 0.0283495), "ounce": ("mass", 0.0283495),
    "mm": ("length", 1e-3), "cm": ("length", 1e-2), "m": ("length", 1.0), "meter": ("length", 1.0), "metre": ("length", 1.0),
    "km": ("length", 1e3), "kilometer": ("length", 1e3), "kilometre": ("length", 1e3), "mile": ("length", 1609.344),
    "ft": ("length", 0.3048), "foot": ("length", 0.3048), "feet": ("length", 0.3048), "inch": ("length", 0.0254),
    "second": ("time", 1.0), "sec": ("time", 1.0), "minute": ("time", 60.0), "min": ("time", 60.0), "hour": ("time", 3600.0),
    "hr": ("time", 3600.0), "day": ("time", 86400.0), "week": ("time", 604800.0),
    "month": ("calendar", 1.0), "year": ("calendar", 12.0), "yr": ("calendar", 12.0),
}
_MONTHS = {m: i for i, m in enumerate(
    "january february march april may june july august september october november december".split(), start=1)}
_MONTHS.update({"jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12})
_MONTH_RE = r"(?:january|february|march|april|may|june|july|august|september|october|november|december|jan|feb|mar|apr|jun|jul|aug|sept|sep|oct|nov|dec)"
_ID_WORDS = frozenset("section chapter figure fig table page pages p pp step item no number version v rev clause article appendix slide "
                      "equation eq ref exhibit paragraph para line lines row column col id issue volume vol part".split())
_WORD_NUMS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_WORD_NUMS.update({"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90})
_YEAR_PREP = frozenset("in by since from between until till during of year fiscal calendar to and through before after early late mid".split())
_DETERMINERS = frozenset("the a an its our their his her this that s".split())
_NOT_UNIT = STOPWORDS | set(_MONTHS) | {"percent", "per", "cent", "approximately", "about", "around", "roughly", "nearly", "over", "up", "down"}

# ----------------------------------------------------------------------------- regular expressions
_NUMBER = r"(?:\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d{1,2}(?:,\d{2})+,\d{3}(?:\.\d+)?|\d+(?:[.,]\d+)?)"
_CUR_SYM_RE = r"(?:US\$|C\$|A\$|S\$|[$€£¥₹])"
_CUR_CODE_RE = r"(?:USD|EUR|GBP|JPY|INR|CHF|CAD|AUD|CNY|RMB|SGD|AED|SEK|NOK|DKK)"
_CUR_WORD_RE = r"(?:(?:US|Canadian|Australian|Singapore|Hong Kong|NZ|New Zealand)\s+)?(?:dollars?|euros?|sterling|yen|rupees?|francs?|yuan|renminbi)"
_MAG_RE = r"(?:thousand|million|billion|trillion|lakh|crore|hundred|mn|bn|tn|k|m|b)"
_PCT_RE = r"(?:%|percentage\s+points?|percent|per\s?cent|pct\b|pp\b|basis\s+points?|bps\b)"
_QTY = re.compile(
    rf"(?<![\w.])(?<![A-Za-z]-)(?P<neg>[-\u2212\u2013])?(?P<sym>{_CUR_SYM_RE}|{_CUR_CODE_RE}(?=\s?\d))?\s?(?P<num>{_NUMBER})"
    rf"(?:\s?(?P<mag>{_MAG_RE})\b(?!-?\w))?"
    rf"(?:\s?(?P<pct>{_PCT_RE})|\s+(?P<cword>{_CUR_WORD_RE})\b|\s?(?P<ccode>{_CUR_CODE_RE})\b)?"
    rf"(?:(?P<gap>\s?)(?P<unit>[A-Za-z][A-Za-z\-]{{0,20}}))?",
    re.IGNORECASE,
)
_DATE_PATTERNS = [
    (re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH_RE})\.?,?\s+(?P<y>\d{{4}})\b", re.I), "dmy"),
    (re.compile(rf"\b(?P<m>{_MONTH_RE})\.?\s+(?P<d>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<y>\d{{4}})\b", re.I), "mdy"),
    (re.compile(r"\b(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})\b"), "iso"),
    (re.compile(r"\b(?P<a>\d{1,2})[/.](?P<b>\d{1,2})[/.](?P<y>\d{4})\b"), "slash"),
    (re.compile(rf"\b(?P<m>{_MONTH_RE})\.?,?\s+(?P<y>\d{{4}})\b", re.I), "my"),
    (re.compile(rf"\b(?P<d>\d{{1,2}})(?:st|nd|rd|th)?\s+(?:of\s+)?(?P<m>{_MONTH_RE})\b(?!\.?,?\s+\d{{4}})", re.I), "dm"),
]
_CODE = re.compile(r"(?<![\w.])(?:\d+(?:\.\d+){2,}|(?:version|ver\.?|v)\s?\d+(?:\.\d+)*)(?![\w])", re.I)
_PERIOD_LABEL = re.compile(r"\b(?:Q[1-4]|H[12]|FY\s?\d{2,4}|CY\s?\d{2,4})\b", re.I)
_WORDNUM_RE = re.compile(
    r"\b(?P<w>(?:(?:twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(?:[- ](?:one|two|three|four|five|six|seven|eight|nine))?"
    r"|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|dozen))"
    r"(?:\s+(?P<mag>hundred|thousand|million|billion))?(?:\s+(?P<unit>[A-Za-z][A-Za-z\-]{2,20}))?\b",
    re.I,
)
_DateT = tuple[int | None, int | None, int | None]


@dataclass(frozen=True)
class Quantity:
    kind: str  # percent | pp | bps | money | number | year | date | code (version numbers, IP addresses: compared exactly)
    value: float
    raw: str
    unit: str = ""  # currency code for money; singular unit noun for numbers
    dim: str = ""  # physical dimension when the unit is a known measure
    base: float = 0.0  # value converted to the dimension's base unit
    decimals: int = 0
    scale: float = 1.0  # magnitude multiplier already applied to `value`
    date: _DateT | None = None  # (year, month, day)
    alt_dates: tuple[_DateT, ...] = ()
    endpoint: bool = False  # one end of a range ("2020 to 2023", "10-15 million")
    start: int = 0
    end: int = 0


def _singular(word: str) -> str:
    w = word.lower().strip("-")
    for suffix, repl in (("ies", "y"), ("ses", "s"), ("s", "")):
        if w.endswith(suffix) and len(w) > len(suffix) + 2 and not w.endswith("ss"):
            return w[: -len(suffix)] + repl
    return w


def _to_float(num: str) -> tuple[float, int]:
    s = num
    if re.fullmatch(r"\d+,\d{1,2}", s):  # decimal comma: 12,5
        s = s.replace(",", ".")
    s = s.replace(",", "")
    decimals = len(s.split(".")[1]) if "." in s else 0
    return float(s), decimals


def _mask(text: str, start: int, end: int) -> str:
    return text[:start] + " " * (end - start) + text[end:]


def _extract_dates(text: str) -> tuple[list[Quantity], str]:
    found: list[Quantity] = []
    for pattern, style in _DATE_PATTERNS:
        for m in list(pattern.finditer(text)):
            if text[m.start() : m.end()].strip() == "":
                continue  # already consumed by an earlier pattern
            g = m.groupdict()
            alts: tuple[_DateT, ...] = ()
            y = int(g["y"]) if g.get("y") else None
            if g.get("m") is None:
                mo = None
            elif g["m"].isdigit():
                mo = int(g["m"])
            else:
                mo = _MONTHS.get(g["m"].lower().rstrip("."))
            d = int(g["d"]) if g.get("d") else None
            if style == "slash":
                a, b = int(g["a"]), int(g["b"])
                if a > 12:
                    d, mo = a, b
                elif b > 12:
                    d, mo = b, a
                else:  # ambiguous: day/month or month/day - accept either reading
                    d, mo, alts = a, b, ((y, a, b),)
            if (mo is not None and not 1 <= mo <= 12) or (d is not None and not 1 <= d <= 31) or (y is not None and not 1000 <= y <= 2200):
                continue
            found.append(Quantity("date", float((y or 0) * 10000 + (mo or 0) * 100 + (d or 0)), m.group(0), date=(y, mo, d),
                                  alt_dates=alts, start=m.start(), end=m.end()))
            text = _mask(text, m.start(), m.end())
    return found, text


def _currency_from(sym: str | None, cword: str | None, ccode: str | None) -> str:
    if sym:
        return _CUR_SYMBOL.get(sym.lower()) or _CUR_CODES[sym.lower()]
    if ccode:
        return _CUR_CODES[ccode.lower()]
    if cword:
        parts = cword.lower().split()
        base = _CUR_WORDS.get(_singular(parts[-1]), "")
        if len(parts) > 1 and base == "USD":
            return _CUR_ADJ.get(" ".join(parts[:-1]), base)
        return base
    return ""


def extract_quantities(text: str) -> list[Quantity]:
    """All typed quantities in `text`, in order of appearance."""
    text = normalize_text(text).replace("\n", " ")
    codes: list[Quantity] = []
    for m in _CODE.finditer(text):
        norm = re.sub(r"^(?:version|ver\.?|v)\s?", "", m.group(0), flags=re.I).strip().lower()
        codes.append(Quantity("code", 0.0, m.group(0), unit=norm, start=m.start(), end=m.end()))
        text = _mask(text, m.start(), m.end())
    dates, masked = _extract_dates(text)
    for m in _PERIOD_LABEL.finditer(masked):
        masked = _mask(masked, m.start(), m.end())
    out: list[Quantity] = [*codes, *dates]

    for m in _QTY.finditer(masked):
        g = m.groupdict()
        prev_words = re.findall(r"[A-Za-z]+", masked[max(0, m.start() - 14) : m.start()])
        if prev_words and prev_words[-1].lower() in _ID_WORDS and not g["sym"]:
            continue  # "Table 2", "page 4", "v2.1" are identifiers, not quantities
        value, decimals = _to_float(g["num"])
        scale = 1.0
        mag, unit_word, gap = g["mag"], g["unit"], g["gap"]
        currency = _currency_from(g["sym"], g["cword"], g["ccode"])
        if mag and mag.lower() in _SHORT_MAG and not currency and not g["pct"]:
            # "5 m" is metres but "5m" is five million: accept a short magnitude only when attached or with a currency.
            attached = masked[m.end("num") : m.end("num") + len(mag)].lower() == mag.lower()
            if not attached:
                unit_word, mag = mag, None
        if mag:
            scale = _MAGNITUDE[mag.lower()]
        if g["neg"] and not (m.start() > 0 and masked[m.start() - 1].isdigit()):
            value = -value
        value *= scale
        if unit_word and not gap and unit_word.lower() in {"st", "nd", "rd", "th"}:
            continue  # ordinals ("3rd") are not quantities
        end_num = m.end("mag") if mag else m.end("num")
        pct = (g["pct"] or "").lower()
        if pct:
            if pct.startswith(("basis", "bps")):
                kind = "bps"
            elif pct.startswith("percentage") or pct == "pp":
                kind = "pp"
            else:
                kind = "percent"
            out.append(Quantity(kind, value, masked[m.start() : m.end("pct")].strip(), decimals=decimals, scale=scale, start=m.start(), end=m.end("pct")))
            continue
        if not currency and unit_word and _singular(unit_word) in _CUR_WORDS:
            currency, unit_word = _CUR_WORDS[_singular(unit_word)], None
            end_num = m.end("unit")
        if currency:
            end = m.end("cword") if g["cword"] else m.end("ccode") if g["ccode"] else end_num
            out.append(Quantity("money", value, masked[m.start() : end].strip(), unit=currency, decimals=decimals, scale=scale, start=m.start(), end=end))
            continue
        unit = _singular(unit_word) if unit_word and unit_word.lower() not in _NOT_UNIT else ""
        end = m.end("unit") if unit else end_num
        year_shaped = not mag and decimals == 0 and "," not in g["num"] and 1000 <= value <= 2100 and not g["neg"]
        prev_word = prev_words[-1].lower() if prev_words else ""
        plural_noun = bool(unit_word) and unit_word.lower().endswith("s") and not unit_word.lower().endswith("ss")
        if year_shaped and (not unit or (unit not in _UNITS and (prev_word in _YEAR_PREP or prev_word in _DETERMINERS or not plural_noun))):
            out.append(Quantity("year", value, g["num"], start=m.start(), end=m.end("num")))
            continue
        dim, factor = _UNITS.get(unit, ("", 1.0))
        out.append(Quantity("number", value, masked[m.start() : end].strip(), unit=unit, dim=dim, base=value * factor if dim else value,
                            decimals=decimals, scale=scale, start=m.start(), end=end))

    for m in _WORDNUM_RE.finditer(masked):
        words = m.group("w").lower().replace("-", " ").split()
        value = 12.0 if words[0] == "dozen" else float(sum(_WORD_NUMS[x] for x in words))
        scale = _MAGNITUDE[m.group("mag").lower()] if m.group("mag") else 1.0
        unit_word = m.group("unit")
        unit = _singular(unit_word) if unit_word and unit_word.lower() not in _NOT_UNIT else ""
        if not unit and not m.group("mag"):
            continue  # a bare "three" is too ambiguous to compare
        dim, factor = _UNITS.get(unit, ("", 1.0))
        out.append(Quantity("number", value * scale, m.group(0), unit=unit, dim=dim, base=value * scale * factor if dim else value * scale,
                            scale=scale, start=m.start(), end=m.end()))

    out.sort(key=lambda q: q.start)
    return _mark_ranges(out, masked)


_RANGE_SEP = re.compile(r"^\s*(?:-|\u2013|\u2014|to|and|through|until)\s*$", re.I)


def _mark_ranges(qs: list[Quantity], text: str) -> list[Quantity]:
    """Flag the two ends of "between A and B" / "A to B" / "A-B" so a value between them is not a conflict."""
    out = list(qs)
    for i in range(len(out) - 1):
        a, b = out[i], out[i + 1]
        if a.kind != b.kind or a.kind not in {"year", "number", "money", "percent"}:
            continue
        gap = text[a.end : b.start] if b.start >= a.end else ""
        before = text[max(0, a.start - 12) : a.start].lower()
        if _RANGE_SEP.match(gap) and (gap.strip() in {"-", "\u2013", "\u2014"} or re.search(r"(between|from)\s*$", before)):
            if a.kind in {"number", "money"} and a.scale == 1.0 and b.scale != 1.0:
                a = replace(a, value=a.value * b.scale, base=a.base * b.scale, scale=b.scale, unit=a.unit or b.unit)
            out[i] = replace(a, endpoint=True)
            out[i + 1] = replace(b, endpoint=True)
    return out


# ----------------------------------------------------------------------------- comparison
@dataclass
class QuantityCheck:
    matched: list[tuple[Quantity, Quantity]] = field(default_factory=list)
    conflicts: list[tuple[Quantity, Quantity, str]] = field(default_factory=list)  # (claim, evidence, why)
    missing: list[Quantity] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    period_mismatch: bool = False  # the claim names a period/date this sentence does not state: it may be about another period

    @property
    def ok(self) -> bool:
        return not self.conflicts and not self.missing

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    def describe_conflict(self) -> str:
        c, e, why = self.conflicts[0]
        return f"the claim says '{c.raw}' but the evidence says '{e.raw}' ({why})"


def _equalish(claim: Quantity, ev: Quantity, use_base: bool) -> tuple[bool, bool]:
    """(equal, rounded). A claim may be a rounded form of a more precise evidence value, never the reverse."""
    a, b = (claim.base, ev.base) if use_base else (claim.value, ev.value)
    if abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b)):
        return True, False
    if ev.decimals > claim.decimals:
        half = 0.5 * 10 ** (-claim.decimals) * (claim.scale or 1.0)
        if use_base and claim.value:
            half *= claim.base / claim.value
        if abs(a - b) <= half * (1 + 1e-9):
            return True, True
    return False, False


def _years_of(q: Quantity) -> set[int]:
    if q.kind == "year":
        return {int(q.value)}
    if q.kind == "date" and q.date and q.date[0]:
        return {q.date[0]}
    return set()


def _date_relation(c: _DateT, e: _DateT) -> str:
    """match | less_specific | conflict for (year, month, day) tuples."""
    for cf, ef in zip(c, e):
        if cf is not None and ef is not None and cf != ef:
            return "conflict"
    if any(cf is not None and ef is None for cf, ef in zip(c, e)):
        return "less_specific"
    return "match"


def _compare_dates(c: Quantity, evs: list[Quantity]) -> tuple[str, Quantity | None, str]:
    points = [e for e in evs if e.kind in {"year", "date"}]
    claim_dates: list[_DateT] = [c.date, *c.alt_dates] if c.kind == "date" and c.date else []
    less_specific: Quantity | None = None
    for e in points:
        if c.kind == "year":
            if int(c.value) in _years_of(e):
                return "match", e, ""
            continue
        ev_dates: list[_DateT] = [e.date, *e.alt_dates] if e.kind == "date" and e.date else [(int(e.value), None, None)]
        for cd in claim_dates:
            for ed in ev_dates:
                rel = _date_relation(cd, ed)
                if rel == "match":
                    return "match", e, ""
                if rel == "less_specific":
                    less_specific = e
    if less_specific is not None:
        return "missing", less_specific, "the evidence gives a less precise date than the claim"
    ends = sorted(int(e.value) for e in points if e.kind == "year" and e.endpoint)
    if c.kind == "year" and len(ends) >= 2 and ends[0] < c.value < ends[-1]:
        return "within", None, "inside a stated period"
    singles = [e for e in points if not e.endpoint]
    distinct = {y for e in singles for y in _years_of(e)}
    if len(distinct) == 1 and singles:  # exactly one date in the evidence and it differs: a real conflict
        return "conflict", singles[0], "different date"
    return "missing", None, ""


def _comparable(c: Quantity, e: Quantity) -> bool:
    if c.kind in {"percent", "pp", "bps"}:
        return e.kind == c.kind
    if c.kind == "money":
        return e.kind == "money"
    if e.kind == "money":  # a claim number without currency may be satisfied by a money amount
        return not c.unit and not c.dim
    if e.kind != "number":
        return False
    if c.dim or e.dim:
        return c.dim == e.dim
    return not c.unit or not e.unit or canon(c.unit) == canon(e.unit)


def _compare_one(c: Quantity, evs: list[Quantity]) -> tuple[str, Quantity | None, str]:
    """Return (match | conflict | missing | within, evidence quantity, note)."""
    if c.kind in {"year", "date"}:
        return _compare_dates(c, evs)
    if c.kind == "code":
        codes = [e for e in evs if e.kind == "code"]
        if any(e.unit == c.unit for e in codes):
            return "match", next(e for e in codes if e.unit == c.unit), ""
        if len(codes) == 1:
            return "conflict", codes[0], "different version or address"
        return "missing", None, ""
    candidates = [e for e in evs if _comparable(c, e)]
    for e in candidates:
        if c.kind == "money" and e.unit != c.unit:
            continue
        eq, rounded = _equalish(c, e, use_base=bool(c.dim and e.dim == c.dim))
        if eq:
            return "match", e, "rounded" if rounded else ""
    if c.kind == "money":
        same_value_other_currency = [e for e in candidates if e.unit != c.unit and abs(e.value - c.value) <= 1e-9 * max(1.0, abs(c.value))]
        pool = [e for e in candidates if e.unit == c.unit]
        if same_value_other_currency and not pool:
            return "conflict", same_value_other_currency[0], f"currency differs: {c.unit} vs {same_value_other_currency[0].unit}"
    else:
        pool = candidates
    if c.kind == "money":
        pool = [e for e in candidates if e.unit == c.unit]
    singles = [e for e in pool if not e.endpoint]
    if len(pool) == 1 and not pool[0].endpoint:
        return "conflict", pool[0], "different value"
    if len(pool) >= 2 and len(singles) == 0:
        values = sorted(e.value for e in pool)
        if values[0] < c.value < values[-1]:
            return "within", None, "inside a stated range"
    return "missing", None, ""


def compare_quantities(claim: str, evidence: str, context: str = "") -> QuantityCheck:
    """Compare every quantity in `claim` with the quantities in `evidence` (one relevant sentence).

    `context` (the surrounding passage and heading) is consulted ONLY for years, so a period stated once in a
    heading or an earlier sentence can cover a claim without the claim being rejected for repeating it.

    Period rule: when the claim names a period (year/date) that this sentence does not itself state, a differing
    amount or percentage is NOT treated as a conflict - it may simply belong to another period ("revenue in 2024"
    versus a sentence about 2023). Such claims end up as 'missing', i.e. NOT_ESTABLISHED, never CONTRADICTED."""
    result = QuantityCheck()
    ev_qs = extract_quantities(evidence)
    ctx_years: set[int] = set()
    if context:
        for q in extract_quantities(context):
            ctx_years |= _years_of(q)
    claim_qs = extract_quantities(claim)
    period_unmatched = False
    for c in claim_qs:
        verdict, ev, note = _compare_one(c, ev_qs)
        is_date = c.kind in {"year", "date"}
        if verdict == "match" and ev is not None:
            result.matched.append((c, ev))
            if note == "rounded":
                result.notes.append(f"'{c.raw}' is a rounded form of '{ev.raw}'")
        elif verdict == "within":
            result.notes.append(f"'{c.raw}' lies {note}")
        elif verdict == "conflict" and ev is not None:
            result.conflicts.append((c, ev, note))
            period_unmatched = period_unmatched or is_date
        elif (c.kind == "year" and int(c.value) in ctx_years) or (c.kind == "date" and c.date and c.date[1] is None and c.date[0] in ctx_years):
            result.notes.append(f"'{c.raw}' is stated in the surrounding text")
            period_unmatched = True  # not in this sentence itself
        else:
            result.missing.append(c)
            period_unmatched = period_unmatched or is_date
    has_amounts = any(c.kind not in {"year", "date"} for c in claim_qs)
    result.period_mismatch = has_amounts and period_unmatched
    if has_amounts and period_unmatched and result.conflicts:
        for c, e, _why in result.conflicts:
            result.missing.append(c)
            result.notes.append(f"'{c.raw}' differs from '{e.raw}', but the periods may differ, so this is not treated as a contradiction")
        result.conflicts = []
    return result
