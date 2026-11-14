"""PDF validation and structure-aware text extraction that preserves page numbers.

Extraction goes beyond "dump text": it keeps paragraphs, headings, list items and tables as separate units, orders
multi-column pages column by column, removes repeated headers/footers and page numbers, ignores hidden (white / tiny)
text, joins hyphenated line breaks, and OCRs only pages that have no text layer. Every unit belongs to exactly one
physical page; page numbers are the 1-based physical page index of the PDF file.

Never invented: text that extraction or OCR could not read is not guessed - unreadable pages are reported in
`ExtractionReport.warnings` and stay empty."""
from __future__ import annotations

import logging
import math
import os
import re
import statistics
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

from .textutils import normalize_text

log = logging.getLogger(__name__)


class PDFValidationError(ValueError):
    """The uploaded file is not an acceptable PDF."""


class ExtractionError(RuntimeError):
    """Text extraction produced nothing usable."""


@dataclass
class TextUnit:
    kind: str  # heading | para | list | table
    text: str


@dataclass
class PageText:
    page: int  # 1-based physical page number
    text: str
    units: list[TextUnit] = field(default_factory=list)
    ocr: bool = False


@dataclass
class ExtractionReport:
    pages: list[PageText]
    method: str
    warnings: list[str] = field(default_factory=list)
    ocr_pages: list[int] = field(default_factory=list)


def validate_pdf(data: bytes, max_bytes: int, max_pages: int) -> int:
    """Check the real file content (not the filename). Returns the page count."""
    if not data:
        raise PDFValidationError("The file is empty.")
    if len(data) > max_bytes:
        raise PDFValidationError(f"The file is larger than the {max_bytes // (1024 * 1024)} MB limit.")
    if not data[:1024].lstrip().startswith(b"%PDF-"):
        raise PDFValidationError("The file content is not a PDF (missing %PDF header).")
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:  # PyMuPDF raises several exception types
        raise PDFValidationError("The PDF could not be opened; it may be corrupted.") from exc
    try:
        if doc.needs_pass:
            raise PDFValidationError("Password-protected PDFs are not supported.")
        if doc.page_count == 0:
            raise PDFValidationError("The PDF has no pages.")
        if doc.page_count > max_pages:
            raise PDFValidationError(f"The PDF has {doc.page_count} pages; the limit is {max_pages}.")
        return doc.page_count
    finally:
        doc.close()


# ----------------------------------------------------------------------------- raw blocks
@dataclass
class _Line:
    text: str
    size: float
    bold: bool
    x0: float
    x1: float


@dataclass
class _Block:
    x0: float
    y0: float
    x1: float
    y1: float
    lines: list[_Line]
    page_height: float
    page_width: float

    @property
    def width(self) -> float:
        return self.x1 - self.x0

    @property
    def text(self) -> str:
        return " ".join(line.text for line in self.lines)


_PAGE_NUMBER = re.compile(r"^(?:page\s*)?(?:\d{1,4}|[ivxlc]{1,6})(?:\s*(?:of|/)\s*\d{1,4})?$|^[-\u2013\u2014]\s*\d{1,4}\s*[-\u2013\u2014]$", re.I)
_BULLET = re.compile(r"^\s*(?:[\u2022\u25aa\u25e6\u25cf\u2023*-]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")
_HEADING_TEXT = re.compile(r"^(?:\d+(?:\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:&'\u2019\-/()]{2,80}$")
_TERMINAL = (".", ",", ";", "?", "!")


def _rgb(color: int) -> tuple[float, float, float]:
    return ((color >> 16) & 255) / 255, ((color >> 8) & 255) / 255, (color & 255) / 255


def _dark_fill_rects(page) -> list:
    rects = []
    try:
        for d in page.get_drawings()[:3000]:
            fill = d.get("fill")
            if fill and sum(fill[:3]) / 3 < 0.85:
                rects.append(d["rect"])
    except Exception:  # drawings are only used to avoid dropping legitimate white-on-dark text
        pass
    return rects


def _blocks_from_dict(page, textpage=None) -> tuple[list[_Block], int]:
    """Text blocks of one page. Returns (blocks, number of hidden spans that were ignored)."""
    data = page.get_text("dict", textpage=textpage) if textpage is not None else page.get_text("dict")
    pw, ph = float(page.rect.width), float(page.rect.height)
    dark = None
    hidden = 0
    blocks: list[_Block] = []
    for b in data.get("blocks", []):
        if b.get("type") != 0:
            continue
        lines: list[_Line] = []
        for ln in b.get("lines", []):
            parts, sizes, bolds = [], [], []
            for sp in ln.get("spans", []):
                text = sp.get("text", "")
                if not text.strip():
                    parts.append(text)
                    continue
                size = float(sp.get("size", 0))
                if size < 2.0:
                    hidden += 1
                    continue
                if min(_rgb(int(sp.get("color", 0)))) > 0.93:  # white text: hidden unless drawn on a dark background
                    if dark is None:
                        dark = _dark_fill_rects(page)
                    box = fitz.Rect(sp["bbox"])
                    if not any(box.intersects(r) for r in dark):
                        hidden += 1
                        continue
                parts.append(text)
                sizes.append((size, len(text)))
                bolds.append(bool(int(sp.get("flags", 0)) & 16) or "bold" in str(sp.get("font", "")).lower())
            text = normalize_text("".join(parts)).replace("\n", " ").strip()
            if text and sizes:
                total = sum(n for _, n in sizes)
                size = sum(s * n for s, n in sizes) / total
                x0, _, x1, _ = ln["bbox"]
                lines.append(_Line(text, size, all(bolds), float(x0), float(x1)))
        if lines:
            x0, y0, x1, y1 = b["bbox"]
            blocks.append(_Block(float(x0), float(y0), float(x1), float(y1), lines, ph, pw))
    return blocks, hidden


# ----------------------------------------------------------------------------- tables
def _tables(page) -> tuple[list[tuple[fitz.Rect, list[str]]], list[str]]:
    """Ruled tables as (bbox, row sentences). Borderless tables are not detected (stated limitation)."""
    found: list[tuple[fitz.Rect, list[str]]] = []
    try:
        finder = page.find_tables()
    except Exception as exc:
        log.debug("find_tables failed on page: %s", type(exc).__name__)
        return [], []
    for tab in getattr(finder, "tables", []):
        try:
            rows = tab.extract()
        except Exception:
            continue
        sentences = _linearise_table(rows)
        if sentences:
            found.append((fitz.Rect(tab.bbox), sentences))
    return found, []


def _clean_cell(cell) -> str:
    return re.sub(r"\s+", " ", normalize_text(str(cell or ""))).strip()


def _linearise_table(rows: list[list]) -> list[str]:
    """Turn a table into one sentence per row that keeps row name, column names and values together:
    'Revenue: 2023 is 45.0; 2022 is 40.0.' Returns [] for tables that are mostly empty or have a single row/column."""
    grid = [[_clean_cell(c) for c in row] for row in rows if row]
    if len(grid) < 2 or max(len(r) for r in grid) < 2:
        return []
    cells = [c for r in grid for c in r]
    if sum(1 for c in cells if c) < 0.4 * len(cells):
        return []
    width = max(len(r) for r in grid)
    header = [h or f"column {i + 1}" for i, h in enumerate(grid[0] + [""] * (width - len(grid[0])))]
    out, last_label = [], ""
    for row in grid[1:]:
        row = row + [""] * (width - len(row))
        label = row[0] or last_label
        last_label = label or last_label
        pairs = [f"{header[i]} is {v}" for i, v in enumerate(row) if i > 0 and v]
        if label and pairs:
            out.append(f"{label}: " + "; ".join(pairs) + ".")
        elif pairs:
            out.append("; ".join(pairs) + ".")
    return out


# ----------------------------------------------------------------------------- ordering and units
def _order_blocks(blocks: list[_Block]) -> list[_Block]:
    """Reading order: top-to-bottom, but column by column on multi-column pages (full-width blocks separate bands)."""
    if not blocks:
        return []
    pw = blocks[0].page_width
    mid = pw / 2
    full = [b for b in blocks if b.width > 0.62 * pw]
    narrow = [b for b in blocks if b not in full]
    left = [b for b in narrow if (b.x0 + b.x1) / 2 < mid]
    right = [b for b in narrow if (b.x0 + b.x1) / 2 >= mid]
    two_col = (len(left) >= 2 and len(right) >= 2
               and statistics.median(b.x1 for b in left) <= statistics.median(b.x0 for b in right) + 0.04 * pw)
    key = lambda b: (round(b.y0 / 3), b.x0)  # noqa: E731 - small tolerance so same-row blocks sort left to right
    if not two_col:
        return sorted(blocks, key=key)
    ordered: list[_Block] = []
    band_left: list[_Block] = []
    band_right: list[_Block] = []

    def flush() -> None:
        ordered.extend(sorted(band_left, key=lambda b: b.y0))
        ordered.extend(sorted(band_right, key=lambda b: b.y0))
        band_left.clear()
        band_right.clear()

    for b in sorted(blocks, key=lambda b: (b.y0, b.x0)):
        if b in full:
            flush()
            ordered.append(b)
        elif b in left:
            band_left.append(b)
        else:
            band_right.append(b)
    flush()
    return ordered


def _zone(b: _Block) -> str | None:
    """'top'/'bottom' for SHORT blocks in the page margins (headers, footers, page numbers); real paragraphs are never margin furniture."""
    if len(b.lines) > 2 or len(b.text.split()) > 10:
        return None
    if b.y1 <= 0.09 * b.page_height:
        return "top"
    if b.y0 >= 0.91 * b.page_height:
        return "bottom"
    return None


def _norm_key(text: str) -> str:
    return re.sub(r"[^a-z# ]", "", re.sub(r"\d+", "#", re.sub(r"\s+", " ", text.lower()))).strip()


def _join_lines(lines: list[_Line]) -> str:
    """Join wrapped lines into a paragraph; repair 'multi-\\nline' hyphenation only when the continuation is lower-case."""
    out = ""
    for ln in lines:
        t = ln.text
        if not out:
            out = t
        elif out.endswith("-") and t[:1].islower() and out[-2:-1].isalpha():
            out = out[:-1] + t
        else:
            out += " " + t
    return out.strip()


def _segments(block: _Block, body: float) -> list[tuple[str, str]]:
    """Split a block into (kind, text) pieces: headings, list items and paragraphs."""
    lines = block.lines
    max_w = max((ln.x1 - ln.x0) for ln in lines) or 1.0
    segs: list[tuple[str, list[_Line]]] = []
    for ln in lines:
        is_head = ln.size >= 1.15 * body or (ln.bold and ln.size >= 0.98 * body and len(ln.text.split()) <= 12 and not ln.text.endswith(_TERMINAL))
        is_bullet = bool(_BULLET.match(ln.text))
        prev = segs[-1] if segs else None
        if is_head:
            if prev and prev[0] == "heading":
                prev[1].append(ln)
            else:
                segs.append(("heading", [ln]))
        elif is_bullet:
            segs.append(("list", [ln]))
        elif prev and prev[0] in ("para", "list") and not (prev[1][-1].x1 - prev[1][-1].x0 < 0.6 * max_w and (
                prev[1][-1].text.endswith(_TERMINAL) or (len(prev[1][-1].text.split()) <= 10 and ln.text[:1].isupper()))):
            prev[1].append(ln)
        else:
            segs.append(("para", [ln]))
    out: list[tuple[str, str]] = []
    for kind, ls in segs:
        text = _join_lines(ls)
        if kind == "para" and len(ls) == 1 and len(text.split()) <= 8 and not text.endswith(_TERMINAL + (":",)) and _HEADING_TEXT.match(text) \
                and ls[0].x1 - ls[0].x0 < 0.7 * max_w + 1:
            kind = "heading"  # short, unpunctuated, line-width shorter than the paragraph: a heading without a font cue
        out.append((kind, text))
    return out


# ----------------------------------------------------------------------------- OCR
def _find_tessdata() -> str | None:
    env = os.environ.get("TESSDATA_PREFIX")
    candidates = [env] if env else []
    candidates += ["/usr/share/tesseract-ocr/5/tessdata", "/usr/share/tesseract-ocr/4.00/tessdata", "/usr/share/tessdata",
                   "/usr/local/share/tessdata", "/opt/homebrew/share/tessdata", r"C:\Program Files\Tesseract-OCR\tessdata"]
    for c in candidates:
        if c and (Path(c) / "eng.traineddata").exists():
            return c
    return None


def _ocr_page_blocks(page, language: str, dpi: int) -> list[_Block]:
    tessdata = _find_tessdata()
    if tessdata is None:
        raise ExtractionError("OCR is enabled but Tesseract language data was not found. Install Tesseract (see docs/HUMAN_ACTIONS.md) or set TESSDATA_PREFIX.")
    try:
        tp = page.get_textpage_ocr(language=language, dpi=dpi, full=True, tessdata=tessdata)
    except Exception as exc:
        raise ExtractionError(f"OCR failed ({type(exc).__name__}). Check that Tesseract and the '{language}' language data are installed.") from exc
    blocks, _ = _blocks_from_dict(page, tp)
    return blocks


_WORDLIKE = re.compile(r"^[A-Za-z][A-Za-z'\u2019\-]*[.,;:!?]?$|^[\d$€£][\d.,%]*$")


def _ocr_quality(text: str) -> float:
    tokens = text.split()
    return sum(1 for t in tokens if _WORDLIKE.match(t)) / len(tokens) if tokens else 0.0


# ----------------------------------------------------------------------------- main entry points
def extract_document(data: bytes, min_chars: int = 50, ocr_enabled: bool = False, *, ocr_language: str = "eng", ocr_dpi: int = 200,
                     ocr_engine: str = "tesseract") -> ExtractionReport:
    doc = fitz.open(stream=data, filetype="pdf")
    warnings: list[str] = []
    try:
        per_page: list[list[_Block]] = []
        tables: list[list[tuple[fitz.Rect, list[str]]]] = []
        ocr_pages: list[int] = []
        hidden_total = 0
        low_quality: list[int] = []
        for i, page in enumerate(doc):
            blocks, hidden = _blocks_from_dict(page)
            hidden_total += hidden
            tabs, _ = _tables(page)
            if tabs:
                blocks = [b for b in blocks if not any(_overlap_fraction(b, bbox) > 0.5 for bbox, _ in tabs)]
            chars = sum(len(b.text) for b in blocks) + sum(len(s) for _, ss in tabs for s in ss)
            if chars < 20 and ocr_enabled and ocr_engine == "tesseract":
                try:
                    blocks = _ocr_page_blocks(page, ocr_language, ocr_dpi)
                except ExtractionError:
                    raise
                text = " ".join(b.text for b in blocks)
                if blocks and _ocr_quality(text) < 0.55:
                    low_quality.append(i + 1)
                    blocks = []
                elif blocks:
                    ocr_pages.append(i + 1)
                tabs = []
            per_page.append(blocks)
            tables.append(tabs)
    finally:
        doc.close()

    # ---- document-wide: body font size, repeated headers/footers
    sizes = [(ln.size, len(ln.text)) for blocks in per_page for b in blocks for ln in b.lines]
    body = _weighted_median(sizes) or 11.0
    n_pages = len(per_page)
    counts: Counter[str] = Counter()
    for blocks in per_page:
        for key in {_norm_key(b.text) for b in blocks if _zone(b)}:
            counts[key] += 1
    repeated = {k for k, c in counts.items() if k and n_pages >= 3 and c >= max(3, math.ceil(0.5 * n_pages))}
    removed = 0

    pages: list[PageText] = []
    for idx, (blocks, tabs) in enumerate(zip(per_page, tables), start=1):
        kept = []
        for b in blocks:
            if _zone(b) and (_norm_key(b.text) in repeated or _PAGE_NUMBER.match(b.text.strip())):
                removed += 1
                continue
            kept.append(b)
        units: list[TextUnit] = []
        placed_tables = sorted(tabs, key=lambda t: t[0].y0)
        ordered = _order_blocks(kept)
        ti = 0
        for b in ordered:
            while ti < len(placed_tables) and placed_tables[ti][0].y0 <= b.y0:
                units.extend(TextUnit("table", s) for s in placed_tables[ti][1])
                ti += 1
            units.extend(TextUnit(k, t) for k, t in _segments(b, body) if t)
        for _, sentences in placed_tables[ti:]:
            units.extend(TextUnit("table", s) for s in sentences)
        units = _merge_fragments(units)
        pages.append(PageText(idx, "\n".join(u.text for u in units), units, idx in ocr_pages))

    if hidden_total:
        warnings.append(f"Ignored {hidden_total} piece(s) of hidden text (white or microscopic) that a reader cannot see.")
    if removed:
        warnings.append(f"Removed {removed} repeated header/footer/page-number line(s).")
    if ocr_pages:
        warnings.append(f"OCR was used on page(s) {_ranges(ocr_pages)}. OCR can misread characters; check important figures against the page image.")
    if low_quality:
        warnings.append(f"OCR text on page(s) {_ranges(low_quality)} was too unreliable and was ignored rather than guessed.")
    empty = [p.page for p in pages if len(p.text.strip()) < 20]
    total = sum(len(p.text) for p in pages)

    if total < min_chars and ocr_enabled and ocr_engine == "docling":
        pages = _extract_docling(data, n_pages)
        total = sum(len(p.text) for p in pages)
        if total >= min_chars:
            return ExtractionReport(pages, "docling", warnings + ["Docling OCR was used for this document."], [p.page for p in pages])
        raise ExtractionError("No usable text was found, even with the OCR fallback.")
    if total < min_chars:
        if ocr_enabled:
            raise ExtractionError("No usable text was found, even with OCR. The scan may be too blurry or in an unsupported language.")
        raise ExtractionError(
            "No usable text was found. The PDF may be scanned images. Enable OCR (OCR_ENABLED=true, Tesseract installed) and upload again."
        )
    if empty and not ocr_enabled:
        warnings.append(f"Page(s) {_ranges(empty)} contain no text layer (probably images) and were skipped. Enable OCR to read them.")
    elif empty:
        warnings.append(f"Page(s) {_ranges(empty)} contain no readable text.")
    method = "pymupdf+ocr" if ocr_pages else "pymupdf"
    return ExtractionReport(pages, method, warnings, ocr_pages)


def extract_pages(data: bytes, min_chars: int, ocr_enabled: bool = False, **kw) -> tuple[list[PageText], str]:
    """Return (pages, method). Raises ExtractionError when nothing usable is found."""
    report = extract_document(data, min_chars, ocr_enabled, **kw)
    return report.pages, report.method


def _extract_docling(data: bytes, n_pages: int) -> list[PageText]:
    """Optional whole-document OCR/layout engine. Requires `pip install docling`."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ExtractionError("OCR engine 'docling' requested but Docling is not installed.") from exc
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "input.pdf"
        path.write_bytes(data)
        result = DocumentConverter().convert(str(path))
    by_page: dict[int, list[str]] = {}
    for item, _level in result.document.iterate_items():
        text, prov = getattr(item, "text", None), getattr(item, "prov", None)
        if text and prov:
            by_page.setdefault(prov[0].page_no, []).append(text)
    return [PageText(p, "\n".join(by_page.get(p, [])).strip(), [TextUnit("para", t) for t in by_page.get(p, [])], True) for p in range(1, n_pages + 1)]


def _merge_fragments(units: list[TextUnit]) -> list[TextUnit]:
    """Re-join a sentence that was split over two blocks (common in OCR output and across columns):
    the previous paragraph has no closing punctuation and the next one starts in lower case."""
    out: list[TextUnit] = []
    for u in units:
        if (out and u.kind == "para" and out[-1].kind == "para" and u.text[:1].islower()
                and not out[-1].text.rstrip().endswith((".", "!", "?", ":", ";", '"', "\u201d"))):
            out[-1] = TextUnit("para", out[-1].text + " " + u.text)
        else:
            out.append(u)
    return out


def _overlap_fraction(b: _Block, rect: fitz.Rect) -> float:
    inter = fitz.Rect(b.x0, b.y0, b.x1, b.y1) & rect
    area = max((b.x1 - b.x0) * (b.y1 - b.y0), 1e-6)
    return (inter.width * inter.height) / area if not inter.is_empty else 0.0


def _weighted_median(pairs: list[tuple[float, int]]) -> float | None:
    if not pairs:
        return None
    pairs = sorted(pairs)
    half = sum(n for _, n in pairs) / 2
    acc = 0
    for size, n in pairs:
        acc += n
        if acc >= half:
            return size
    return pairs[-1][0]


def _ranges(nums: list[int]) -> str:
    nums = sorted(nums)
    parts, start, prev = [], nums[0], nums[0]
    for n in nums[1:] + [None]:  # type: ignore[list-item]
        if n is not None and n == prev + 1:
            prev = n
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if n is not None:
            start = prev = n
    return ", ".join(parts)
