"""Split structured page units into overlapping, sentence-aligned chunks.

* A chunk never crosses a page boundary, so every chunk maps to exactly one page number.
* Chunks end at sentence boundaries; the overlap is made of whole sentences.
* Headings update the current section (stored with each chunk) and are never left dangling at the end of a chunk.
* Table rows (already linearised by the PDF layer) are chunked on their own and flagged kind='table'."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .pdf import PageText, TextUnit
from .textutils import split_sentences

_HEADING = re.compile(r"^(\d+(\.\d+)*\.?\s+)?[A-Z][A-Za-z0-9 ,:&'\u2019\-/()]{2,70}$")


@dataclass
class Chunk:
    idx: int
    page: int
    section: str | None
    text: str
    kind: str = "text"  # text | table


@dataclass
class _Atom:
    text: str
    words: int
    new_para: bool
    heading: bool = False


def _looks_like_heading(line: str) -> bool:
    line = line.strip()
    return bool(line) and len(line.split()) <= 10 and not line.endswith((".", ",", ";")) and bool(_HEADING.match(line))


def _units_from_text(text: str) -> list[TextUnit]:
    """Fallback for pages that only carry plain text (tests, evaluation fixtures): headings by shape, paragraphs by blank line."""
    units: list[TextUnit] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            units.append(TextUnit("para", " ".join(buf)))
            buf.clear()

    for line in text.splitlines():
        if not line.strip():
            flush()
        elif _looks_like_heading(line):
            flush()
            units.append(TextUnit("heading", line.strip()))
        else:
            buf.append(line.strip())
    flush()
    return units


def _atoms(unit: TextUnit, max_words: int) -> list[_Atom]:
    if unit.kind == "heading":
        return [_Atom(unit.text, len(unit.text.split()), True, True)]
    atoms: list[_Atom] = []
    for i, s in enumerate(split_sentences(unit.text) if unit.kind != "table" else [unit.text]):
        words = s.split()
        if len(words) > max_words:  # one enormous "sentence" (e.g. a table dump): cut it into word windows
            for j in range(0, len(words), max_words):
                part = words[j : j + max_words]
                atoms.append(_Atom(" ".join(part), len(part), i == 0 and j == 0))
        else:
            atoms.append(_Atom(s, len(words), i == 0))
    return atoms


def _render(atoms: list[_Atom]) -> str:
    out = ""
    for a in atoms:
        out += (("\n" if a.new_para else " ") if out else "") + a.text
    return out


class _PageBuilder:
    """Accumulates sentence atoms of one page into chunks."""

    def __init__(self, page: int, chunk_words: int, overlap_words: int, section: str | None, out: list[Chunk]):
        self.page, self.chunk_words, self.overlap_words, self.out = page, chunk_words, overlap_words, out
        self.section = section  # current section (updated by headings)
        self.buf: list[_Atom] = []
        self.buf_section = section
        self.fresh = 0  # words in buf that are not carried-over overlap
        self.kind = "text"

    def words(self) -> int:
        return sum(a.words for a in self.buf)

    def emit(self) -> None:
        if self.buf and self.fresh > 0:
            text = _render(self.buf).strip()
            if text:
                self.out.append(Chunk(len(self.out), self.page, self.buf_section, text, self.kind))

    def restart(self, keep_overlap: bool) -> None:
        carry: list[_Atom] = []
        if keep_overlap and self.overlap_words and self.kind == "text":
            total = 0
            for a in reversed(self.buf):
                if a.heading or total + a.words > self.overlap_words:
                    break
                carry.insert(0, a)
                total += a.words
        self.buf, self.fresh, self.buf_section = carry, 0, self.section

    def switch_kind(self, kind: str) -> None:
        if kind != self.kind:
            self.emit()
            self.kind = kind
            self.restart(False)

    def add_unit(self, unit: TextUnit) -> None:
        if unit.kind == "heading":
            if self.buf and self.fresh > self.chunk_words * 0.5:  # do not strand a heading at the end of a long chunk
                self.emit()
                self.restart(False)
            self.section = unit.text
            if not self.buf or self.fresh == 0:
                self.buf_section = self.section
        self.switch_kind("table" if unit.kind == "table" else "text")
        for atom in _atoms(unit, self.chunk_words):
            if self.buf and self.fresh > 0 and self.words() + atom.words > self.chunk_words:
                self.emit()
                self.restart(True)
                while self.buf and self.words() + atom.words > self.chunk_words:  # overlap too large for this atom
                    self.buf.pop(0)
            if not self.buf:
                self.buf_section = self.section
            self.buf.append(atom)
            self.fresh += atom.words


def chunk_pages(pages: list[PageText], chunk_words: int = 180, overlap_words: int = 30) -> list[Chunk]:
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("Require chunk_words > 0 and 0 <= overlap_words < chunk_words")
    chunks: list[Chunk] = []
    section: str | None = None
    for page in pages:
        builder = _PageBuilder(page.page, chunk_words, overlap_words, section, chunks)
        for unit in page.units or _units_from_text(page.text):
            builder.add_unit(unit)
        builder.emit()
        section = builder.section  # a heading stays in force on the next page until a new one appears
    return chunks
