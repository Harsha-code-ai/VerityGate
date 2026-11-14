
"""Deterministic generator for synthetic evaluation PDFs.

These invented documents test VerityGate against different layouts:
plain text, headings, tables, multiple pages, conflicting statements,
scanned documents, and long documents.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pymupdf

PAGE_W, PAGE_H = 595, 842
MARGIN = 56


class _Writer:
    """Minimal top-to-bottom layout helper for one page."""

    def __init__(
        self,
        doc: pymupdf.Document,
        header: str | None = None,
        footer: str | None = None,
        number: int | None = None,
    ):
        self.page = doc.new_page(width=PAGE_W, height=PAGE_H)
        self.y = 90.0

        if header:
            self.page.insert_text(
                (MARGIN, 36), header, fontsize=8.5, fontname="helv"
            )

        if footer:
            self.page.insert_text(
                (MARGIN, PAGE_H - 30),
                footer,
                fontsize=8.5,
                fontname="helv",
            )

        if number is not None:
            self.page.insert_text(
                (PAGE_W - 110, PAGE_H - 30),
                f"Page {number}",
                fontsize=8.5,
                fontname="helv",
            )

    def heading(self, text: str, size: float = 17) -> None:
        self.page.insert_text(
            (MARGIN, self.y + size),
            text,
            fontsize=size,
            fontname="hebo",
        )
        self.y += size + 16

    def para(
        self,
        text: str,
        size: float = 11,
        x0: float = MARGIN,
        x1: float = PAGE_W - MARGIN,
    ) -> None:
        width = x1 - x0
        lines = max(1, math.ceil(len(text) * size * 0.52 / width)) + 1
        height = lines * size * 1.35

        self.page.insert_textbox(
            pymupdf.Rect(x0, self.y, x1, self.y + height),
            text,
            fontsize=size,
            fontname="helv",
        )

        self.y += height + 8

    def table(self, rows: list[list[str]], col_w: float = 120) -> None:
        row_h = 24

        for row_index, row in enumerate(rows):
            for col_index, value in enumerate(row):
                rect = pymupdf.Rect(
                    MARGIN + col_index * col_w,
                    self.y + row_index * row_h,
                    MARGIN + (col_index + 1) * col_w,
                    self.y + (row_index + 1) * row_h,
                )

                self.page.draw_rect(rect, color=(0, 0, 0), width=0.8)

                self.page.insert_text(
                    (rect.x0 + 5, rect.y1 - 8),
                    value,
                    fontsize=10.5,
                    fontname="hebo" if row_index == 0 else "helv",
                )

        self.y += len(rows) * row_h + 14

    def two_columns(
        self,
        left: str,
        right: str,
        size: float = 10,
    ) -> None:
        mid = PAGE_W / 2

        columns = (
            (MARGIN, mid - 12, left),
            (mid + 12, PAGE_W - MARGIN, right),
        )

        heights = []

        for x0, x1, text in columns:
            width = x1 - x0
            height = (
                math.ceil(len(text) * size * 0.52 / width) + 1
            ) * size * 1.35

            self.page.insert_textbox(
                pymupdf.Rect(x0, self.y, x1, self.y + height),
                text,
                fontsize=size,
                fontname="helv",
            )

            heights.append(height)

        self.y += max(heights) + 8


def _save(doc: pymupdf.Document, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path), deflate=True)
    doc.close()
    return path


def company_report(path: Path) -> Path:
    d = pymupdf.open()

    w = _Writer(d)
    w.heading("Northstar Logistics Annual Report 2023")
    w.heading("Company Overview", 13)

    w.para(
        "The company opened its Lisbon office on 15 June 2019. "
        "The Lisbon office employs 120 people and handles customer "
        "support for southern Europe."
    )

    w.para(
        "The Madrid office opened in 2021 and employs 95 people. "
        "Warehouse maintenance is scheduled for March and will last "
        "two weeks."
    )

    w = _Writer(d)
    w.heading("Financial Results")

    w.para(
        "Revenue was 45 million US dollars in 2023. "
        "Revenue increased by 12.5 percent from 2022 to 2023, "
        "driven by strong demand for freight services."
    )

    w.para(
        "Operating costs were stable across the year. "
        "Net profit reached 6.1 million US dollars."
    )

    w = _Writer(d)
    w.heading("Sustainability")

    w.para(
        "The company reduced carbon emissions by 30 percent "
        "between 2020 and 2023. Renewable sources supplied "
        "62 percent of electricity in 2023."
    )

    return _save(d, path)


def financial_table(path: Path) -> Path:
    d = pymupdf.open()

    w = _Writer(d)
    w.heading("Group Financial Summary")

    w.para(
        "All amounts in this report are stated in millions of euros "
        "unless noted otherwise."
    )

    w.heading("Revenue by Region", 13)

    w.table([
        ["Region", "2022", "2023"],
        ["Europe", "38.2", "41.5"],
        ["Asia", "21.0", "24.8"],
        ["Americas", "15.7", "15.7"],
    ])

    w.para(
        "Group revenue was 82 million euros in 2023. "
        "The operating margin was 9 percent."
    )

    w = _Writer(d)
    w.heading("Outlook")

    w.para(
        "Management expects revenue growth of 5 percent in 2024. "
        "Capital expenditure will be 7 million euros."
    )

    return _save(d, path)


def research_paper(path: Path) -> Path:
    d = pymupdf.open()

    header = "Efficient Image Classification - Preprint"
    footer = "Draft for review - do not distribute"

    w = _Writer(d, header, footer, 1)

    w.heading(
        "Efficient Image Classification with Small Networks",
        15,
    )

    w.para(
        "Abstract. We study compact convolutional networks for image "
        "classification. Our model reaches 91.4 percent accuracy on "
        "the test set while using 3.2 million parameters.",
        10.5,
    )

    w.heading("1 Introduction", 12)

    w.two_columns(
        "Image classification is widely used in industry. Small "
        "networks are attractive because they run on mobile devices. "
        "Prior work focused on larger models with more than "
        "20 million parameters.",
        "In this paper we propose a pruning strategy that removes "
        "redundant filters. The method requires no retraining from "
        "scratch and keeps the training schedule unchanged.",
    )

    w = _Writer(d, header, footer, 2)
    w.heading("2 Experiments", 12)

    w.two_columns(
        "The dataset contains 12,000 labelled images across 10 "
        "classes. We trained every model for 40 epochs using a "
        "batch size of 64.",
        "The baseline network reached 88.9 percent accuracy. "
        "The pruned network is 4 times smaller than the baseline "
        "and 2.5 percent more accurate.",
    )

    w.heading("3 Limitations", 12)

    w.para(
        "The evaluation uses a single dataset. "
        "We did not measure energy consumption on real devices.",
        10.5,
    )

    w = _Writer(d, header, footer, 3)
    w.heading("4 Conclusion", 12)

    w.para(
        "Pruning compact networks gives accurate and small models. "
        "Future work will study transformer architectures.",
        10.5,
    )

    return _save(d, path)


def tech_manual(path: Path) -> Path:
    d = pymupdf.open()

    header = "X200 Router - Installation Guide"

    w = _Writer(d, header, None, 1)
    w.heading("Router X200 Installation Guide")
    w.heading("1. Specifications", 13)

    w.para(
        "The Router X200 uses a 12 V power supply. "
        "The default management address is 192.168.10.1. "
        "The installed firmware version is 3.2.1."
    )

    w.para(
        "The Router X300 uses a 24 V power supply and ships "
        "with firmware version 4.0.0."
    )

    w = _Writer(d, header, None, 2)
    w.heading("2. Maintenance", 13)
    w.heading("2.1 Factory reset", 11.5)

    w.para(
        "To restore factory settings, hold the reset button "
        "for 10 seconds while the device is powered on."
    )

    w.heading("2.2 Warranty", 11.5)

    w.para(
        "The warranty period is 24 months from the date of purchase. "
        "Water damage is not covered by the warranty."
    )

    return _save(d, path)


def assignment_brief(path: Path) -> Path:
    d = pymupdf.open()

    w = _Writer(d, "BUS402 Strategy - Assessment 2", None, 1)
    w.heading("Assessment 2: Strategy Report")

    w.para(
        "This assessment is worth 40 percent of the final mark. "
        "The report must not exceed 2,500 words, excluding references."
    )

    w.para(
        "The submission deadline is 14 November 2025 at 23:59. "
        "Submit the report through the course portal as a single PDF file."
    )

    w = _Writer(d, "BUS402 Strategy - Assessment 2", None, 2)
    w.heading("Late submissions")

    w.para(
        "A penalty of 10 percent of the available marks is applied "
        "for each day a submission is late. Extensions require a "
        "request at least 48 hours before the deadline."
    )

    return _save(d, path)


def conflicting_policy(path: Path) -> Path:
    d = pymupdf.open()

    w = _Writer(d)
    w.heading("Remote Work Policy")

    w.para(
        "Employees may work remotely 3 days per week. "
        "Managers approve remote schedules each quarter."
    )

    w = _Writer(d)
    w.heading("Equipment")

    w.para(
        "The company provides a laptop and a monitor to every employee. "
        "Equipment must be returned when employment ends."
    )

    w = _Writer(d)
    w.heading("Policy Update")

    w.para(
        "Employees may work remotely 2 days per week. "
        "Core office days are Tuesday and Thursday."
    )

    return _save(d, path)


def scanned_memo(path: Path) -> Path:
    src = pymupdf.open()

    page = src.new_page(width=PAGE_W, height=PAGE_H)

    page.insert_textbox(
        pymupdf.Rect(MARGIN, 80, PAGE_W - MARGIN, 400),
        (
            "Board Memo\n"
            "The board approved a budget of 2.4 million dollars "
            "on 9 March 2022. The project manager is Elena Marsh. "
            "Construction will start in the autumn."
        ),
        fontsize=15,
        fontname="helv",
    )

    pix = page.get_pixmap(dpi=200)

    d = pymupdf.open()
    output_page = d.new_page(width=PAGE_W, height=PAGE_H)
    output_page.insert_image(output_page.rect, pixmap=pix)

    src.close()

    return _save(d, path)


def long_report(path: Path, pages: int = 40) -> Path:
    d = pymupdf.open()

    filler = (
        "Operations continued under the standard service model with "
        "routine inspections and scheduled reviews. Staffing levels "
        "remained within the planned range and no material changes "
        "to procedures were introduced. "
    )

    facts = {
        7: "Station Delta recorded 417 safety incidents during the year.",
        23: "The central depot was relocated to Harbour Road in September 2021.",
        38: "The audit committee approved a contingency reserve of 3.5 million dollars.",
    }

    for number in range(1, pages + 1):
        w = _Writer(
            d,
            "Regional Operations Review",
            "Confidential - internal use",
            number,
        )

        w.heading(f"Section {number}: Operations Update", 14)

        w.para(
            filler * 3 + facts.get(number, "") + " " + filler
        )

    return _save(d, path)


def incomplete_info(path: Path) -> Path:
    d = pymupdf.open()

    w = _Writer(d)
    w.heading("Corner Bakery Information Sheet")

    w.para(
        "Corner Bakery is open from 7 am to 6 pm on weekdays. "
        "The bakery sells sourdough bread, croissants and cinnamon rolls."
    )

    w.para(
        "Orders for large events must be placed at least three days "
        "in advance. The bakery is located on Mill Street."
    )

    return _save(d, path)


BUILDERS = {
    "company_report": company_report,
    "financial_table": financial_table,
    "research_paper": research_paper,
    "tech_manual": tech_manual,
    "assignment_brief": assignment_brief,
    "conflicting_policy": conflicting_policy,
    "scanned_memo": scanned_memo,
    "long_report": long_report,
    "incomplete_info": incomplete_info,
}


def build_corpus(out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    return {
        name: builder(out_dir / f"{name}.pdf")
        for name, builder in BUILDERS.items()
    }


if __name__ == "__main__":
    target = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else Path(__file__).parent / "pdfs"
    )

    for name, path in build_corpus(target).items():
        print(
            f"{name:20} {path.stat().st_size // 1024:5d} KB  {path}"
        )
