#!/usr/bin/env python3
"""Create the SYNTHETIC sample PDFs (company report, table, two-column paper, manual, scan, ...) for trying VerityGate by hand.

    python scripts/make_sample_pdf.py            # writes sample_pdfs/*.pdf  (needs `pip install pymupdf`)
Then upload them in the web UI and try e.g. "When did the Lisbon office start operating?" on company_report.pdf."""
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "backend"))
from evaluation.corpus import build_corpus  # noqa: E402

out = root / "sample_pdfs"
for name, path in build_corpus(out).items():
    print(f"{name:20} {path.relative_to(root)}")
