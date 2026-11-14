#!/usr/bin/env python3
"""Convenience wrapper:  python scripts/evaluate.py --mode mock|real --out evaluation_results/run.json
(the real code is backend/evaluation/run.py; run it inside the backend directory or the test container)."""
import subprocess
import sys
from pathlib import Path

backend = Path(__file__).resolve().parents[1] / "backend"
sys.exit(subprocess.call([sys.executable, "-m", "evaluation.run", *sys.argv[1:]], cwd=backend))
