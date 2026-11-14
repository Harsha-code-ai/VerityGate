#!/usr/bin/env python3
"""Convenience wrapper for backend/evaluation/calibrate.py:  python scripts/calibrate.py --mode real"""
import subprocess
import sys
from pathlib import Path

backend = Path(__file__).resolve().parents[1] / "backend"
sys.exit(subprocess.call([sys.executable, "-m", "evaluation.calibrate", *sys.argv[1:]], cwd=backend))
