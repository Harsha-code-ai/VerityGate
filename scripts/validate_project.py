#!/usr/bin/env python3
"""Sanity checks before you commit or zip: required files exist and no secrets are present."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ["README.md", ".env.example", ".gitignore", "docker-compose.yml", "backend/Dockerfile", "frontend/Dockerfile",
            "docs/HUMAN_ACTIONS.md", "docs/ARCHITECTURE.md", "docs/SECURITY.md", "docs/API.md", "docs/EVALUATION.md", ".github/workflows/ci.yml",
            "PROJECT_AUDIT_REPORT.md", "TESTING_REPORT.md", "backend/evaluation/run.py", "backend/evaluation/data/questions.json"]
SECRET = re.compile(r"(sk-[A-Za-z0-9_\-]{20,}|AKIA[0-9A-Z]{16})")
bad = [f for f in REQUIRED if not (ROOT / f).exists()]
leaks = []
for p in ROOT.rglob("*"):
    if p.is_file() and not any(x in p.parts for x in ("node_modules", ".venv", ".git", "dist")) and p.stat().st_size < 2_000_000:
        if p.name == ".env" or p.suffix in {".sqlite3", ".db"}:
            leaks.append(f"{p}: should not be committed")
        elif p.suffix in {".py", ".md", ".yml", ".json", ".ts", ".tsx", ".example", ".env"} or p.name.startswith(".env"):
            if SECRET.search(p.read_text(errors="ignore")):
                leaks.append(f"{p}: looks like a real key")
print("missing:", bad or "none")
print("leaks:", leaks or "none")
sys.exit(1 if bad or leaks else 0)
