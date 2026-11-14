"""Choose verification thresholds from data instead of guessing.

    python -m evaluation.calibrate --mode real      # with your real NLI/embedding models: the numbers that matter
    python -m evaluation.calibrate --mode mock      # pipeline check only (stand-in models; values meaningless for production)

Every claim in evaluation/data/claims.json is verified once per threshold pair (model scores are cached, so the sweep is cheap).
The recommendation is the pair with ZERO false support and the highest combined recall; add your own documents/claims to
evaluation/data/claims.json (and the PDF builders) to calibrate on your domain."""
from __future__ import annotations

import argparse
import json
import tempfile
import time
from dataclasses import replace
from pathlib import Path

from app.claims import ParsedClaim
from app.db import new_id
from app.verifier import ClaimVerifier, NLIScores

from . import metrics
from .corpus import build_corpus
from .run import DATA, PDFS, ingest_corpus, make_services


class CachingNLI:
    def __init__(self, inner):
        self.inner, self.cache = inner, {}

    def predict(self, pairs):
        missing = [p for p in dict.fromkeys(pairs) if p not in self.cache]
        for p, s in zip(missing, self.inner.predict(missing)):
            self.cache[p] = s
        return [self.cache[p] if isinstance(self.cache[p], NLIScores) else NLIScores(*self.cache[p]) for p in pairs]


def calibrate(mode: str) -> dict:
    claims = json.loads((DATA / "claims.json").read_text())
    if not PDFS.exists() or not any(PDFS.glob("*.pdf")):
        build_corpus(PDFS)
    with tempfile.TemporaryDirectory() as tmp:
        from app.pdf import _find_tessdata

        svc = make_services(mode, Path(tmp), _find_tessdata() is not None)
        owner = new_id()
        svc.db.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES (?,?,?,?)", (owner, "cal@example.com", "x", time.time()))
        docs = ingest_corpus(svc, owner, PDFS)
        usable = [c for c in claims if c["doc"] in docs and not c.get("semantic") or (mode == "real" and c["doc"] in docs)]
        nli = CachingNLI(svc.nli)
        base = svc.verifier_config()
        retrieved = {c["id"]: (svc.retriever.retrieve(owner, docs[c["doc"]], c["claim"], 6), svc.retriever.term_df(owner, docs[c["doc"]])) for c in usable}
        rows = []
        for ent in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95):
            for con in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95):
                cfg = replace(base, entail_thr=ent, contra_thr=con, conflict_thr=max(con, 0.9))
                v = ClaimVerifier(nli, embedder=svc.embedder, config=cfg)
                preds = []
                for c in usable:
                    passages, df = retrieved[c["id"]]
                    r = v.verify_claims([ParsedClaim(c["claim"], [])], {}, passages, df)[0]
                    preds.append({"expected": c["expected"], "predicted": r.status, "conflict": r.conflict, "expects_conflict": c.get("expects_conflict", False)})
                m = metrics.verification_metrics(preds)
                rows.append({"entail": ent, "contra": con, "accuracy": m["accuracy"], "false_support": m["false_support_count"],
                             "sup_precision": m["supported"]["precision"], "sup_recall": m["supported"]["recall"],
                             "con_precision": m["contradicted"]["precision"], "con_recall": m["contradicted"]["recall"]})
        svc.db.close()
    safe = [r for r in rows if r["false_support"] == 0 and (r["con_precision"] or 0) >= 0.95]
    best = max(safe, key=lambda r: ((r["sup_recall"] or 0) + (r["con_recall"] or 0), r["entail"], r["contra"])) if safe else None
    return {"mode": mode, "claims": len(usable), "grid": rows, "recommended": best}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["mock", "real"], default="mock")
    args = ap.parse_args()
    res = calibrate(args.mode)
    print(f"Threshold sweep on {res['claims']} claims ({res['mode']} models)\n")
    print("entail contra  acc   false_sup  sup_P  sup_R  con_P  con_R")
    for r in res["grid"]:
        print(f"{r['entail']:<7} {r['contra']:<6} {r['accuracy']:<5} {r['false_support']:<10} {r['sup_precision']!s:<6} {r['sup_recall']!s:<6} {r['con_precision']!s:<6} {r['con_recall']}")
    rec = res["recommended"]
    print("\nRecommended:", f"ENTAILMENT_THRESHOLD={rec['entail']} CONTRADICTION_THRESHOLD={rec['contra']}" if rec else "no setting had zero false support - do not deploy these models")
    if args.mode == "mock":
        print("(mock models: this only proves the tool works; calibrate with --mode real)")


if __name__ == "__main__":
    main()
