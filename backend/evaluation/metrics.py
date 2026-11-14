"""Metric definitions for the evaluation report."""
from __future__ import annotations

import re
from collections import Counter
from statistics import mean


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("%", " percent")).strip()


def contains_all(text: str | None, needles: list[str]) -> bool:
    t = _norm(text or "")
    return bool(needles) and all(_norm(n) in t for n in needles)


def retrieval_metrics(rows: list[dict], ks=(1, 3, 5)) -> dict:
    """rows: {gold: [str], expected_pages: [int], ranked: [{page, text}]}. A chunk counts for a gold string when it is on
    an expected page and contains the string. Recall@K = share of gold strings found in the top K chunks; Hit@K = at least one;
    Precision@K = share of top-K chunks that contain at least one gold string on an expected page; MRR = 1/rank of first such chunk."""
    out: dict = {"questions": len(rows)}
    if not rows:
        return out
    for k in ks:
        rec, hit, prec = [], [], []
        for r in rows:
            top = r["ranked"][:k]
            found = {g for g in r["gold"] if any(c["page"] in r["expected_pages"] and _norm(g) in _norm(c["text"]) for c in top)}
            rec.append(len(found) / len(r["gold"]))
            hit.append(1.0 if found else 0.0)
            rel = [c for c in top if c["page"] in r["expected_pages"] and any(_norm(g) in _norm(c["text"]) for g in r["gold"])]
            prec.append(len(rel) / max(len(top), 1))
        out[f"recall@{k}"], out[f"hit@{k}"], out[f"precision@{k}"] = round(mean(rec), 3), round(mean(hit), 3), round(mean(prec), 3)
    rr = []
    for r in rows:
        rank = next((i for i, c in enumerate(r["ranked"], 1)
                     if c["page"] in r["expected_pages"] and any(_norm(g) in _norm(c["text"]) for g in r["gold"])), None)
        rr.append(1.0 / rank if rank else 0.0)
    out["mrr"] = round(mean(rr), 3)
    return out


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    return {"precision": None if p is None else round(p, 3), "recall": None if r is None else round(r, 3), "tp": tp, "fp": fp, "fn": fn}


def verification_metrics(rows: list[dict]) -> dict:
    """rows: {expected, predicted, conflict, expects_conflict}. Positive class per label for precision/recall."""
    conf = Counter((r["expected"], r["predicted"]) for r in rows)
    out: dict = {"claims": len(rows), "confusion": {f"{e}->{p}": n for (e, p), n in sorted(conf.items())}}
    for label in ("SUPPORTED", "CONTRADICTED"):
        tp = sum(1 for r in rows if r["expected"] == label and r["predicted"] == label)
        fp = sum(1 for r in rows if r["expected"] != label and r["predicted"] == label)
        fn = sum(1 for r in rows if r["expected"] == label and r["predicted"] != label)
        out[label.lower()] = prf(tp, fp, fn)
    not_supported = [r for r in rows if r["expected"] != "SUPPORTED"]
    false_support = sum(1 for r in not_supported if r["predicted"] == "SUPPORTED")
    out["false_support_rate"] = round(false_support / len(not_supported), 3) if not_supported else None
    out["false_support_count"] = false_support
    out["accuracy"] = round(sum(1 for r in rows if r["expected"] == r["predicted"]) / len(rows), 3) if rows else None
    cr = [r for r in rows if r.get("expects_conflict")]
    out["conflicts_reported"] = f"{sum(1 for r in cr if r['conflict'])}/{len(cr)}"
    return out


def timing(values: list[float]) -> float | None:
    return round(mean(values), 3) if values else None
