
"""Run the evaluation: retrieval quality, claim verification, and end-to-end question answering.

    python -m evaluation.run --mode mock
        # Fast; deterministic stand-in models (measures pipeline LOGIC only)

    python -m evaluation.run --mode real
        # Real embedding + NLI + LLM from your .env (needs models, LLM key)

`--mode real` reports the numbers that matter for production.
`--mode mock` numbers must never be quoted as model accuracy.
"""

from __future__ import annotations

import argparse
import json
import platform
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from app.claims import ParsedClaim
from app.config import Settings
from app.db import new_id
from app.logging_setup import setup_logging
from app.pdf import _find_tessdata
from app.services import Services, build_services, ingest_document
from app.verifier import ClaimVerifier
from app.workflow import run_question

from . import metrics
from .corpus import BUILDERS, build_corpus


DATA = Path(__file__).parent / "data"
PDFS = Path(__file__).parent / "pdfs"


def _load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text())


def make_services(mode: str, workdir: Path, ocr: bool) -> Services:
    common = dict(
        data_dir=workdir,
        qdrant_url=":memory:",
        secret_key="e" * 40,
        rate_limit_per_minute=100000,
        ocr_enabled=ocr,
    )

    if mode == "mock":
        from .mocks import ExtractiveMockLLM, HashedEmbedder, MockNLI

        settings = Settings(
            **common,
            llm_provider="openai",
            llm_model="mock-extractive",
            llm_api_key="mock",
        )

        llm = ExtractiveMockLLM()

        return build_services(
            settings,
            embedder=HashedEmbedder(),
            nli=MockNLI(),
            llm_factory=lambda: llm,
        )

    settings = Settings(**common)

    return build_services(settings)


def ingest_corpus(
    svc: Services,
    owner: str,
    pdf_dir: Path,
) -> dict[str, str]:

    ids: dict[str, str] = {}

    for name in BUILDERS:
        data = (pdf_dir / f"{name}.pdf").read_bytes()

        doc_id = new_id()
        stored = f"{new_id()}.pdf"

        (svc.settings.uploads_dir / stored).write_bytes(data)

        now = time.time()

        svc.db.execute(
            """
            INSERT INTO documents(
                id,
                owner_id,
                display_name,
                stored_name,
                sha256,
                size_bytes,
                status,
                created_at,
                updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                doc_id,
                owner,
                f"{name}.pdf",
                stored,
                new_id(),
                len(data),
                "pending",
                now,
                now,
            ),
        )

        ingest_document(svc, doc_id)

        row = svc.db.one(
            "SELECT status, error FROM documents WHERE id=?",
            (doc_id,),
        )

        if row["status"] != "ready":
            print(
                f"  ! {name}: "
                f"{row['status']} - {row['error']}"
            )
            continue

        ids[name] = doc_id

    return ids


def evaluate(
    mode: str,
    pdf_dir: Path = PDFS,
    limit_docs: set[str] | None = None,
) -> dict:

    questions = _load("questions.json")
    claims = _load("claims.json")

    if not all(
        (pdf_dir / f"{name}.pdf").exists()
        for name in BUILDERS
    ):
        build_corpus(pdf_dir)

    ocr = _find_tessdata() is not None

    with tempfile.TemporaryDirectory() as tmp:

        svc = make_services(mode, Path(tmp), ocr)

        owner = new_id()

        svc.db.execute(
            """
            INSERT INTO users(
                id,
                email,
                password_hash,
                created_at
            )
            VALUES (?, ?, ?, ?)
            """,
            (
                owner,
                "eval@example.com",
                "x",
                time.time(),
            ),
        )

        t_ingest = time.perf_counter()

        doc_ids = ingest_corpus(
            svc,
            owner,
            pdf_dir,
        )

        ingest_s = time.perf_counter() - t_ingest

        skipped_ocr = 0

        results: dict = {
            "mode": mode,
            "ocr_available": ocr,
        }

        def usable(item: dict) -> bool:
            nonlocal skipped_ocr

            if (
                item["doc"] not in doc_ids
                or (
                    limit_docs
                    and item["doc"] not in limit_docs
                )
            ):
                if item.get("needs_ocr"):
                    skipped_ocr += 1

                return False

            return True

        # --------------------------------------------------
        # RETRIEVAL EVALUATION
        # --------------------------------------------------

        answerable = [
            q
            for q in questions
            if q["answerable"] and usable(q)
        ]

        retrieval: dict = {}

        for rmode in ("hybrid", "bm25", "dense"):

            rows = []
            ms = []
            by_kind = {}

            for q in answerable:

                t0 = time.perf_counter()

                passages = svc.retriever.retrieve(
                    owner,
                    doc_ids[q["doc"]],
                    q["question"],
                    5,
                    mode=rmode,
                )

                ms.append(
                    (time.perf_counter() - t0) * 1000
                )

                row = {
                    "gold": q["expected_answer_contains"],
                    "expected_pages": q["expected_pages"],
                    "ranked": [
                        {
                            "page": p.page,
                            "text": p.text,
                        }
                        for p in passages
                    ],
                    "kind": q["kind"],
                    "semantic": q.get("semantic", False),
                }

                rows.append(row)

                by_kind.setdefault(
                    q["kind"], []
                ).append(row)

            retrieval[rmode] = {
                **metrics.retrieval_metrics(rows),

                "avg_ms": metrics.timing(ms),

                "by_kind": {
                    k: metrics.retrieval_metrics(v)
                    for k, v in sorted(by_kind.items())
                },

                "failures": [
                    q["id"]
                    for q, r in zip(answerable, rows)
                    if not any(
                        c["page"] in r["expected_pages"]
                        and any(
                            metrics._norm(g)
                            in metrics._norm(c["text"])
                            for g in r["gold"]
                        )
                        for c in r["ranked"][:5]
                    )
                ],
            }

        results["retrieval"] = retrieval

        # --------------------------------------------------
        # CLAIM VERIFICATION EVALUATION
        # --------------------------------------------------

        verifier = ClaimVerifier(
            svc.nli,
            embedder=svc.embedder,
            config=svc.verifier_config(),
        )

        vrows = []
        detail = []

        for c in claims:

            if not usable(c):
                continue

            doc = doc_ids[c["doc"]]

            passages = svc.retriever.retrieve(
                owner,
                doc,
                c["claim"],
                6,
            )

            res = verifier.verify_claims(
                [ParsedClaim(c["claim"], [])],
                {},
                passages,
                svc.retriever.term_df(owner, doc),
            )[0]

            pages = sorted({
                e.page
                for e in res.evidence
                if e.role in (
                    "supporting",
                    "contradicting",
                )
            })

            row = {
                "expected": c["expected"],
                "predicted": res.status,
                "conflict": res.conflict,
                "expects_conflict": c.get(
                    "expects_conflict",
                    False,
                ),
                "semantic": c.get("semantic", False),
                "kind": c["kind"],
            }

            vrows.append(row)

            detail.append({
                "id": c["id"],
                "claim": c["claim"],
                "expected": c["expected"],
                "predicted": res.status,
                "reason_code": res.reason_code,
                "ok": c["expected"] == res.status,
                "semantic": c.get("semantic", False),
                "evidence_pages": pages,
                "page_ok": (
                    not c["expected_pages"]
                    or bool(
                        set(pages)
                        & set(c["expected_pages"])
                    )
                ),
            })

        results["verification"] = {

            "all": metrics.verification_metrics(vrows),

            "non_semantic": metrics.verification_metrics(
                [
                    r
                    for r in vrows
                    if not r["semantic"]
                ]
            ),

            "semantic_only": metrics.verification_metrics(
                [
                    r
                    for r in vrows
                    if r["semantic"]
                ]
            ),

            "failures": [
                d
                for d in detail
                if not d["ok"]
            ],

            "citation_page_accuracy": round(
                sum(
                    d["page_ok"]
                    for d in detail
                    if d["expected"] != "NOT_ESTABLISHED"
                )
                / max(
                    1,
                    sum(
                        1
                        for d in detail
                        if d["expected"] != "NOT_ESTABLISHED"
                    ),
                ),
                3,
            ),
        }

        # --------------------------------------------------
        # END-TO-END QUESTION ANSWERING
        # --------------------------------------------------

        graph, cfg = svc.build_graph()

        e2e: list[dict] = []

        times: list[float] = []
        ret_ms: list[float] = []
        retries: list[int] = []
        calls: list[int] = []

        ans_ok = 0
        cite_ok = 0
        cite_n = 0
        abst_ok = 0
        false_support = 0
        n_ans = 0
        n_unans = 0

        for q in questions:

            if not usable(q):
                continue

            out = run_question(
                graph,
                owner,
                doc_ids[q["doc"]],
                q["question"],
                cfg,
            )

            times.append(out["latency_seconds"])
            ret_ms.append(out["retrieval_ms"])
            retries.append(out["retries_used"])
            calls.append(out["llm_calls"])

            verified = out["status"] in (
                "VERIFIED",
                "PARTIALLY_VERIFIED",
            )

            if q["answerable"]:

                n_ans += 1

                correct = (
                    bool(out["conflicts"])
                    if q.get("expects_conflict")
                    else (
                        verified
                        and metrics.contains_all(
                            out["verified_answer"],
                            q["expected_answer_contains"],
                        )
                    )
                )

                ans_ok += int(correct)

                abst_ok += int(
                    verified or bool(out["conflicts"])
                )

                if verified and not q.get("expects_conflict"):

                    cite_n += 1

                    cite_ok += int(
                        bool(out["sources"])
                        and set(out["sources"])
                        <= set(q["expected_pages"])
                    )

            else:

                n_unans += 1

                abst_ok += int(
                    out["status"] == "ABSTAINED"
                )

                false_support += int(verified)

            # Store diagnostic information for every question.
            e2e.append({

                "id": q["id"],

                "kind": q["kind"],

                "question": q["question"],

                "status": out["status"],

                "answerable": q["answerable"],

                "answer": out["verified_answer"],

                "sources": out["sources"],

                "retries": out["retries_used"],

                "abstain_reason": out.get(
                    "abstain_reason"
                ),

                "llm_calls": out.get(
                    "llm_calls",
                    0,
                ),

                "ok": (
                    out["status"] == "ABSTAINED"
                    if not q["answerable"]
                    else (
                        bool(out["conflicts"])
                        if q.get("expects_conflict")
                        else (
                            verified
                            and metrics.contains_all(
                                out["verified_answer"],
                                q["expected_answer_contains"],
                            )
                        )
                    )
                ),
            })

        total = n_ans + n_unans

        results["end_to_end"] = {

            "questions": total,

            "answerable": n_ans,

            "unanswerable": n_unans,

            "answer_accuracy": (
                round(ans_ok / n_ans, 3)
                if n_ans
                else None
            ),

            "citation_accuracy": (
                round(cite_ok / cite_n, 3)
                if cite_n
                else None
            ),

            "citations_checked": cite_n,

            "abstention_accuracy": (
                round(abst_ok / total, 3)
                if total
                else None
            ),

            "unanswerable_false_support": (
                f"{false_support}/{n_unans}"
            ),

            "avg_response_s": metrics.timing(times),

            "avg_retrieval_ms": metrics.timing(ret_ms),

            "retry_frequency": (
                round(
                    sum(1 for r in retries if r)
                    / len(retries),
                    3,
                )
                if retries
                else None
            ),

            "avg_llm_calls": metrics.timing(calls),

            "failures": [
                e
                for e in e2e
                if not e["ok"]
            ],
        }

        # --------------------------------------------------
        # METADATA
        # --------------------------------------------------

        results["meta"] = {

            "date_utc": datetime.now(
                timezone.utc
            ).isoformat(timespec="seconds"),

            "python": platform.python_version(),

            "documents": len(doc_ids),

            "questions": len(questions),

            "claims": len(claims),

            "ingest_seconds": round(ingest_s, 2),

            "skipped_needing_ocr": skipped_ocr,

            "settings": {
                k: getattr(svc.settings, k)
                for k in (
                    "embedding_model",
                    "nli_model",
                    "llm_provider",
                    "llm_model",
                    "chunk_words",
                    "chunk_overlap_words",
                    "retrieval_top_k",
                    "entailment_threshold",
                    "contradiction_threshold",
                    "conflict_threshold",
                    "relevance_min_coverage",
                    "relevance_min_support_coverage",
                    "relevance_min_similarity",
                )
            },

            "models": (
                "MOCK (hashed embeddings, overlap NLI, extractive LLM)"
                if mode == "mock"
                else "REAL (as configured)"
            ),
        }

        svc.db.close()

    return results


def to_markdown(r: dict) -> str:

    m = r["meta"]
    rt = r["retrieval"]
    v = r["verification"]
    e = r["end_to_end"]

    lines = [
        f"# VerityGate evaluation ({r['mode']} mode)",
        "",
        f"- Models: **{m['models']}**",
        f"- Date (UTC): {m['date_utc']}  |  Python {m['python']}",
        (
            f"- Dataset: {m['documents']} synthetic PDFs, "
            f"{m['questions']} questions, {m['claims']} claim cases"
            f"  |  OCR available: {r['ocr_available']}"
        ),
        "",
    ]

    if r["mode"] == "mock":

        lines += [
            (
                "> **These numbers come from deterministic stand-in models. "
                "They test the pipeline logic, not model accuracy, and must "
                "not be quoted as production accuracy.**"
            ),
            "",
        ]

    lines += [
        "## Retrieval (answerable questions, top-5 chunks)",
        "",
        "| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | Avg ms |",
        "|---|---|---|---|---|---|---|",
    ]

    for k, x in rt.items():

        lines.append(
            f"| {k} | {x['recall@1']} | {x['recall@3']} "
            f"| {x['recall@5']} | {x['precision@5']} "
            f"| {x['mrr']} | {x['avg_ms']} |"
        )

    lines += [
        "",
        (
            "Hybrid retrieval misses (question ids): "
            f"{rt['hybrid']['failures'] or 'none'}"
        ),
        "",
        "## Claim verification",
        "",
    ]

    for label, key in (
        ("All cases", "all"),
        (
            "Cases not needing language understanding",
            "non_semantic",
        ),
        (
            "Cases needing real NLI (`semantic`)",
            "semantic_only",
        ),
    ):

        x = v[key]

        if not x["claims"]:
            continue

        lines.append(
            f"**{label}** ({x['claims']} claims): "
            f"accuracy {x['accuracy']}, "
            f"SUPPORTED P/R "
            f"{x['supported']['precision']}/"
            f"{x['supported']['recall']}, "
            f"CONTRADICTED P/R "
            f"{x['contradicted']['precision']}/"
            f"{x['contradicted']['recall']}, "
            f"false-support rate {x['false_support_rate']} "
            f"({x['false_support_count']}), "
            f"conflicts reported {x['conflicts_reported']}"
        )

        lines.append("")

    lines += [
        (
            "Citation page accuracy (claims with expected evidence): "
            f"{v['citation_page_accuracy']}"
        ),
        "",
        "Verification mismatches:",
        "",
    ]

    for f in v["failures"]:

        lines.append(
            f"- `{f['id']}` expected {f['expected']}, "
            f"got {f['predicted']} ({f['reason_code']})"
            f"{' [semantic]' if f['semantic'] else ''}: "
            f"{f['claim']}"
        )

    lines += [
        "",
        "## End-to-end",
        "",
        (
            f"- Questions: {e['questions']} "
            f"({e['answerable']} answerable, "
            f"{e['unanswerable']} unanswerable)"
        ),
        (
            f"- Answer accuracy: {e['answer_accuracy']}  |  "
            f"Citation accuracy: {e['citation_accuracy']} "
            f"(n={e['citations_checked']})"
        ),
        (
            f"- Abstention accuracy: {e['abstention_accuracy']}  |  "
            f"Unanswerable questions answered as verified: "
            f"{e['unanswerable_false_support']}"
        ),
        (
            f"- Avg response: {e['avg_response_s']} s  |  "
            f"Avg retrieval: {e['avg_retrieval_ms']} ms  |  "
            f"Retry frequency: {e['retry_frequency']}  |  "
            f"Avg LLM calls: {e['avg_llm_calls']}"
        ),
        "",
        "End-to-end mismatches:",
        "",
    ]

    for f in e["failures"]:

        lines.append(
            f"- `{f['id']}` ({f['kind']}) "
            f"status {f['status']}: {f['question']} "
            f"| abstain reason: "
            f"{f.get('abstain_reason') or 'not recorded'} "
            f"| LLM calls: {f.get('llm_calls', 0)}"
        )

    return "\n".join(lines) + "\n"


def main() -> None:

    # Enable INFO-level application logs during evaluation.
    setup_logging("INFO")

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    ap.add_argument(
        "--mode",
        choices=["mock", "real"],
        default="mock",
    )

    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSON results here (a .md report is written next to it)",
    )

    args = ap.parse_args()

    res = evaluate(args.mode)

    md = to_markdown(res)

    print(md)

    if args.out:

        args.out.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.out.write_text(
            json.dumps(
                res,
                indent=1,
                default=str,
            )
        )

        args.out.with_suffix(".md").write_text(md)


if __name__ == "__main__":
    main()
