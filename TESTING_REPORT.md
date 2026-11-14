# VerityGate – Testing report

Everything below was **actually executed** in the build sandbox (Python 3.12.3, Node 22, 1 CPU, 3 GB RAM, no Docker, **no access to Hugging Face**). Anything not executed is listed as such. Mocked and real components are separated explicitly.

## 1. What was and was not possible

| Item | Status |
|---|---|
| Python compile (`compileall`) of app, evaluation, tests, scripts | ✅ run – OK |
| Lint (`ruff check backend scripts`) | ✅ run – All checks passed |
| Backend tests | ✅ run – **231 passed, 0 failed** (5 real-model tests deselected by default) |
| Real-model tests (`pytest -m real_models`) | ⚠️ **5 skipped** – `transformers`/`sentence_transformers` are not installed and the models cannot be downloaded here. A full PyTorch install was attempted and failed with *No space left on device* (the PyPI wheel pulls ~3 GB of CUDA libraries); the partial install was removed. **No real NLI, embedding or reranker model was loaded or measured.** |
| Real LLM calls | ❌ none (no API key/network to a provider). All LLM behaviour was tested with scripted replies. |
| Frontend `tsc --noEmit` | ✅ run – OK |
| Frontend production build (`npm run build`) | ✅ run – built, 99 modules |
| Docker Compose file | ⚠️ **YAML parsed and structure reviewed only.** `docker compose config/build/up` were **not run** (no Docker in the sandbox). Dockerfiles, health checks (including the Qdrant `bash /dev/tcp` check), `env_file: required: false` and the Tesseract install step are therefore **unverified**. |
| Live server process | ✅ `uvicorn app.main:app_factory --factory` (the container's entry point) was started; `/api/health` and `/api/ready` answered; register + upload worked; with no embedding model available the document failed with a clear, actionable message (no fake success). Qdrant ran in local in-memory mode, not server mode. |
| OCR | ✅ real Tesseract 5.3.4 via PyMuPDF on an image-only PDF (text read correctly; blank page reported, not invented) |
| GitHub Actions | ❌ not run |

## 2. Backend test suite (231 passed)

| File | Tests | What it covers | Real or stand-in components |
|---|---|---|---|
| `test_verifier.py` | 33 | **Lisbon and revenue regressions**, unrelated contradictions ignored, premise = one sentence, currency/percent/unit/direction/version errors, genuine page conflict, incomplete claims, truncation, coreference, thresholds, NLI label mapping + self-test | scripted NLI scores (decision logic); label tests use stubs, not a real model |
| `test_numeric.py` | 47 | currencies, percent vs units, 12.5 vs 125 %, magnitudes (million/bn/lakh), rounding, units/dimensions, dates (ambiguous d/m), ranges, versions/IPs, identifiers, period mismatch | pure logic |
| `test_textutils.py` | 13 | sentence splitting (abbreviations, decimals), stemming/synonyms, anchors, term statistics | pure logic |
| `test_claims.py` | 23 | fragments, pronoun-led claims, JSON/fenced/plain-text answers, citations, insufficiency, caps | pure logic |
| `test_pdf_structure.py` | 14 | two-column order, header/footer/page-number removal, headings, ruled tables → row sentences, hidden white text, white-on-dark kept, hyphenation, lists, exact page numbers in a 40-page PDF, scan + **real OCR**, blank scan page | **real PyMuPDF and Tesseract** |
| `test_chunking.py` | 10 | page boundaries, sentence overlap, section carry-over, heading not stranded, table chunks | pure logic |
| `test_retrieval.py` | 16 | exact/paraphrase/numeric/long-doc retrieval, page filter, **cross-user isolation at Qdrant, BM25 cache and SQL**, deletion, gate for tiny documents, RRF, BM25 two-chunk regression | hashed fake embedder (lexical), **real qdrant-client in-memory engine**, real BM25 |
| `test_security.py` | 14 | logout/token epoch, tampered tokens, login enumeration, unauthenticated routes, other-user 404s, oversize requests, document limit, failed-file reprocessing, warnings exposed, no secrets in `/ready` or errors, headers | real FastAPI app via TestClient |
| `test_workflow.py` | 20 | end-to-end with scripted LLM: verified/partial/abstain, bounded retries (≤ 2), feedback to the retry, best-attempt retention, conflict not retried, no retry when terms absent, widening only when saturated, injection fencing, evidence shown = evidence used | scripted LLM, stand-in NLI/embedder |
| `test_api.py`, `test_units.py` | 19 + 16 | original API/auth/upload/delete behaviour and utilities (retained; `test_units` updated for the new numeric API) | |
| `test_eval_harness.py` | 6 | regression floors on the evaluation (below) | stand-in models |
| `test_real_models.py` | 5 | real label mapping, Lisbon + revenue regressions, contradiction, paraphrase embeddings | **skipped here (models unavailable)** |

Stand-in models = `evaluation/mocks.py`: hashed bag-of-words embedder (lexical, not semantic), overlap-based NLI imitation, extractive LLM baseline. They validate **logic**, not model quality.

## 3. Evaluation (mock mode – pipeline logic only)

Dataset: 9 synthetic PDFs (plain report, ruled table, two-column paper with repeating header/footer, manual, assignment brief, self-contradicting policy, scanned memo, 40-page report, incomplete sheet), 53 questions (44 answerable, 9 unanswerable), 56 claim cases. Full file: `evaluation_results/mock_run.md` / `.json`. **These numbers are not model accuracy and must not be quoted as production performance. The verification logic was developed while looking at this dataset: they are development scores, not held-out.**

| Retrieval (top-5 chunks, 44 questions) | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR |
|---|---|---|---|---|---|
| hybrid | 0.886 | 1.0 | 1.0 | 0.474 | 0.955 |
| BM25 only | 0.955 | 1.0 | 1.0 | 0.474 | 0.989 |
| dense (hashed words, not semantic) | 0.886 | 1.0 | 1.0 | 0.474 | 0.955 |

Average retrieval 2.8 ms (tiny corpus, in-memory Qdrant, stand-in embedder).

| Claim verification | Claims | Accuracy | SUPPORTED P / R | CONTRADICTED P / R | False-support rate |
|---|---|---|---|---|---|
| all | 56 | 0.982 | 1.0 / 1.0 | 1.0 / 0.952 | 0.0 (0 of 31 non-supported) |
| not needing language understanding | 52 | 1.0 | 1.0 / 1.0 | 1.0 / 1.0 | 0.0 |
| needing a real NLI (`semantic`) | 4 | 0.75 | 1.0 / 1.0 | 1.0 / 0.5 | 0.0 |

The document conflict (3 vs 2 remote days) was reported as a conflict (1/1). Citation page accuracy for claims with expected evidence: 0.978.

| End to end (extractive mock LLM) | Value |
|---|---|
| Answer accuracy | 0.773 – the misses are the **mock LLM abstaining** (its crude "insufficient" rule), not pipeline errors; a real LLM is needed for a meaningful figure |
| Citation (page) accuracy | 1.0 (n = 33 verified answers) |
| Abstention accuracy | 0.811; **unanswerable questions shown as verified: 0 / 9** |
| Avg response / retrieval | 0.01 s / 2.8 ms (mock; no real model latency measured) |
| Retry frequency / avg LLM calls | 0.0 / 0.98 |

**Not measured at all:** real NLI/embedding accuracy, real LLM answer accuracy, real model latency, Docker performance. Run `python scripts/evaluate.py --mode real` (and `scripts/calibrate.py --mode real`) to obtain them.

## 4. Defects found by testing during development (all fixed, all covered)
Windows of three sentences produced false support; deterministic number conflicts from loosely related sentences produced false document conflicts; document-frequency statistics in small documents removed the meaningful terms; euros-vs-dollars lowered relevance of the very sentence that contradicts; an absolute BM25 gate abstained on answerable questions in short documents; **BM25Okapi returned non-positive scores in short documents so keyword retrieval silently did nothing** (BM25-only Recall@1 0.614 → 0.955 with BM25Plus); the header/footer remover could delete a real repeated one-line paragraph (now restricted to short margin blocks); years followed by a noun ("2023 revenue") were read as counts; currency codes before numbers ("EUR 82 million") were not recognised.

## 5. Known limitations (testing-related)
* Thresholds `ENTAILMENT 0.80 / CONTRADICTION 0.85 / CONFLICT 0.90`, relevance and gate settings are starting values; the calibration tool was exercised only with stand-in models.
* The 4 `semantic` cases (negation, "spring" vs "autumn") can only be judged with a real NLI model.
* Tables: ruled tables only; units declared outside the table are not attached (one dataset case documents this conservative NOT_ESTABLISHED).
* Test PDFs are synthetic and small; real-world layouts (borderless tables, rotated text, forms) are untested.
* Concurrency was tested only functionally (threading locks, TestClient); no load test.
