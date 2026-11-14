# VerityGate – Project audit report

Scope: the complete uploaded project `VerityGate-Complete.zip` (≈ 40 source files, 48 tests, Docker/Compose, React frontend). Every backend module, test, Dockerfile, compose file, frontend source file, script and doc was read before anything was changed. All "before" evidence below was produced by running the **unmodified original code**; "after" evidence by running the new code (`TESTING_REPORT.md` lists the runs).

**Honesty note.** The two reported failures were reproduced and fixed at their *logical* root cause using scripted NLI scores. The real Hugging Face models could not be downloaded in the build sandbox, so (a) it is *suspected, not measured*, that whole-chunk NLI premises also diluted real entailment scores, and (b) the new thresholds are uncalibrated starting values. Both are called out again under "What remains unresolved".

## 1. Inventory and execution flow (original)

`pdf.py` (plain `page.get_text("text")` per page) → `chunking.py` (≈180-word word windows) → `embeddings.py`/`vectorstore.py` (Qdrant) + BM25 (`retrieval.py`) → RRF → `generation.py` → LLM → `claims.py` (sentence split, ≥ 3 words) → `verifier.py` (NLI on **whole chunks**, `numeric.py` digit comparison) → `workflow.py` (LangGraph, 2 retries) → `main.py` (FastAPI) → React UI. Tests: 48, all passing – none exercised the reported failures.

## 2. Confirmed bugs, root causes, fixes

### 2.1 Final verification decision (the critical issue)

| # | Finding | Evidence (original code) | Root cause | Fix |
|---|---|---|---|---|
| V1 | **Lisbon → CONTRADICTED** | Passage 1 entails the claim (0.95); an unrelated Madrid passage scores contradiction 0.97 → `CONTRADICTED: A passage on page 2 contradicts the claim (model score 0.97)` | The verifier took the **maximum contradiction over all retrieved passages** and let `con ≥ 0.8 and con > ent` override entailment *from a different passage*. Relevance was never checked. | Sentence-level evidence; only sentences that are *about* the claim (idf-weighted coverage of the claim without its quantities, entity anchors, embedding similarity) may vote; support and contradiction are decided per sentence; unrelated contradiction scores are discarded (`verifier.py`, `evidence.py`). Test: `test_unrelated_contradiction_never_overrides_supporting_evidence`. |
| V2 | **Revenue 12.5 % → NOT_ESTABLISHED** | `The number '2022' in the claim does not appear in the supporting passage.` for evidence "Revenue grew 12.5 percent year over year." | `numeric.py` required every digit sequence of the claim to occur literally in the single best passage, incl. years stated only in a heading/neighbouring sentence. | Typed quantities; years may be covered by the surrounding text/heading; missing quantities only block *support*, never create contradictions; period-mismatch rule. Tests: `test_revenue_increase_*`, `test_year_*`. |
| V3 | Numeric blind spots | `82 million dollars` vs `82 million euros` → `ok=True`; `50 units` vs `50%` → `ok=True`; `fell 5%` vs `rose 5%` → `ok=True`; `15 June 2019` vs `16 June 2019` → only "missing", no typed conflict | Currency, unit, direction and date structure were not modelled; digits only. | New typed engine (money+currency, percent/pp/bps, units with dimensions, dates incl. ambiguous d/m, years, ranges, version/IP codes, magnitudes incl. lakh/crore, rounding tolerance for the claim). 59 numeric tests. |
| V4 | NLI premise = whole chunk | `verifier.py` passed ~180-word chunks (several facts) as premise | Mixed facts in one premise; contradiction from one fact leaks to the claim; long premises dilute entailment. *Effect on real scores: suspected, not measured here.* | Premise = **one PDF sentence** (plus the previous sentence when it starts with "It/They/This…"); hypothesis = the claim. Test `test_nli_receives_one_sentence_as_premise_and_the_claim_as_hypothesis`. |
| V5 | Label mapping / truncation risk | Labels taken from model config without validation; truncation silent | A swapped label order silently turns "supported" into "contradicted". | `resolve_nli_labels` (names → explicit `NLI_LABEL_ORDER` → known families → **refuse**), 2-class models supported, a **label self-test on two unambiguous pairs at load (fail closed)**, truncated premises are flagged and ignored. Tested with stubs; **real-model run pending** (`test_real_models.py`). |
| V6 | Entity/identifier confusion | `Router X200` claim could be supported by an `X300` sentence | No entity awareness. | Identifier anchors (letters+digits) must appear in a supporting sentence; entity anchors gate contradictions. |
| V7 | A real document conflict was unrepresentable | – | – | `conflict=True`: supporting **and** hard-contradicting relevant sentences → NOT_ESTABLISHED, both shown, no retry, answer-level warning. |

### 2.2 Other confirmed defects

| Area | Finding | Fix |
|---|---|---|
| Claims | Fragments such as "15 June 2019." (≥ 3 words) were accepted as claims; "INSUFFICIENT_EVIDENCE" anywhere in the text aborted the answer | Strict JSON claims with sources; completeness check (subject, verb, not number-only, no leading pronoun); exact insufficiency handling |
| Retrieval | BM25 cache keyed by `doc_id` only; absolute BM25 gate (`≥ 1.0`) is meaningless for short documents – found by the new evaluation (it abstained on answerable questions); no stemming/synonym handling ("start operating" vs "opened") | Cache keyed by (owner, doc); corpus-size-independent coverage gate; stemmed, synonym-folded terms; page filter; term statistics |
| PDF | Plain text only: multi-column pages interleaved, headers/footers inside chunks, tables flattened, hidden text read, hyphenated line breaks, no OCR warnings | Structure-aware extraction (see `pdf.py`); tables → row sentences; repeated headers/footers and page numbers removed (short margin blocks only); white/tiny hidden text ignored and reported; OCR (Tesseract) per page with quality filter and warnings |
| Chunking | Word windows could cut sentences, ignored headings/tables | Sentence-aligned, heading-aware, table-aware; never crosses a page (tested with 40-page document) |
| Workflow | Retried even when the document clearly lacked the information; widened `top_k` when it could not help; stored claim ids by re-building the prompt | Purposeful retry policy (§ ARCHITECTURE), best-attempt key, `id_map` stored in state |
| Generation | Plain-text answers; fence could be closed by crafted PDF text | JSON contract; random-nonce fences; delimiter neutralisation |
| Evidence display | Top 3 items without roles | All decisive sentences with role (supporting/contradicting/related), scores, notes; UI groups them |
| Security | Login timing revealed registered emails; tokens could not be revoked; **login rate limit saw only the nginx container IP** (all users shared one bucket); no per-user document cap; ingestion racing with deletion left orphan vectors; chunk insert not atomic | Dummy-hash login; token epoch + `/auth/logout`; `FORWARDED_ALLOW_IPS` + per-IP-and-email limits; cap; post-index existence check; transactional `replace_chunks`/`delete_document_rows` |
| LLM | Newer OpenAI models reject `max_tokens`/`temperature`; opaque provider errors; Ollama JSON not requested | Automatic parameter fallback, provider error text, one retry on 429/5xx, Ollama `format: json` |
| Docker/ops | `env_file: .env` made `compose up` **fail without .env**; Qdrant had no health check; image excluded tests (so "run tests in Docker" was impossible); README recommended `docker compose down -v` as the reset; Python 3.11 image vs 3.12-tested code; OCR required manual Docling edits | Optional `env_file`, Qdrant health check + `service_healthy`, frontend health check, `runtime`/`test` targets, `--profile test`, Python 3.12, Tesseract in image (build arg), model preload, no `-v` advice anywhere |
| Docs | Evaluation doc quoted a 6-chunk smoke test | Replaced by a real evaluation package + methodology |

### 2.3 Suspected / not confirmed
* Real NLI score dilution by long premises (V4) and real-model behaviour on alphanumeric identifiers (X200 vs X300): handled structurally, unmeasured.
* Thresholds `0.80/0.85/0.90`: starting values; `scripts/calibrate.py` exists to choose them from data.
* Docker build/run, Qdrant server mode, GitHub Actions: not executable in the sandbox.

## 3. Files changed
**Rewritten:** `app/verifier.py`, `numeric.py`, `claims.py`, `generation.py`, `workflow.py`, `pdf.py`, `chunking.py`, `retrieval.py`, `services.py`, `embeddings.py`. **Edited:** `app/config.py`, `db.py` (additive migrations), `llm.py`, `main.py`, `security.py`, `vectorstore.py`. **New:** `app/textutils.py`, `app/evidence.py`, `backend/evaluation/` (corpus, mocks, metrics, run, calibrate, data), tests `test_numeric/textutils/verifier/claims/pdf_structure/chunking/retrieval/security/eval_harness/real_models.py`, `scripts/calibrate.py`. **Updated:** `docker-compose.yml`, `backend/Dockerfile`, `.dockerignore`, `frontend/nginx.conf`, `.env.example`, CI, `ruff.toml`, `pyproject.toml`, `tests/conftest.py`, `test_units.py`, `test_workflow.py`, frontend `types.ts`, `api.ts`, `App.tsx`, `AnswerView.tsx`, `StatusBadge.tsx`, `DocumentsPage.tsx`, all docs, `scripts/*`. **Removed:** `eval/` (6-chunk smoke fixtures, superseded). `test_api.py` and the original API behaviour are unchanged and still pass.

## 4. Process notes (what the evaluation harness found in my own first draft)
Building the evaluation corpus exposed defects in the first version of the new verifier, all fixed and covered by tests: three-sentence windows created false support by mixing the Madrid and Lisbon sentences (windows removed); a deterministic number conflict from a loosely related sentence ("prior work used > 20 million parameters") created false document conflicts (now needs NLI agreement); document-frequency statistics in tiny documents removed the meaningful terms (now "unknown" below 6 chunks); the "euros vs dollars" difference counted *against* relevance (quantities are stripped before relevance); an absolute BM25 gate abstained on answerable questions in short documents.

## 5. What remains unresolved
1. **Real-model accuracy is unmeasured** (no model download in the sandbox). Run `python scripts/evaluate.py --mode real` and `scripts/calibrate.py --mode real` and `pytest -m real_models` before relying on the system.
2. The verification logic was developed against the same synthetic dataset it is scored on (development scores, not held-out).
3. Tables: only ruled tables; units declared outside a table are not attached to cells (claims quoting "million euros" for a table value come out NOT_ESTABLISHED – conservative).
4. No arithmetic inference (derived percentages), no cross-page sentence joining, no printed page labels.
5. Claim completeness is a heuristic; an LLM that ignores the JSON contract falls back to sentence splitting.
6. Docker images were not built; compose was validated structurally only.
7. Scanned-page OCR quality depends on Tesseract; word-quality filtering is a heuristic.
