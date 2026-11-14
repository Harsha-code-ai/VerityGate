# Architecture

```
PDF upload ─► validate (header, size, pages, password) ─► background ingestion
   extract (PyMuPDF dict blocks: columns, headings, lists, tables, hidden text, header/footer removal; OCR for pages without text)
   ─► units (heading | para | list | table) per physical page ─► chunks (≤ page, sentence-aligned, overlap, section label)
   ─► embeddings (section + text) ─► Qdrant (owner_id, doc_id, page payload) + SQLite chunks + BM25 index (per owner+doc)

question ─► retrieve (dense + BM25, synonym-normalised/stemmed terms, RRF, optional cross-encoder) ─► gate (dense | bm25 | query-term coverage)
        ─► generate (JSON claims with [S#] sources; excerpts fenced with a random nonce as untrusted data)
        ─► parse claims ─► completeness check (fragments / pronoun-led claims are not verified as claims)
        ─► verify: sentence units ─► relevance ─► NLI (premise = ONE PDF sentence, hypothesis = claim) + typed quantity checks ─► decision
        ─► retry policy (≤ 2) ─► best attempt ─► answer object
```

## Modules (backend/app)
| File | Responsibility |
|---|---|
| `pdf.py` | validation + structure-aware extraction + OCR; `ExtractionReport(pages, method, warnings)` |
| `chunking.py` | heading-aware, sentence-aligned chunking; never crosses a page |
| `textutils.py` | normalisation, sentence splitting (abbreviations, decimals), stemming, synonym folding, entity anchors |
| `retrieval.py` | hybrid retrieval, owner-keyed BM25 cache, evidence gate, term statistics |
| `evidence.py` | sentence units and claim↔sentence relevance |
| `numeric.py` | typed quantities (money+currency, percent, units, dates, years, ranges, versions/IPs) and comparison |
| `claims.py` | JSON/text answer parsing, citations, claim completeness |
| `verifier.py` | NLI wrapper (label resolution + self-test), decision logic |
| `generation.py` | system/user prompts, injection fencing |
| `workflow.py` | LangGraph: retrieve → generate → verify → retry/finalize; response shaping |
| `services.py`, `main.py`, `db.py`, `security.py`, `llm.py`, `vectorstore.py`, `embeddings.py`, `config.py` | wiring, API, SQLite (+ additive migrations), auth, LLM clients, Qdrant, models, settings |

## Verification decision (verifier.py)
For each claim, candidate evidence = sentences of the retrieved passages ranked by relevance (idf-weighted term coverage of the claim **without its quantities**, entity anchors, embedding similarity). Only the best `VERIFY_MAX_UNITS` go to the NLI model.

* **Supporting sentence**: related + entailment ≥ threshold and > contradiction + every number/date/currency/percentage/version of the claim is found in that sentence (years may come from the surrounding text) + no direction reversal + identifier anchors such as `X200` present.
* **Contradicting sentence**: *strictly* relevant (same entities, same subject) and either NLI contradiction ≥ threshold, or a typed conflict (different value/currency/date/version with exactly one comparable candidate; ignored when the claim names another period or the NLI model calls the pair unrelated).
* SUPPORTED if ≥1 supporting and no hard contradiction (contradiction ≥ `CONFLICT_THRESHOLD`) from a relevant sentence; with both → `conflict` (NOT_ESTABLISHED, both shown); CONTRADICTED if only contradicting; otherwise NOT_ESTABLISHED with a reason code.
* Contradiction scores from sentences that are not about the claim are discarded: this is the fix for "Lisbon → CONTRADICTED".

## Retry policy (workflow.py)
Bounded by `MAX_RETRIES` (hard cap 2). Retry when claims are incomplete/unsupported and a retry can help; widen `top_k` only if retrieval was saturated; one widened retry when the model says "insufficient" but the question's distinctive terms occur in the retrieved text; **no** retry when those terms occur nowhere, or when the document contradicts itself. A later attempt replaces the best one only if strictly better (all supported > more supported > fewer contradicted > fewer unestablished).

## Data model
SQLite: `users(token_epoch)`, `documents(status, warnings, ocr_pages, …)`, `chunks(kind)`, `audit`. Old databases are migrated additively at startup. Qdrant payload: `owner_id`, `doc_id`, `page`, `section`; every query filters on owner and document.
