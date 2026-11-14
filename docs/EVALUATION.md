# Evaluation

Code: `backend/evaluation/` (`corpus.py` builds nine synthetic PDFs; `data/questions.json` 53 questions with expected pages and answers; `data/claims.json` 56 claim cases with expected labels; `run.py`, `calibrate.py`, `metrics.py`, `mocks.py`).

```
python scripts/evaluate.py --mode mock                       # stand-in models: tests the PIPELINE LOGIC only
python scripts/evaluate.py --mode real --out evaluation_results/real.json   # real embedding + NLI + your LLM
python scripts/calibrate.py --mode real                      # threshold sweep, recommends zero-false-support settings
```

## Metrics
Retrieval Recall@1/3/5 (share of gold answer strings found on the right page in the top-K chunks), Hit@K, Precision@5, MRR for hybrid, BM25 and dense separately; claim verification: SUPPORTED and CONTRADICTED precision/recall, **false-support rate**, confusion matrix, conflicts reported; end to end: answer accuracy, citation (page) accuracy, abstention accuracy, unanswerable-but-verified count, average response and retrieval time, retry frequency, LLM calls.

## Methodology and limits
* Documents are **synthetic** (9 PDFs, 1–40 pages): plain report, ruled table, two-column paper with repeating header/footer, technical manual, assignment brief, self-contradicting policy, scanned memo (real OCR), 40-page report, incomplete-information sheet.
* Cases flagged `semantic` need genuine language understanding (negation, "autumn" vs "spring"); in mock mode they are reported separately and are expected to fail.
* The verification logic was iterated while looking at this dataset: its scores are **development** scores, not a held-out test. Use `--mode real` on your own documents before trusting any number.
* The mock LLM is an extractive baseline, the mock embedder is hashed words (not semantic), the mock NLI is overlap-based. Mock results must not be quoted as production accuracy. Real-model results are produced only when you run `--mode real`.
