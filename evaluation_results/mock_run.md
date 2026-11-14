# VerityGate evaluation (mock mode)

- Models: **MOCK (hashed embeddings, overlap NLI, extractive LLM)**
- Date (UTC): 2026-11-05T08:44:06+00:00  |  Python 3.12.3
- Dataset: 9 synthetic PDFs, 53 questions, 56 claim cases  |  OCR available: True

> **These numbers come from deterministic stand-in models. They test the pipeline logic, not model accuracy, and must not be quoted as production accuracy.**

## Retrieval (answerable questions, top-5 chunks)

| Mode | Recall@1 | Recall@3 | Recall@5 | Precision@5 | MRR | Avg ms |
|---|---|---|---|---|---|---|
| hybrid | 0.886 | 1.0 | 1.0 | 0.474 | 0.955 | 3.119 |
| bm25 | 0.955 | 1.0 | 1.0 | 0.474 | 0.989 | 0.175 |
| dense | 0.886 | 1.0 | 1.0 | 0.474 | 0.955 | 2.505 |

Hybrid retrieval misses (question ids): none

## Claim verification

**All cases** (56 claims): accuracy 0.982, SUPPORTED P/R 1.0/1.0, CONTRADICTED P/R 1.0/0.952, false-support rate 0.0 (0), conflicts reported 1/1

**Cases not needing language understanding** (52 claims): accuracy 1.0, SUPPORTED P/R 1.0/1.0, CONTRADICTED P/R 1.0/1.0, false-support rate 0.0 (0), conflicts reported 1/1

**Cases needing real NLI (`semantic`)** (4 claims): accuracy 0.75, SUPPORTED P/R 1.0/1.0, CONTRADICTED P/R 1.0/0.5, false-support rate 0.0 (0), conflicts reported 0/0

Citation page accuracy (claims with expected evidence): 0.978

Verification mismatches:

- `c49` expected CONTRADICTED, got NOT_ESTABLISHED (neutral_evidence) [semantic]: Construction will start in the spring.

## End-to-end

- Questions: 53 (44 answerable, 9 unanswerable)
- Answer accuracy: 0.773  |  Citation accuracy: 1.0 (n=33)
- Abstention accuracy: 0.811  |  Unanswerable questions answered as verified: 0/9
- Avg response: 0.01 s  |  Avg retrieval: 2.796 ms  |  Retry frequency: 0.0  |  Avg LLM calls: 0.981

End-to-end mismatches:

- `q07` (paraphrase_hard) status ABSTAINED: By how much did CO2 output fall between 2020 and 2023?
- `q08` (exact) status ABSTAINED: What share of electricity came from renewable sources in 2023?
- `q09` (multi_page) status ABSTAINED: Which offices does the company have and what was its 2023 revenue?
- `q12` (table) status ABSTAINED: What was Asia's revenue in 2023?
- `q13` (table) status ABSTAINED: What was the revenue of Europe in 2022?
- `q15` (exact) status ABSTAINED: In which currency are the amounts stated?
- `q19` (numeric) status ABSTAINED: How many parameters does the model use?
- `q23` (exact) status ABSTAINED: What limitations does the study mention?
- `q30` (paraphrase) status ABSTAINED: How long does the warranty last?
- `q33` (exact) status ABSTAINED: What is the word limit for the report?
