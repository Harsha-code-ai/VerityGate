# API reference

Base path `/api`. Interactive docs (Swagger UI) at http://localhost:8000/docs. Authenticated routes need `Authorization: Bearer <token>`.

| Method & path | Auth | Description |
|---|---|---|
| `GET /api/health` | no | Liveness: `{"status":"ok"}` |
| `GET /api/ready` | no | Readiness: database + Qdrant checks (503 if down), `llm_configured`, `llm_problems`, `models` (which model runs where; never any key), `ocr` |
| `POST /api/auth/register` | no | `{email, password (≥10 chars)}` → `{token, email}`; 403 if registration disabled, 409 if the email exists |
| `POST /api/auth/login` | no | Same body → `{token, email}`; 401 bad credentials (identical answer for unknown email), 429 throttled (per IP and per IP+email) |
| `POST /api/auth/logout` | yes | 204. Invalidates **every** token issued to this user so far |
| `GET /api/auth/me` | yes | Current user |
| `POST /api/documents` | yes | multipart field `file` (PDF). 202 + document (`pending`); 200 with `duplicate: true` if already uploaded; 202 with `reprocessing: true` if the same file had failed before; 409 per-user document limit; 413 too large (also refused early from `Content-Length`); 422 not a valid PDF |
| `GET /api/documents` | yes | Your documents (incl. `warnings`, `ocr_pages`) |
| `GET /api/documents/{id}` | yes | One document (`status`: pending / processing / ready / failed, `error`, `warnings`) |
| `GET /api/documents/{id}/pages/{n}` | yes | The indexed text of page *n* (chunks), to check a citation against what VerityGate read |
| `DELETE /api/documents/{id}` | yes | 204; 502 if the vector index could not be cleaned (nothing removed) |
| `POST /api/documents/{id}/ask` | yes | `{question}` → answer object. 409 not ready, 503 `llm_not_configured`, 502 `llm_error`, 429 throttled |
| `GET /api/audit?limit=50` | yes | Your audit events (no document text) |

Documents of other users answer **404**, exactly like documents that do not exist.

## Answer object
```json
{
  "status": "VERIFIED | PARTIALLY_VERIFIED | ABSTAINED",
  "abstain_reason": null,
  "message": "human-readable explanation",
  "verified_answer": "supported claims only, each followed by [p.N] (null if none)",
  "claims": [{
    "text": "The Lisbon office opened on 15 June 2019.",
    "status": "SUPPORTED | CONTRADICTED | NOT_ESTABLISHED",
    "reason": "Page 1 states this (entailment 0.97).",
    "reason_code": "supported | contradicted_nli | contradicted_quantity | contradicted_direction | conflict | no_relevant_evidence | neutral_evidence | quantity_not_stated | incomplete_claim",
    "conflict": false, "considered": 6,
    "cited_ids": ["S1"], "citation_valid": true, "warnings": [],
    "best_entailment": 0.97, "best_contradiction": 0.01,
    "evidence": [{
      "chunk_id": "...", "page": 1, "text": "the exact PDF sentence", "role": "supporting | contradicting | related",
      "entailment": 0.97, "contradiction": 0.01, "relevance": 0.8, "cited_by_model": true,
      "note": "quantity/rounding remark", "context": "surrounding sentences"
    }]
  }],
  "unverified_claims": ["..."],
  "conflicts": [{"claim": "...", "reason": "...", "evidence": []}],
  "attempts": [{"attempt": 0, "top_k": 6, "passages": 6, "claims": 2, "supported": 1, "contradicted": 0, "not_established": 1, "conflicts": 0, "answer_format": "json"}],
  "best_attempt": 0, "retries_used": 0, "llm_calls": 1, "latency_seconds": 2.4, "retrieval_ms": 12.5, "warnings": [], "sources": [1]
}
```
`abstain_reason` (when `ABSTAINED`): `no_relevant_passages`, `model_insufficient`, `no_claims`, `unverifiable`, `document_conflict`.
Evidence shown is exactly the evidence the decision used (the same sentences, scores and pages).
