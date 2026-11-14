# Security notes

| Topic | What is done | Where tested |
|---|---|---|
| Authentication | scrypt password hashes; HMAC-signed tokens with expiry and a per-user **epoch** (sign-out invalidates all tokens); login throttled per IP and per IP+email; constant-time-ish login for unknown emails | `test_security.py`, `test_api.py` |
| Authorisation | every document route checks ownership; foreign ids answer 404; Qdrant filter, BM25 cache key and SQL all include the owner | `test_security.py`, `test_retrieval.py` |
| Uploads | real `%PDF` header, size limit (early `Content-Length` check + hard check), page limit, encrypted/corrupt PDFs refused, stored under random names, per-user document limit | `test_units.py`, `test_security.py` |
| Prompt injection | excerpts are untrusted data between random-nonce markers (a PDF cannot forge the closing marker); the system prompt forbids following excerpt instructions; hidden (white/microscopic) PDF text is dropped and reported; claims are verified against the PDF, so an injected "fact" is not verified unless it is in the PDF | `test_workflow.py`, `test_pdf_structure.py` |
| Secrets | API key only from the environment, never returned by any endpoint, logs redact secrets; provider error text is passed on but requests (with keys) are not | `test_security.py` |
| Headers | `nosniff`, `no-store`, `no-referrer`, CSP/frame protection in nginx | `test_security.py` |
| Data lifecycle | delete removes file, vectors, chunks and metadata; an ingestion that finishes after a deletion cleans up after itself | `test_api.py`, `test_retrieval.py` |

**Not covered**: HTTPS (put a TLS proxy in front for anything beyond localhost), email-based password reset, per-tenant encryption at rest, malware scanning of PDFs beyond parsing with PyMuPDF. If you use an external LLM provider, retrieved excerpts leave your machine.
