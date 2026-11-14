import io

import pytest
from conftest import make_pdf, register, upload


def ask(client, headers, doc_id, q="When did the Lisbon office open?"):
    return client.post(f"/api/documents/{doc_id}/ask", headers=headers, json={"question": q})


# ---- health / readiness ----
def test_health_and_ready(client):
    assert client.get("/api/health").json()["status"] == "ok"
    r = client.get("/api/ready")
    assert r.status_code == 200 and r.json()["checks"] == {"database": True, "qdrant": True} and r.json()["llm_configured"] is True


def test_ready_reports_missing_llm_config(client, svc):
    svc.settings.llm_provider = ""
    body = client.get("/api/ready").json()
    assert body["llm_configured"] is False and "LLM_PROVIDER" in body["llm_problems"][0]


def test_cors_restricted(client):
    ok = client.options("/api/health", headers={"Origin": "http://allowed.example", "Access-Control-Request-Method": "GET"})
    bad = client.options("/api/health", headers={"Origin": "http://evil.example", "Access-Control-Request-Method": "GET"})
    assert ok.headers.get("access-control-allow-origin") == "http://allowed.example"
    assert "access-control-allow-origin" not in bad.headers


# ---- auth ----
def test_auth_required_and_login(client):
    assert client.get("/api/documents").status_code == 401
    assert client.get("/api/documents", headers={"Authorization": "Bearer junk"}).status_code == 401
    register(client, "bob@example.com")
    assert client.post("/api/auth/login", json={"email": "bob@example.com", "password": "wrong-password-1"}).status_code == 401
    assert client.post("/api/auth/login", json={"email": "bob@example.com", "password": "correct-horse-battery"}).status_code == 200
    assert client.post("/api/auth/register", json={"email": "bob@example.com", "password": "correct-horse-battery"}).status_code == 409
    assert client.post("/api/auth/register", json={"email": "x@y.zz", "password": "short"}).status_code == 422


# ---- upload ----
def test_upload_indexes_document(client, auth, svc):
    doc = upload(client, auth)
    d = client.get(f"/api/documents/{doc['id']}", headers=auth).json()
    assert d["status"] == "ready" and d["pages"] == 3 and d["chunk_count"] >= 3 and d["error"] is None
    assert svc.store.count(doc["id"]) == d["chunk_count"]
    stored = list(svc.settings.uploads_dir.iterdir())
    assert len(stored) == 1 and stored[0].name != "report.pdf" and stored[0].suffix == ".pdf"


def test_upload_duplicate_reuses_index(client, auth):
    data = make_pdf(__import__("conftest").REPORT_PAGES)  # same bytes twice (PDF generation is not deterministic)
    a = client.post("/api/documents", headers=auth, files={"file": ("a.pdf", data, "application/pdf")}).json()
    b = client.post("/api/documents", headers=auth, files={"file": ("again.pdf", data, "application/pdf")})
    assert b.status_code == 200 and b.json()["duplicate"] is True and b.json()["id"] == a["id"]


@pytest.mark.parametrize("content,status", [(b"not a pdf at all", 422), (b"", 422)])
def test_upload_rejects_non_pdf_even_with_pdf_name(client, auth, content, status):
    r = client.post("/api/documents", headers=auth, files={"file": ("fake.pdf", io.BytesIO(content), "application/pdf")})
    assert r.status_code == status


def test_upload_rejects_oversize(client, auth, svc):
    svc.settings.max_upload_mb = 0
    r = client.post("/api/documents", headers=auth, files={"file": ("a.pdf", make_pdf(["hello world text here"]), "application/pdf")})
    assert r.status_code in (413, 422) and "limit" in r.json()["detail"] or "larger" in r.json()["detail"]


def test_scanned_pdf_reports_clear_error(client, auth):
    doc = upload(client, auth, pages=["", ""], name="scan.pdf")
    d = client.get(f"/api/documents/{doc['id']}", headers=auth).json()
    assert d["status"] == "failed" and "No usable text" in d["error"]
    assert ask(client, auth, doc["id"]).status_code == 409


# ---- authorization / isolation ----
def test_documents_are_isolated_between_users(client, auth):
    doc = upload(client, auth)
    other = register(client, "mallory@example.com")
    assert client.get("/api/documents", headers=other).json() == []
    assert client.get(f"/api/documents/{doc['id']}", headers=other).status_code == 404
    assert ask(client, other, doc["id"]).status_code == 404
    assert client.delete(f"/api/documents/{doc['id']}", headers=other).status_code == 404
    assert client.get(f"/api/documents/{doc['id']}", headers=auth).status_code == 200  # untouched
    assert client.get("/api/documents/../../etc/passwd", headers=auth).status_code == 404


def test_retrieval_never_crosses_users(client, auth, svc):
    mine = upload(client, auth)
    other = register(client, "eve@example.com")
    theirs = upload(client, other, pages=["Secret plans mention Lisbon office openings in 2031 and beyond for everyone."], name="x.pdf")
    owner_id = svc.db.one("SELECT owner_id FROM documents WHERE id=?", (mine["id"],))["owner_id"]
    passages = svc.retriever.retrieve(owner_id, mine["id"], "Lisbon office", 10)
    assert passages and all(p.doc_id == mine["id"] for p in passages)
    assert svc.retriever.retrieve(owner_id, theirs["id"], "Lisbon office", 10) == []  # wrong owner -> nothing


# ---- delete ----
def test_delete_removes_file_rows_and_vectors(client, auth, svc):
    doc = upload(client, auth)
    assert client.delete(f"/api/documents/{doc['id']}", headers=auth).status_code == 204
    assert list(svc.settings.uploads_dir.iterdir()) == []
    assert svc.db.one("SELECT 1 AS x FROM chunks WHERE doc_id=?", (doc["id"],)) is None
    assert svc.store.count(doc["id"]) == 0
    assert client.get(f"/api/documents/{doc['id']}", headers=auth).status_code == 404


def test_delete_keeps_metadata_if_vector_delete_fails(client, auth, svc, monkeypatch):
    doc = upload(client, auth)
    monkeypatch.setattr(svc.store, "delete_document", lambda *a: (_ for _ in ()).throw(ConnectionError("down")))
    assert client.delete(f"/api/documents/{doc['id']}", headers=auth).status_code == 502
    assert client.get(f"/api/documents/{doc['id']}", headers=auth).status_code == 200


# ---- ask: errors and config ----
def test_ask_without_llm_config_is_helpful(client, auth, svc):
    doc = upload(client, auth)
    svc.settings.llm_provider = ""
    r = ask(client, auth, doc["id"])
    assert r.status_code == 503 and r.json()["detail"]["code"] == "llm_not_configured"


def test_ask_validates_question(client, auth):
    doc = upload(client, auth)
    assert client.post(f"/api/documents/{doc['id']}/ask", headers=auth, json={"question": "hi"}).status_code == 422
    assert client.post(f"/api/documents/{doc['id']}/ask", headers=auth, json={"question": "x" * 2000}).status_code == 422


def test_llm_failure_becomes_502(client, auth, svc):
    from app.llm import LLMError

    class Boom:
        def complete(self, *a):
            raise LLMError("The LLM provider rejected the credentials (HTTP 401). Check LLM_API_KEY.")

    svc.llm_factory = lambda: Boom()
    doc = upload(client, auth)
    r = ask(client, auth, doc["id"])
    assert r.status_code == 502 and "LLM_API_KEY" in r.json()["detail"]["message"]


def test_rate_limit(client, auth, svc):
    from app.security import RateLimiter

    svc.limiter = RateLimiter(2)
    doc = upload(client, auth)
    codes = [ask(client, auth, doc["id"]).status_code for _ in range(3)]
    assert 429 in codes


def test_audit_has_no_question_text_or_secrets(client, auth, llm):
    doc = upload(client, auth)
    ask(client, auth, doc["id"], "When did the Lisbon office open?")
    rows = client.get("/api/audit", headers=auth).json()
    blob = str(rows)
    assert any(r["event"] == "ask" for r in rows)
    assert "Lisbon office open" not in blob and "test-key" not in blob and "Revenue increased" not in blob
