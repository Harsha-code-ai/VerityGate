"""Authentication, authorisation, data isolation and upload hardening."""
import json

from conftest import REPORT_PAGES, make_pdf, register, upload


def test_logout_invalidates_every_existing_token(client):
    h = register(client, "carol@example.com")
    assert client.get("/api/auth/me", headers=h).status_code == 200
    assert client.post("/api/auth/logout", headers=h).status_code == 204
    assert client.get("/api/auth/me", headers=h).status_code == 401  # the old token is dead
    r = client.post("/api/auth/login", json={"email": "carol@example.com", "password": "correct-horse-battery"})
    assert r.status_code == 200
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {r.json()['token']}"}).status_code == 200


def test_tampered_or_foreign_tokens_are_rejected(client):
    h = register(client, "dave@example.com")
    tok = h["Authorization"].split()[1]
    assert client.get("/api/auth/me", headers={"Authorization": f"Bearer {tok[:-3]}abc"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Bearer garbage"}).status_code == 401
    assert client.get("/api/auth/me", headers={"Authorization": "Basic abc"}).status_code == 401


def test_login_does_not_reveal_whether_an_email_exists(client):
    register(client, "erin@example.com")
    a = client.post("/api/auth/login", json={"email": "erin@example.com", "password": "wrong-password-123"})
    b = client.post("/api/auth/login", json={"email": "nobody@example.com", "password": "wrong-password-123"})
    assert a.status_code == b.status_code == 401 and a.json() == b.json()


def test_every_document_route_requires_authentication(client, auth):
    doc = upload(client, auth)
    for method, url in [("get", "/api/documents"), ("get", f"/api/documents/{doc['id']}"), ("delete", f"/api/documents/{doc['id']}"),
                        ("post", f"/api/documents/{doc['id']}/ask"), ("get", f"/api/documents/{doc['id']}/pages/1"), ("get", "/api/audit")]:
        assert getattr(client, method)(url).status_code == 401, url


def test_other_users_cannot_read_ask_page_or_delete(client, auth):
    doc = upload(client, auth)
    other = register(client, "mallory@example.com")
    assert client.get(f"/api/documents/{doc['id']}/pages/1", headers=other).status_code == 404
    assert client.post(f"/api/documents/{doc['id']}/ask", headers=other, json={"question": "When did the Lisbon office open?"}).status_code == 404
    assert client.delete(f"/api/documents/{doc['id']}", headers=other).status_code == 404
    assert client.get(f"/api/documents/{doc['id']}", headers=auth).status_code == 200  # still there for the owner
    assert client.get("/api/documents", headers=other).json() == []


def test_page_endpoint_returns_what_was_indexed_for_that_page(client, auth):
    doc = upload(client, auth)
    page2 = client.get(f"/api/documents/{doc['id']}/pages/2", headers=auth).json()
    assert page2["page"] == 2 and "45 million dollars" in " ".join(c["text"] for c in page2["chunks"])
    assert "Lisbon" not in " ".join(c["text"] for c in page2["chunks"])


def test_declared_oversize_request_is_refused_before_reading(client, auth, svc):
    svc.settings.max_upload_mb = 1
    big = b"%PDF-1.4\n" + b"0" * (3 * 1024 * 1024)
    r = client.post("/api/documents", headers=auth, files={"file": ("big.pdf", big, "application/pdf")})
    assert r.status_code == 413


def test_per_user_document_limit(client, auth, svc):
    svc.settings.max_documents_per_user = 2
    upload(client, auth, pages=["First distinct document about harbour logistics and cranes."])
    upload(client, auth, pages=["Second distinct document about warehouse staffing levels."])
    r = client.post("/api/documents", headers=auth, files={"file": ("c.pdf", make_pdf(["Third distinct document about fleet maintenance."]), "application/pdf")})
    assert r.status_code == 409 and "limit" in r.json()["detail"]


def test_failed_document_is_reprocessed_when_the_same_file_is_uploaded_again(client, auth, svc, monkeypatch):
    import app.services as services

    calls = {"n": 0}
    real = services.extract_document

    def flaky(*a, **kw):
        calls["n"] += 1
        if calls["n"] == 1:
            raise services.ExtractionError("temporary problem")
        return real(*a, **kw)

    monkeypatch.setattr(services, "extract_document", flaky)
    data = make_pdf(REPORT_PAGES)  # identical bytes both times (a PDF embeds a random id, so regenerating would differ)
    first = client.post("/api/documents", headers=auth, files={"file": ("report.pdf", data, "application/pdf")}).json()
    assert client.get(f"/api/documents/{first['id']}", headers=auth).json()["status"] == "failed"
    again = client.post("/api/documents", headers=auth, files={"file": ("report.pdf", data, "application/pdf")})
    assert again.status_code == 202 and again.json()["id"] == first["id"] and again.json().get("reprocessing")
    assert client.get(f"/api/documents/{first['id']}", headers=auth).json()["status"] == "ready"


def test_extraction_warnings_are_exposed_to_the_user(client, auth):
    import pymupdf

    doc = pymupdf.open()
    doc.new_page().insert_textbox(pymupdf.Rect(50, 50, 550, 200), "A readable page with enough words to be indexed by the extractor.", fontsize=11)
    doc.new_page()  # blank page
    r = client.post("/api/documents", headers=auth, files={"file": ("mixed.pdf", doc.tobytes(), "application/pdf")})
    info = client.get(f"/api/documents/{r.json()['id']}", headers=auth).json()
    assert info["status"] == "ready" and any("no text layer" in w for w in info["warnings"])


def test_ready_endpoint_describes_models_without_leaking_secrets(client, svc):
    body = client.get("/api/ready").json()
    text = json.dumps(body)
    assert "test-key" not in text and body["models"]["llm"]["model"] == "test-model"
    assert body["models"]["embedding"]["runs"] == "locally" and "external API" in body["models"]["llm"]["runs"]


def test_ollama_is_reported_as_local(client, svc):
    svc.settings.llm_provider, svc.settings.llm_model = "ollama", "llama3.1"
    assert client.get("/api/ready").json()["models"]["llm"]["runs"] == "locally"


def test_security_headers_present(client):
    h = client.get("/api/health").headers
    assert h["x-content-type-options"] == "nosniff" and h["cache-control"] == "no-store" and h["referrer-policy"] == "no-referrer"


def test_api_key_never_appears_in_error_responses(client, auth, svc):
    from app.llm import LLMError

    class Boom:
        def complete(self, *a, **k):
            raise LLMError("The LLM provider rejected the credentials (HTTP 401). Check LLM_API_KEY.")

    svc.llm_factory = lambda: Boom()
    doc = upload(client, auth)
    r = client.post(f"/api/documents/{doc['id']}/ask", headers=auth, json={"question": "When did the Lisbon office open?"})
    assert r.status_code == 502 and "test-key" not in r.text
