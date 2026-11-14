"""FastAPI application and routes."""
from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import Settings, get_settings
from .db import new_id
from .llm import LLMError, validate_llm_config
from .logging_setup import setup_logging
from .pdf import PDFValidationError, validate_pdf
from .security import create_token, hash_password, safe_display_name, verify_password_or_dummy, verify_token_payload
from .services import Services, build_services, ingest_document, sha256
from .workflow import run_question

log = logging.getLogger("verigate")


class Credentials(BaseModel):
    email: str = Field(min_length=3, max_length=254, pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=10, max_length=200)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


def create_app(services: Services | None = None, settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.svc = services or build_services(settings)
        if settings.preload_models and services is None:
            threading.Thread(target=app.state.svc.warm_models, name="model-warmup", daemon=True).start()
        log.info("VerityGate %s started (env=%s)", __version__, settings.app_env)
        yield
        app.state.svc.db.close()

    app = FastAPI(title="VerityGate", version=__version__, lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=settings.cors_origin_list, allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Authorization", "Content-Type"], allow_credentials=False,
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        rid, t0 = uuid.uuid4().hex[:12], time.perf_counter()
        declared = request.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > (settings.max_upload_mb + 1) * 1024 * 1024:  # refuse before reading the body
            return JSONResponse({"detail": f"The request is larger than the {settings.max_upload_mb} MB limit."}, status_code=413)
        try:
            response = await call_next(request)
        except Exception:
            log.exception("Unhandled error", extra={"ctx": {"request_id": rid, "path": request.url.path}})
            return JSONResponse({"detail": "Internal server error."}, status_code=500)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store"
        response.headers["Referrer-Policy"] = "no-referrer"
        log.info("request", extra={"ctx": {"request_id": rid, "method": request.method, "path": request.url.path,
                                           "status": response.status_code, "ms": round((time.perf_counter() - t0) * 1000)}})
        return response

    def svc_dep(request: Request) -> Services:
        return request.app.state.svc

    def current_user(authorization: str | None = Header(default=None), svc: Services = Depends(svc_dep)) -> dict:
        if not authorization or not authorization.lower().startswith("bearer "):
            raise HTTPException(401, "Sign in required.")
        payload = verify_token_payload(authorization[7:].strip(), svc.secret_key)
        user = svc.db.one("SELECT id, email, token_epoch FROM users WHERE id=?", (payload["sub"],)) if payload else None
        if not user or int(user["token_epoch"]) != payload["ep"]:  # epoch changes on sign-out: old tokens stop working
            raise HTTPException(401, "Your session is invalid or expired. Sign in again.")
        return user

    def owned_doc(doc_id: str, user: dict, svc: Services) -> dict:
        doc = svc.db.one("SELECT * FROM documents WHERE id=? AND owner_id=?", (doc_id, user["id"]))
        if not doc:  # same answer for "missing" and "not yours" so ids cannot be probed
            raise HTTPException(404, "Document not found.")
        return doc

    def throttle(request: Request, user: dict, svc: Services) -> None:
        if not svc.limiter.allow(f"{user['id']}"):
            raise HTTPException(429, "Too many requests. Please wait a minute and try again.")

    def public_doc(d: dict) -> dict:
        out = {k: d[k] for k in ("id", "display_name", "size_bytes", "pages", "chunk_count", "status", "error", "extraction_method", "created_at", "ocr_pages")}
        try:
            out["warnings"] = json.loads(d["warnings"]) if d.get("warnings") else []
        except ValueError:
            out["warnings"] = []
        return out

    # ---------- health ----------
    @app.get("/api/health")
    def health():
        return {"status": "ok", "version": __version__}

    @app.get("/api/ready")
    def ready(svc: Services = Depends(svc_dep)):
        checks = {"database": svc.db.ping(), "qdrant": svc.store.ping()}
        problems = validate_llm_config(svc.settings)
        s = svc.settings
        local_llm = s.llm_provider == "ollama" or (s.llm_provider == "openai" and any(h in (s.llm_base_url or "") for h in ("localhost", "127.0.0.1", "host.docker.internal")))
        body = {"ready": all(checks.values()), "checks": checks, "llm_configured": not problems, "llm_problems": problems,
                "llm_provider": s.llm_provider or None, "registration_open": s.allow_registration,
                "models": {
                    "llm": {"provider": s.llm_provider or None, "model": s.llm_model or None, "runs": "locally" if local_llm else "external API (document excerpts are sent to the provider)"},
                    "embedding": {"model": s.embedding_model, "runs": "locally", "status": svc.model_status.get("embedding")},
                    "nli": {"model": s.nli_model, "runs": "locally", "status": svc.model_status.get("nli")},
                    "reranker": {"model": s.reranker_model, "runs": "locally"} if s.reranker_enabled else None,
                },
                "ocr": {"enabled": s.ocr_enabled, "engine": s.ocr_engine if s.ocr_enabled else None}}
        return JSONResponse(body, status_code=200 if body["ready"] else 503)

    # ---------- auth ----------
    @app.post("/api/auth/register", status_code=201)
    def register(c: Credentials, svc: Services = Depends(svc_dep)):
        if not svc.settings.allow_registration:
            raise HTTPException(403, "Registration is disabled on this server.")
        email = c.email.lower()
        if svc.db.one("SELECT id FROM users WHERE email=?", (email,)):
            raise HTTPException(409, "An account with this email already exists.")
        uid = new_id()
        svc.db.execute("INSERT INTO users(id,email,password_hash,created_at) VALUES (?,?,?,?)", (uid, email, hash_password(c.password), time.time()))
        svc.db.audit("user.registered", uid)
        return {"token": create_token(uid, svc.secret_key, svc.settings.token_ttl_minutes), "email": email}

    @app.post("/api/auth/login")
    def login(c: Credentials, request: Request, svc: Services = Depends(svc_dep)):
        client = request.client.host if request.client else "unknown"
        if not svc.limiter.allow(f"login:{client}") or not svc.limiter.allow(f"login:{client}:{c.email.lower()}"):
            raise HTTPException(429, "Too many sign-in attempts. Please wait a minute.")
        user = svc.db.one("SELECT * FROM users WHERE email=?", (c.email.lower(),))
        if not verify_password_or_dummy(c.password, user["password_hash"] if user else None):
            svc.db.audit("user.login_failed", user["id"] if user else None)
            raise HTTPException(401, "Email or password is incorrect.")
        svc.db.audit("user.login", user["id"])
        return {"token": create_token(user["id"], svc.secret_key, svc.settings.token_ttl_minutes, int(user["token_epoch"])), "email": user["email"]}

    @app.post("/api/auth/logout", status_code=204)
    def logout(user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        """Sign out everywhere: bumping the epoch invalidates every token issued so far."""
        svc.db.execute("UPDATE users SET token_epoch = token_epoch + 1 WHERE id=?", (user["id"],))
        svc.db.audit("user.logout", user["id"])

    @app.get("/api/auth/me")
    def me(user: dict = Depends(current_user)):
        return {"id": user["id"], "email": user["email"]}

    # ---------- documents ----------
    @app.post("/api/documents", status_code=202)
    def upload(request: Request, background: BackgroundTasks, file: UploadFile = File(...),
               user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        throttle(request, user, svc)
        s = svc.settings
        max_bytes = s.max_upload_mb * 1024 * 1024
        data = file.file.read(max_bytes + 1)  # never read more than the limit + 1 byte
        try:
            validate_pdf(data, max_bytes, s.max_pages)
        except PDFValidationError as exc:
            svc.db.audit("document.rejected", user["id"], reason=str(exc)[:120])
            raise HTTPException(422 if "limit" not in str(exc) else 413, str(exc)) from exc
        digest = sha256(data)
        existing = svc.db.one("SELECT * FROM documents WHERE owner_id=? AND sha256=?", (user["id"], digest))
        if existing and existing["status"] == "failed" and (s.uploads_dir / Path(existing["stored_name"]).name).exists():
            svc.db.execute("UPDATE documents SET status='pending', error=NULL, updated_at=? WHERE id=?", (time.time(), existing["id"]))
            background.add_task(ingest_document, svc, existing["id"])  # same file uploaded again after a failure: try again
            return JSONResponse({**public_doc(svc.db.one("SELECT * FROM documents WHERE id=?", (existing["id"],))), "duplicate": False, "reprocessing": True}, status_code=202)
        if existing:
            return JSONResponse({**public_doc(existing), "duplicate": True}, status_code=200)
        count = svc.db.one("SELECT COUNT(*) AS n FROM documents WHERE owner_id=?", (user["id"],))["n"]
        if count >= s.max_documents_per_user:
            raise HTTPException(409, f"You have reached the limit of {s.max_documents_per_user} documents. Delete one first.")
        doc_id, stored = new_id(), f"{new_id()}.pdf"
        (s.uploads_dir / stored).write_bytes(data)
        now = time.time()
        svc.db.execute(
            "INSERT INTO documents(id,owner_id,display_name,stored_name,sha256,size_bytes,status,created_at,updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (doc_id, user["id"], safe_display_name(file.filename), stored, digest, len(data), "pending", now, now),
        )
        svc.db.audit("document.uploaded", user["id"], doc_id, size=len(data))
        background.add_task(ingest_document, svc, doc_id)
        return {**public_doc(svc.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))), "duplicate": False}

    @app.get("/api/documents")
    def list_documents(user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        return [public_doc(d) for d in svc.db.all("SELECT * FROM documents WHERE owner_id=? ORDER BY created_at DESC", (user["id"],))]

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: str, user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        return public_doc(owned_doc(doc_id, user, svc))

    @app.delete("/api/documents/{doc_id}", status_code=204)
    def delete_document(doc_id: str, user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        doc = owned_doc(doc_id, user, svc)
        try:
            svc.store.delete_document(user["id"], doc_id)
        except Exception as exc:
            log.error("Vector deletion failed: %s", type(exc).__name__)
            raise HTTPException(502, "Could not delete the document's search index. Nothing was removed; try again once Qdrant is available.") from exc
        (svc.settings.uploads_dir / Path(doc["stored_name"]).name).unlink(missing_ok=True)
        svc.db.delete_document_rows(doc_id, user["id"])
        svc.retriever.invalidate(doc_id)
        svc.db.audit("document.deleted", user["id"], doc_id)

    @app.get("/api/documents/{doc_id}/pages/{page}")
    def page_text(doc_id: str, page: int, user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        """The indexed text of one page, so a citation can be checked against what VerityGate actually read."""
        owned_doc(doc_id, user, svc)
        rows = svc.db.all("SELECT section, text, kind FROM chunks WHERE doc_id=? AND owner_id=? AND page=? ORDER BY idx", (doc_id, user["id"], page))
        return {"page": page, "chunks": rows}

    @app.post("/api/documents/{doc_id}/ask")
    def ask(doc_id: str, body: AskRequest, request: Request, user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        doc = owned_doc(doc_id, user, svc)
        throttle(request, user, svc)
        if doc["status"] != "ready":
            raise HTTPException(409, f"The document is not ready (status: {doc['status']}).")
        problems = validate_llm_config(svc.settings)
        if problems:
            raise HTTPException(503, {"code": "llm_not_configured", "message": " ".join(problems)})
        try:
            graph, cfg = svc.build_graph()
            result = run_question(graph, user["id"], doc_id, body.question.strip(), cfg)
        except LLMError as exc:
            svc.db.audit("ask.llm_error", user["id"], doc_id)
            raise HTTPException(502, {"code": "llm_error", "message": str(exc)}) from exc
        except RuntimeError as exc:  # e.g. model libraries missing
            log.error("Ask failed: %s", exc)
            raise HTTPException(503, {"code": "dependency_error", "message": str(exc)}) from exc
        detail = {"question_sha256": sha256(body.question.encode())[:16], "question_chars": len(body.question),
                  "status": result["status"], "claims": len(result["claims"]), "retries": result["retries_used"],
                  "llm_calls": result["llm_calls"], "latency_s": result["latency_seconds"], "abstain_reason": result.get("abstain_reason")}
        if svc.settings.audit_store_questions:
            detail["question"] = body.question[:300]
        svc.db.audit("ask", user["id"], doc_id, **detail)
        return result

    @app.get("/api/audit")
    def audit(limit: int = 50, user: dict = Depends(current_user), svc: Services = Depends(svc_dep)):
        rows = svc.db.all("SELECT ts, event, doc_id, detail FROM audit WHERE user_id=? ORDER BY id DESC LIMIT ?", (user["id"], max(1, min(limit, 200))))
        return [{**r, "detail": json.loads(r["detail"] or "{}")} for r in rows]

    return app


def app_factory() -> FastAPI:  # uvicorn --factory app.main:app_factory
    return create_app()
