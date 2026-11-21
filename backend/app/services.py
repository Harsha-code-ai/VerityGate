"""Service container + document ingestion pipeline. Heavy models are created lazily (or pre-loaded at startup)."""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from .chunking import chunk_pages
from .config import Settings
from .db import Database, new_id
from .embeddings import Embedder, SentenceTransformerEmbedder
from .llm import LLMClient, get_llm
from .pdf import ExtractionError, extract_document
from .retrieval import CrossEncoderReranker, Retriever
from .security import RateLimiter
from .vectorstore import QdrantStore, make_client
from .verifier import ClaimVerifier, NLIModel, TransformersNLI, VerifierConfig
from .workflow import WorkflowConfig, build_graph

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    db: Database
    store: QdrantStore
    embedder: Embedder
    retriever: Retriever
    nli: NLIModel
    secret_key: str
    limiter: RateLimiter
    llm_factory: Callable[[], LLMClient] = field(default=None)  # type: ignore[assignment]
    ingest_slots: threading.Semaphore = field(default_factory=lambda: threading.Semaphore(1))
    model_status: dict[str, str] = field(default_factory=lambda: {"embedding": "not loaded", "nli": "not loaded"})

    def verifier_config(self) -> VerifierConfig:
        s = self.settings
        return VerifierConfig(entail_thr=s.entailment_threshold, contra_thr=s.contradiction_threshold, conflict_thr=s.conflict_threshold,
                              max_units=s.verify_max_units, min_coverage=s.relevance_min_coverage,
                              min_support_coverage=s.relevance_min_support_coverage, min_similarity=s.relevance_min_similarity)

    def build_graph(self):
        llm = self.llm_factory()
        verifier = ClaimVerifier(self.nli, embedder=self.embedder, config=self.verifier_config())
        cfg = WorkflowConfig(self.settings.retrieval_top_k, self.settings.effective_max_retries, self.settings.llm_max_tokens)
        return build_graph(self.retriever, llm, verifier, cfg), cfg

    def warm_models(self) -> None:
        """Load the local models now so the first upload/question is fast; records what worked in `model_status`."""
        for name, obj, fn in (("embedding", self.embedder, "warmup"), ("nli", self.nli, "_load")):
            loader = getattr(obj, fn, None)
            if loader is None:
                self.model_status[name] = "ready (custom)"
                continue
            try:
                loader()
                self.model_status[name] = "ready"
            except Exception as exc:
                self.model_status[name] = f"error: {str(exc)[:200]}"
                log.error("Could not load %s model: %s", name, exc)


def build_services(settings: Settings, *, embedder: Embedder | None = None, nli: NLIModel | None = None,
                   llm_factory: Callable[[], LLMClient] | None = None) -> Services:
    settings.ensure_dirs()
    db = Database(settings.db_path)
    interrupted = db.mark_interrupted_documents()
    if interrupted:
        log.warning("Marked %d interrupted document(s) as failed", interrupted)
    store = QdrantStore(make_client(settings), settings.qdrant_collection)
    embedder = embedder or SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_query_prefix, settings.embedding_document_prefix)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.reranker_enabled else None
    retriever = Retriever(db, store, embedder, candidates=settings.retrieval_candidates, rrf_k=settings.rrf_k,
                          min_dense=settings.min_dense_score, min_bm25=settings.min_bm25_score,
                          min_query_coverage=settings.min_query_coverage, reranker=reranker)
    return Services(
        settings=settings, db=db, store=store, embedder=embedder, retriever=retriever,
        nli=nli or TransformersNLI(settings.nli_model, settings.nli_label_order), secret_key=settings.resolve_secret_key(),
        limiter=RateLimiter(settings.rate_limit_per_minute), llm_factory=llm_factory or (lambda: get_llm(settings)),
        ingest_slots=threading.Semaphore(max(1, settings.ingest_concurrency)),
    )


def embedding_text(section: str | None, text: str) -> str:
    """Dense retrieval sees the section heading too; the stored/verified chunk text stays exactly what the PDF says."""
    return f"{section}\n{text}" if section else text


def ingest_document(svc: Services, doc_id: str) -> None:
    """Extract -> chunk -> embed -> index. Updates the document row with status/error."""
    with svc.ingest_slots:
        _ingest(svc, doc_id)


def _ingest(svc: Services, doc_id: str) -> None:
    s = svc.settings
    doc = svc.db.one("SELECT * FROM documents WHERE id=?", (doc_id,))
    if not doc:
        return
    owner = doc["owner_id"]
    t0 = time.perf_counter()

    def fail(message: str) -> None:
        svc.db.execute("UPDATE documents SET status='failed', error=?, updated_at=? WHERE id=?", (message, time.time(), doc_id))
        svc.db.audit("document.failed", owner, doc_id, reason=message[:200])

    try:
        svc.db.execute("UPDATE documents SET status='processing', updated_at=? WHERE id=?", (time.time(), doc_id))
        data = (s.uploads_dir / doc["stored_name"]).read_bytes()
        report = extract_document(data, s.min_extracted_chars, s.ocr_enabled, ocr_language=s.ocr_language, ocr_dpi=s.ocr_dpi, ocr_engine=s.ocr_engine)
        chunks = chunk_pages(report.pages, s.chunk_words, s.chunk_overlap_words)
        if not chunks:
            return fail("The PDF produced no text chunks.")
        ids = [new_id() for _ in chunks]
        vectors = svc.embedder.embed([embedding_text(c.section, c.text) for c in chunks])
        if not svc.db.one("SELECT 1 FROM documents WHERE id=?", (doc_id,)):  # deleted while we were working
            return
        svc.store.upsert(owner, doc_id, [(i, c.page, c.section) for i, c in zip(ids, chunks)], vectors)
        svc.db.replace_chunks(doc_id, owner, [(i, c.page, c.idx, c.section, c.text, c.kind) for i, c in zip(ids, chunks)])
        svc.retriever.invalidate(doc_id)
        if not svc.db.one("SELECT 1 FROM documents WHERE id=?", (doc_id,)):  # deleted during indexing: remove what we just wrote
            svc.store.delete_document(owner, doc_id)
            svc.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            return
        svc.db.execute(
            "UPDATE documents SET status='ready', error=NULL, pages=?, chunk_count=?, extraction_method=?, warnings=?, ocr_pages=?, updated_at=? WHERE id=?",
            (len(report.pages), len(chunks), report.method, json.dumps(report.warnings), len(report.ocr_pages), time.time(), doc_id),
        )
        svc.db.audit("document.ready", owner, doc_id, pages=len(report.pages), chunks=len(chunks), method=report.method,
                     seconds=round(time.perf_counter() - t0, 2))
    except ExtractionError as exc:
        fail(str(exc))
    except Exception as exc:  # report a clean message; details go to the log without document text
        log.exception("Ingestion failed for %s", doc_id)
        try:
            svc.store.delete_document(owner, doc_id)
        except Exception:
            pass
        svc.db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
        detail = f" {str(exc)[:200]}" if str(exc) else ""
        fail(f"Indexing failed ({type(exc).__name__}).{detail} Check that Qdrant is running and the embedding model could be loaded; see the backend logs.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()
