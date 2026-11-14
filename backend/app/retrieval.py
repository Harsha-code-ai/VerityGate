"""Hybrid retrieval: dense (Qdrant) + BM25, fused with Reciprocal Rank Fusion, optional cross-encoder reranking.

Every lookup is scoped to (owner, document): the vector filter, the BM25 index cache key and the SQL lookups all
include the owner id, so one user's chunks can never be returned for another user."""
from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Protocol

from rank_bm25 import BM25Plus

from .db import Database
from .embeddings import Embedder, embed_query
from .textutils import content_terms, term_document_frequency
from .vectorstore import QdrantStore

log = logging.getLogger(__name__)


# words that make a question a question but say nothing about the topic
_QUESTION_FILLER = frozenset({"long", "many", "much", "often", "far", "big", "old", "mention", "tell", "say", "state", "describe", "explain"})


def tokenize(text: str) -> list[str]:
    """BM25 tokens: stop words removed, stemmed, synonyms folded to one canonical term, numbers kept."""
    return content_terms(text, keep_numbers=True)


@dataclass
class Passage:
    chunk_id: str
    doc_id: str
    page: int
    section: str | None
    text: str
    score: float = 0.0  # final ranking score (RRF, or reranker if enabled)
    dense_score: float | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None
    kind: str = "text"


class Reranker(Protocol):
    def rerank(self, query: str, passages: list[Passage]) -> list[Passage]: ...


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._lock = threading.Lock()

    def rerank(self, query: str, passages: list[Passage]) -> list[Passage]:
        with self._lock:
            if self._model is None:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
        scores = self._model.predict([(query, p.text) for p in passages])
        for p, s in zip(passages, scores):
            p.rerank_score = float(s)
            p.score = float(s)
        return sorted(passages, key=lambda p: p.score, reverse=True)


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> dict[str, float]:
    """Reciprocal Rank Fusion over several ranked lists of ids."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return scores


@dataclass
class _DocIndex:
    chunk_ids: list[str]
    model: BM25Plus  # NOT BM25Okapi: its idf turns negative for terms in over half the chunks, i.e. most terms of a short document
    df: dict[str, float] = field(default_factory=dict)


class Retriever:
    def __init__(self, db: Database, store: QdrantStore, embedder: Embedder, *, candidates: int = 20, rrf_k: int = 60,
                 min_dense: float = 0.25, min_bm25: float = 1.0, min_query_coverage: float = 0.34, reranker: Reranker | None = None):
        self.db, self.store, self.embedder = db, store, embedder
        self.candidates, self.rrf_k = candidates, rrf_k
        self.min_dense, self.min_bm25, self.min_query_coverage = min_dense, min_bm25, min_query_coverage
        self.reranker = reranker
        self._index: dict[tuple[str, str], _DocIndex] = {}
        self._lock = threading.Lock()

    def invalidate(self, doc_id: str) -> None:
        with self._lock:
            for key in [k for k in self._index if k[1] == doc_id]:
                del self._index[key]

    def _index_for(self, owner_id: str, doc_id: str) -> _DocIndex | None:
        key = (owner_id, doc_id)
        with self._lock:
            if key in self._index:
                return self._index[key]
        rows = self.db.all("SELECT id, section, text FROM chunks WHERE doc_id=? AND owner_id=? ORDER BY idx", (doc_id, owner_id))
        if not rows:
            return None
        docs = [f"{r['section'] or ''} {r['text']}" for r in rows]
        index = _DocIndex([r["id"] for r in rows], BM25Plus([tokenize(d) or ["_"] for d in docs]), term_document_frequency(docs))
        with self._lock:
            self._index[key] = index
        return index

    def term_df(self, owner_id: str, doc_id: str) -> dict[str, float]:
        """Fraction of the document's chunks containing each canonical term (used to judge how distinctive a term is)."""
        index = self._index_for(owner_id, doc_id)
        return index.df if index else {}

    def retrieve(self, owner_id: str, doc_id: str, query: str, top_k: int, mode: str = "hybrid", page: int | None = None) -> list[Passage]:
        """mode: 'hybrid' (default), 'dense' or 'bm25' - the single-signal modes exist for evaluation."""
        depth = max(self.candidates, top_k * 3)
        dense_hits = []
        if mode in ("hybrid", "dense"):
            dense_hits = self.store.search(owner_id, doc_id, embed_query(self.embedder, query), depth, page=page)
        dense_rank = {h.chunk_id: i for i, h in enumerate(dense_hits, start=1)}
        dense_score = {h.chunk_id: h.score for h in dense_hits}
        bm25_rank: dict[str, int] = {}
        bm25_score: dict[str, float] = {}
        index = self._index_for(owner_id, doc_id) if mode in ("hybrid", "bm25") else None
        q_tokens = tokenize(query)
        if index and q_tokens:
            scores = index.model.get_scores(q_tokens)
            order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:depth]
            for rank, i in enumerate((i for i in order if scores[i] > 0), start=1):
                bm25_rank[index.chunk_ids[i]] = rank
                bm25_score[index.chunk_ids[i]] = float(scores[i])
        fused = rrf_fuse([sorted(dense_rank, key=dense_rank.__getitem__), sorted(bm25_rank, key=bm25_rank.__getitem__)], self.rrf_k)
        if not fused:
            return []
        keep = max(top_k, self.candidates) if self.reranker else top_k
        ids = sorted(fused, key=fused.__getitem__, reverse=True)[: keep * 2 if page is not None else keep]
        placeholders = ",".join("?" * len(ids))
        rows = {
            r["id"]: r
            for r in self.db.all(
                f"SELECT id, doc_id, page, section, text, kind FROM chunks WHERE id IN ({placeholders}) AND owner_id=? AND doc_id=?",
                (*ids, owner_id, doc_id),
            )
        }
        passages = [
            Passage(chunk_id=i, doc_id=doc_id, page=rows[i]["page"], section=rows[i]["section"], text=rows[i]["text"],
                    score=fused[i], dense_score=dense_score.get(i), bm25_score=bm25_score.get(i),
                    dense_rank=dense_rank.get(i), bm25_rank=bm25_rank.get(i), kind=rows[i]["kind"] or "text")
            for i in ids
            if i in rows and (page is None or rows[i]["page"] == page)
        ]
        if self.reranker and passages:
            passages = self.reranker.rerank(query, passages)
        return passages[:top_k]

    def evidence_is_sufficient(self, passages: list[Passage], query: str | None = None) -> bool:
        """Conservative gate: abstain early when nothing looks related to the question.

        Three independent signals, any of which lets the pipeline continue: a dense similarity above `min_dense`, a BM25 score
        above `min_bm25`, or - independent of corpus size - at least `min_query_coverage` of the question's content terms
        appearing in the retrieved text. (Absolute BM25 scores are tiny in short documents, so BM25 alone must not decide.)
        Thresholds are configuration values; calibrate them with scripts/calibrate.py on your own data."""
        if any((p.dense_score is not None and p.dense_score >= self.min_dense) or (p.bm25_score is not None and p.bm25_score >= self.min_bm25)
               for p in passages):
            return True
        if query:
            q_terms = {t for t in content_terms(query) if not t.isdigit() and t not in _QUESTION_FILLER}
            if q_terms:
                seen: set[str] = set()
                for p in passages:
                    seen |= set(content_terms(f"{p.section or ''} {p.text}"))
                return len(q_terms & seen) / len(q_terms) >= self.min_query_coverage
        return False
