"""Embedding interface. The default implementation lazily loads sentence-transformers (runs locally)."""
from __future__ import annotations

import logging
import threading
from typing import Protocol

log = logging.getLogger(__name__)


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def embed_query(embedder: Embedder, text: str) -> list[float]:
    """Embed a search query (uses the query prefix of models such as E5/BGE when the embedder defines one)."""
    fn = getattr(embedder, "embed_query", None)
    return fn(text) if callable(fn) else embedder.embed([text])[0]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, query_prefix: str = "", document_prefix: str = ""):
        self.model_name = model_name
        self.query_prefix, self.document_prefix = query_prefix, document_prefix
        self._model = None
        self._lock = threading.Lock()
        self._dim: int | None = None

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                except ImportError as exc:
                    raise RuntimeError("sentence-transformers is not installed (see requirements-ml.txt).") from exc
                log.info("Loading embedding model %s (first run downloads it)", self.model_name)
                self._model = SentenceTransformer(self.model_name)
                self._dim = int(self._model.get_sentence_embedding_dimension())
        return self._model

    def warmup(self) -> None:
        self._load()

    @property
    def dim(self) -> int:
        self._load()
        assert self._dim is not None
        return self._dim

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._load()
        vecs = model.encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vecs]

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed document text or sentences (document prefix applied)."""
        return self._encode([self.document_prefix + t for t in texts] if self.document_prefix else texts)

    def embed_query(self, text: str) -> list[float]:
        return self._encode([self.query_prefix + text])[0]
