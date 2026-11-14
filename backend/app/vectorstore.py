"""Qdrant wrapper. Every search is filtered by owner AND document."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from qdrant_client import QdrantClient, models

from .config import Settings

log = logging.getLogger(__name__)


@dataclass
class VectorHit:
    chunk_id: str
    score: float


def _point_id(chunk_id: str) -> str:
    return str(uuid.UUID(chunk_id))  # chunk ids are uuid4 hex strings


def make_client(settings: Settings) -> QdrantClient:
    url = settings.qdrant_url
    if url == ":memory:":
        return QdrantClient(location=":memory:")
    if not url:
        path = settings.data_dir / "qdrant_local"
        path.mkdir(parents=True, exist_ok=True)
        return QdrantClient(path=str(path))
    return QdrantClient(url=url, api_key=settings.qdrant_api_key or None, timeout=30)


class QdrantStore:
    def __init__(self, client: QdrantClient, collection: str):
        self.client = client
        self.collection = collection
        self._ready = False

    def ping(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            log.warning("Qdrant unreachable: %s", type(exc).__name__)
            return False

    def ensure_collection(self, dim: int) -> None:
        if self._ready:
            return
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            self.client.create_collection(
                self.collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE)
            )
            for field in ("doc_id", "owner_id"):
                try:
                    if self.client._client.__class__.__name__ == "QdrantLocal":  # indexes are server-only
                        break
                    self.client.create_payload_index(self.collection, field, models.PayloadSchemaType.KEYWORD)
                except Exception:  # not supported by every mode; filtering still works
                    pass
        else:
            info = self.client.get_collection(self.collection)
            size = info.config.params.vectors.size  # type: ignore[union-attr]
            if size != dim:
                raise RuntimeError(
                    f"Qdrant collection '{self.collection}' has dimension {size} but the embedding model "
                    f"produces {dim}. Set QDRANT_COLLECTION to a new name (your old documents must then be re-uploaded)."
                )
        self._ready = True

    def upsert(self, owner_id: str, doc_id: str, chunks: list[tuple[str, int, str | None]], vectors: list[list[float]]) -> None:
        """chunks: (chunk_id, page, section)"""
        if not chunks:
            return
        self.ensure_collection(len(vectors[0]))
        points = [
            models.PointStruct(
                id=_point_id(cid),
                vector=vec,
                payload={"chunk_id": cid, "doc_id": doc_id, "owner_id": owner_id, "page": page, "section": section},
            )
            for (cid, page, section), vec in zip(chunks, vectors)
        ]
        for i in range(0, len(points), 128):
            self.client.upsert(self.collection, points=points[i : i + 128], wait=True)

    def search(self, owner_id: str, doc_id: str, vector: list[float], limit: int, page: int | None = None) -> list[VectorHit]:
        """Nearest chunks, ALWAYS restricted to this owner and this document (optionally one page)."""
        self.ensure_collection(len(vector))
        must = [
            models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        ]
        if page is not None:
            must.append(models.FieldCondition(key="page", match=models.MatchValue(value=page)))
        flt = models.Filter(must=must)
        res = self.client.query_points(self.collection, query=vector, query_filter=flt, limit=limit, with_payload=True)
        return [VectorHit(p.payload["chunk_id"], float(p.score)) for p in res.points]

    def delete_document(self, owner_id: str, doc_id: str) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            return
        self.client.delete(
            self.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="owner_id", match=models.MatchValue(value=owner_id)),
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
                    ]
                )
            ),
            wait=True,
        )

    def count(self, doc_id: str) -> int:
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection not in existing:
            return 0
        flt = models.Filter(must=[models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))])
        return self.client.count(self.collection, count_filter=flt, exact=True).count
