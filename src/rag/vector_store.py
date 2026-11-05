"""Vector storage behind a small interface, with a Qdrant implementation.

Local persistent mode (a folder on disk) is the default; pass a URL to use a remote
Qdrant server later without changing any calling code.

Safety rules implemented here:
* Point IDs are derived from chunk IDs (UUIDv5), so re-upserting never duplicates.
* An existing collection is inspected, never silently reconfigured; a dimension or
  distance mismatch raises CollectionConfigError.
* Only the configured collection is ever touched.
"""
from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from typing import Any, Sequence

from src.rag.schema import Chunk

_NAMESPACE = uuid.UUID("6f1c2d3e-4a5b-5c6d-8e7f-0a1b2c3d4e5f")


class CollectionConfigError(RuntimeError):
    """Existing collection does not match the requested configuration."""


class CollectionMissingError(RuntimeError):
    """The configured collection does not exist (nothing has been ingested yet)."""


def point_id(chunk_id: str) -> str:
    """Stable Qdrant point ID for a chunk ID."""
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


class VectorStore(ABC):
    """What the rest of the RAG module needs from a vector database."""

    @abstractmethod
    def ensure_collection(self, dim: int) -> None: ...
    @abstractmethod
    def collection_exists(self) -> bool: ...
    @abstractmethod
    def upsert_chunks(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]], embedding_fingerprint: str) -> None: ...
    @abstractmethod
    def delete_stale_chunks(self, document_id: str, keep_document_hash: str) -> None: ...
    @abstractmethod
    def delete_document(self, document_id: str) -> None: ...
    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int, filters: dict[str, Any] | None = None) -> list[tuple[dict, float]]: ...
    @abstractmethod
    def count(self) -> int: ...
    @abstractmethod
    def chunk_counts_by_document(self) -> dict[str, int]: ...
    @abstractmethod
    def get_document_chunks(self, document_id: str) -> list[dict]: ...
    @abstractmethod
    def iter_payloads(self): ...
    @abstractmethod
    def get_vectors(self, chunk_ids: Sequence[str]) -> dict[str, list[float]]: ...
    @abstractmethod
    def drop_collection(self) -> None: ...
    def close(self) -> None:  # optional
        return None


class QdrantVectorStore(VectorStore):
    def __init__(self, collection: str, path: str | None = None, url: str | None = None,
                 api_key: str | None = None, client: Any = None):
        from qdrant_client import QdrantClient

        self.collection = collection
        if client is not None:
            self.client = client
        elif url:
            self.client = QdrantClient(url=url, api_key=api_key)
        else:
            if not path:
                raise ValueError("either path (local) or url (remote) is required")
            self.client = QdrantClient(path=str(path))

    # -- collection management ------------------------------------------------------
    def collection_exists(self) -> bool:
        return bool(self.client.collection_exists(self.collection))

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        if not self.collection_exists():
            self.client.create_collection(self.collection, vectors_config=VectorParams(size=dim, distance=Distance.COSINE))
            return
        params = self.client.get_collection(self.collection).config.params.vectors
        size = getattr(params, "size", None)
        distance = getattr(params, "distance", None)
        if size != dim or distance != Distance.COSINE:
            raise CollectionConfigError(
                f"collection '{self.collection}' has size={size}, distance={distance}; expected size={dim}, distance=Cosine. "
                "Use a different collection name, or rebuild explicitly with --rebuild-collection.")

    def drop_collection(self) -> None:
        if self.collection_exists():
            self.client.delete_collection(self.collection)

    # -- writes ---------------------------------------------------------------------
    def upsert_chunks(self, chunks, vectors, embedding_fingerprint: str) -> None:
        from qdrant_client.models import PointStruct

        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors differ in length")
        points = [PointStruct(id=point_id(c.chunk_id), vector=list(v),
                              payload={**c.to_dict(), "embedding_fingerprint": embedding_fingerprint})
                  for c, v in zip(chunks, vectors)]
        for i in range(0, len(points), 256):
            self.client.upsert(self.collection, points=points[i:i + 256], wait=True)

    def delete_stale_chunks(self, document_id: str, keep_document_hash: str) -> None:
        """Remove this document's chunks that belong to any *other* version of the document."""
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))],
                     must_not=[FieldCondition(key="document_hash", match=MatchValue(value=keep_document_hash))])
        self.client.delete(self.collection, points_selector=FilterSelector(filter=flt), wait=True)

    def delete_document(self, document_id: str) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        self.client.delete(self.collection, points_selector=FilterSelector(filter=flt), wait=True)

    # -- reads ----------------------------------------------------------------------
    @staticmethod
    def _build_filter(filters: dict[str, Any] | None):
        from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

        if not filters:
            return None
        conditions = []
        for key, value in filters.items():
            match = MatchAny(any=list(value)) if isinstance(value, (list, tuple, set)) else MatchValue(value=value)
            conditions.append(FieldCondition(key=key, match=match))
        return Filter(must=conditions)

    def search(self, vector, top_k, filters=None):
        if not self.collection_exists():
            raise CollectionMissingError(f"collection '{self.collection}' does not exist; run ingestion first")
        res = self.client.query_points(self.collection, query=list(vector), limit=top_k,
                                       query_filter=self._build_filter(filters), with_payload=True)
        return [(dict(p.payload or {}), float(p.score)) for p in res.points]

    def count(self) -> int:
        if not self.collection_exists():
            return 0
        return int(self.client.count(self.collection, exact=True).count)

    def _scroll(self, flt=None, payload=True):
        offset = None
        while True:
            points, offset = self.client.scroll(self.collection, scroll_filter=flt, limit=512,
                                                with_payload=payload, with_vectors=False, offset=offset)
            yield from points
            if offset is None:
                break

    def chunk_counts_by_document(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        if self.collection_exists():
            for p in self._scroll(payload=["document_id"]):
                d = (p.payload or {}).get("document_id")
                counts[d] = counts.get(d, 0) + 1
        return counts

    def get_document_chunks(self, document_id: str) -> list[dict]:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        if not self.collection_exists():
            return []
        flt = Filter(must=[FieldCondition(key="document_id", match=MatchValue(value=document_id))])
        return sorted((dict(p.payload or {}) for p in self._scroll(flt)), key=lambda x: x.get("chunk_index", 0))

    def iter_payloads(self):
        """Yield the payload (chunk text + metadata) of every stored chunk."""
        if self.collection_exists():
            for p in self._scroll():
                yield dict(p.payload or {})

    def get_vectors(self, chunk_ids):
        """Return {chunk_id: vector} for the given chunk IDs (missing ones are omitted)."""
        if not chunk_ids or not self.collection_exists():
            return {}
        pts = self.client.retrieve(self.collection, ids=[point_id(c) for c in chunk_ids], with_vectors=True, with_payload=["chunk_id"])
        return {(p.payload or {}).get("chunk_id"): list(p.vector) for p in pts if p.vector is not None}

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:  # pragma: no cover
            pass
