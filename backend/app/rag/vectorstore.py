"""Vector store backends: Pinecone (serverless) and a local JSON-persisted store.

Both implement the same minimal interface (upsert / query / count /
clear), so the retriever and ingestion pipeline are backend-agnostic.
"""

import json
import logging
from pathlib import Path
from typing import Protocol

import numpy as np

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class VectorMatch:
    __slots__ = ("id", "score", "metadata")

    def __init__(self, id: str, score: float, metadata: dict) -> None:
        self.id = id
        self.score = score
        self.metadata = metadata

    def __repr__(self) -> str:  # pragma: no cover
        return f"VectorMatch(id={self.id!r}, score={self.score:.3f})"


class VectorStore(Protocol):
    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None: ...
    def query(self, vector: list[float], top_k: int) -> list[VectorMatch]: ...
    def count(self) -> int: ...
    def clear(self) -> None: ...


class LocalVectorStore:
    """Cosine-similarity store persisted to a JSON file.

    Suitable for this project's small corpus (~50 chunks); swap in
    Pinecone via config for production-scale corpora.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._ids: list[str] = []
        self._vectors: np.ndarray | None = None
        self._metadatas: list[dict] = []
        self._load()

    def _load(self) -> None:
        if self.path.exists():
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self._ids = data["ids"]
            self._metadatas = data["metadatas"]
            vectors = data["vectors"]
            self._vectors = np.array(vectors, dtype=np.float32) if vectors else None

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "ids": self._ids,
            "vectors": self._vectors.tolist() if self._vectors is not None else [],
            "metadatas": self._metadatas,
        }
        self.path.write_text(json.dumps(payload), encoding="utf-8")

    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        index_of = {id_: i for i, id_ in enumerate(self._ids)}
        new_vectors = self._vectors.tolist() if self._vectors is not None else []
        for id_, vec, meta in zip(ids, vectors, metadatas):
            if id_ in index_of:  # duplicate-safe: overwrite in place
                i = index_of[id_]
                new_vectors[i] = vec
                self._metadatas[i] = meta
            else:
                index_of[id_] = len(self._ids)
                self._ids.append(id_)
                new_vectors.append(vec)
                self._metadatas.append(meta)
        self._vectors = np.array(new_vectors, dtype=np.float32)
        self._save()

    def query(self, vector: list[float], top_k: int) -> list[VectorMatch]:
        if self._vectors is None or len(self._ids) == 0:
            return []
        q = np.array(vector, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        norms = np.linalg.norm(self._vectors, axis=1)
        norms[norms == 0] = 1e-9
        scores = (self._vectors @ q) / (norms * q_norm)
        order = np.argsort(-scores)[:top_k]
        return [
            VectorMatch(self._ids[i], float(scores[i]), self._metadatas[i])
            for i in order
        ]

    def count(self) -> int:
        return len(self._ids)

    def clear(self) -> None:
        self._ids, self._vectors, self._metadatas = [], None, []
        if self.path.exists():
            self.path.unlink()


class PineconeVectorStore:
    """Pinecone serverless index wrapper."""

    def __init__(self, settings: Settings, dimension: int) -> None:
        from pinecone import Pinecone, ServerlessSpec

        self._pc = Pinecone(api_key=settings.pinecone_api_key)
        name = settings.pinecone_index
        if not self._pc.has_index(name):
            logger.info("Creating Pinecone serverless index %r (dim=%d)", name, dimension)
            self._pc.create_index(
                name=name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
            )
        self._index = self._pc.Index(name)

    def upsert(self, ids: list[str], vectors: list[list[float]], metadatas: list[dict]) -> None:
        batch = [
            {"id": id_, "values": vec, "metadata": meta}
            for id_, vec, meta in zip(ids, vectors, metadatas)
        ]
        for start in range(0, len(batch), 100):
            self._index.upsert(vectors=batch[start : start + 100])

    def query(self, vector: list[float], top_k: int) -> list[VectorMatch]:
        response = self._index.query(vector=vector, top_k=top_k, include_metadata=True)
        return [
            VectorMatch(m["id"], float(m["score"]), dict(m.get("metadata") or {}))
            for m in response.get("matches", [])
        ]

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        return int(stats.get("total_vector_count", 0))

    def clear(self) -> None:
        self._index.delete(delete_all=True)


def get_vector_store(settings: Settings | None = None, *, dimension: int) -> VectorStore:
    settings = settings or get_settings()
    if settings.resolved_vector_backend == "pinecone":
        return PineconeVectorStore(settings, dimension)
    return LocalVectorStore(settings.local_vector_store_path)
