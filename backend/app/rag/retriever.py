"""Query-time retrieval: embed the query, search the vector store, return passages."""

import logging
from dataclasses import dataclass

from app.config import Settings, get_settings
from app.rag.embeddings import EmbeddingProvider, get_embedding_provider
from app.rag.vectorstore import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


@dataclass
class RetrievedPassage:
    chunk_id: str
    doc_id: str
    source: str
    title: str
    category: str
    chunk_index: int
    text: str
    score: float

    def as_source_ref(self) -> dict:
        """Compact citation payload returned to API clients."""
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "source": self.source,
            "chunk_index": self.chunk_index,
            "score": round(self.score, 4),
            "snippet": self.text[:240],
        }


class Retriever:
    def __init__(
        self,
        embeddings: EmbeddingProvider | None = None,
        store: VectorStore | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.embeddings = embeddings or get_embedding_provider(self.settings)
        self.store = store or get_vector_store(self.settings, dimension=self.embeddings.dimension)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> list[RetrievedPassage]:
        top_k = top_k or self.settings.top_k
        min_score = self.settings.min_retrieval_score if min_score is None else min_score

        vector = self.embeddings.embed_query(query)
        matches = self.store.query(vector, top_k=top_k)
        passages = [
            RetrievedPassage(
                chunk_id=m.id,
                doc_id=m.metadata.get("doc_id", ""),
                source=m.metadata.get("source", ""),
                title=m.metadata.get("title", ""),
                category=m.metadata.get("category", ""),
                chunk_index=int(m.metadata.get("chunk_index", 0)),
                text=m.metadata.get("text", ""),
                score=m.score,
            )
            for m in matches
        ]
        kept = [p for p in passages if p.score >= min_score]
        logger.debug(
            "retrieval query=%r top_k=%d matches=%d kept=%d scores=%s",
            query, top_k, len(passages), len(kept),
            [round(p.score, 3) for p in passages],
        )
        return kept
