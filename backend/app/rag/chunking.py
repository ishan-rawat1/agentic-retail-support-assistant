"""Document chunking with deterministic, duplicate-safe chunk IDs."""

import hashlib
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.rag.loader import KBDocument


@dataclass
class Chunk:
    chunk_id: str        # deterministic: "<doc_id>::<index>::<content-hash-8>"
    doc_id: str
    source: str
    title: str
    category: str
    chunk_index: int
    text: str
    metadata: dict = field(default_factory=dict)

    def to_metadata(self) -> dict:
        """Metadata payload stored alongside the vector."""
        return {
            "doc_id": self.doc_id,
            "source": self.source,
            "title": self.title,
            "category": self.category,
            "chunk_index": self.chunk_index,
            "text": self.text,
        }


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


def chunk_documents(
    documents: list[KBDocument],
    *,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[Chunk]:
    """Split documents into overlapping chunks.

    Markdown-aware separators keep headings attached to their section
    bodies. IDs are deterministic (doc id + index + content hash) so
    re-ingesting unchanged documents upserts the same vector IDs instead
    of creating duplicates.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        keep_separator=True,
    )
    chunks: list[Chunk] = []
    for doc in documents:
        for idx, piece in enumerate(splitter.split_text(doc.text)):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.doc_id}::{idx}::{_content_hash(piece)}",
                    doc_id=doc.doc_id,
                    source=doc.source,
                    title=doc.title,
                    category=doc.category,
                    chunk_index=idx,
                    text=piece,
                )
            )
    return chunks
