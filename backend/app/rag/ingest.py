"""CLI ingestion pipeline: load -> normalize -> chunk -> embed -> upsert.

Usage (from the repo root, venv active):

    python -m app.rag.ingest                 # ingest with configured settings
    python -m app.rag.ingest --clear         # wipe the index first
    python -m app.rag.ingest --chunk-size 500 --chunk-overlap 80

Deterministic chunk IDs make re-ingestion duplicate-safe: unchanged
chunks overwrite themselves, changed chunks get new content-hash IDs.
"""

import argparse
import logging

from app.config import get_settings
from app.rag.chunking import chunk_documents
from app.rag.embeddings import get_embedding_provider
from app.rag.loader import load_documents
from app.rag.vectorstore import get_vector_store

logger = logging.getLogger(__name__)


def ingest(
    *,
    docs_dir: str | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    clear: bool = False,
    embeddings=None,
    store=None,
) -> dict:
    """Run the full ingestion pipeline. Returns summary stats."""
    settings = get_settings()
    docs_dir = docs_dir or settings.docs_dir
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap

    embeddings = embeddings or get_embedding_provider(settings)
    store = store or get_vector_store(settings, dimension=embeddings.dimension)

    if clear:
        logger.info("Clearing vector store before ingestion")
        store.clear()

    documents = load_documents(docs_dir)
    chunks = chunk_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    logger.info("Chunked %d documents into %d chunks (size=%d overlap=%d)",
                len(documents), len(chunks), chunk_size, chunk_overlap)

    vectors = embeddings.embed_texts([c.text for c in chunks])
    store.upsert(
        ids=[c.chunk_id for c in chunks],
        vectors=vectors,
        metadatas=[c.to_metadata() for c in chunks],
    )

    stats = {
        "documents": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "vector_count": store.count(),
        "embedding_provider": settings.resolved_embedding_provider,
        "vector_backend": settings.resolved_vector_backend,
    }
    logger.info("Ingestion complete: %s", stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest knowledge-base docs into the vector store.")
    parser.add_argument("--docs-dir", default=None)
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    parser.add_argument("--clear", action="store_true", help="Clear the index before ingesting")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stats = ingest(
        docs_dir=args.docs_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        clear=args.clear,
    )
    print("Ingestion summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
