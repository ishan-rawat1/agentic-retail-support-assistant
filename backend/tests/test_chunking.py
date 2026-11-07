"""Unit tests: document loading and chunking."""

from app.config import get_settings
from app.rag.chunking import chunk_documents
from app.rag.loader import KBDocument, load_documents, normalize_text


def _docs():
    return load_documents(get_settings().docs_dir)


def test_loads_all_documents_with_metadata():
    docs = _docs()
    assert len(docs) >= 6
    ids = {d.doc_id for d in docs}
    assert {"returns-policy", "shipping-policy", "faq", "product-catalog"} <= ids
    for doc in docs:
        assert doc.title and doc.category and doc.text
        # Frontmatter must be stripped from the body.
        assert not doc.text.startswith("---")


def test_normalize_text_collapses_whitespace():
    raw = "line one   \r\nline two\r\n\n\n\n\nline three"
    assert normalize_text(raw) == "line one\nline two\n\nline three"


def test_chunks_respect_size_and_carry_metadata():
    chunks = chunk_documents(_docs(), chunk_size=500, chunk_overlap=80)
    assert chunks
    for chunk in chunks:
        assert len(chunk.text) <= 500 + 80  # splitter may keep a separator
        meta = chunk.to_metadata()
        assert meta["doc_id"] and meta["source"].endswith(".md")
        assert meta["text"] == chunk.text
        assert isinstance(meta["chunk_index"], int)


def test_chunk_overlap_shares_content_between_neighbors():
    doc = KBDocument(
        doc_id="x", source="x.md", title="X", category="test",
        text=" ".join(f"word{i}" for i in range(400)),
    )
    chunks = chunk_documents([doc], chunk_size=300, chunk_overlap=100)
    assert len(chunks) > 1
    first_words = set(chunks[0].text.split())
    second_words = set(chunks[1].text.split())
    assert first_words & second_words, "adjacent chunks should overlap"


def test_chunk_ids_are_deterministic_and_unique():
    docs = _docs()
    a = chunk_documents(docs, chunk_size=500, chunk_overlap=80)
    b = chunk_documents(docs, chunk_size=500, chunk_overlap=80)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert len({c.chunk_id for c in a}) == len(a)


def test_reingest_is_duplicate_safe(tmp_path):
    from app.rag.embeddings import LocalHashEmbeddingProvider
    from app.rag.ingest import ingest
    from app.rag.vectorstore import LocalVectorStore

    embeddings = LocalHashEmbeddingProvider(128)
    store = LocalVectorStore(tmp_path / "v.json")
    first = ingest(embeddings=embeddings, store=store)
    second = ingest(embeddings=embeddings, store=store)
    assert first["vector_count"] == second["vector_count"]
