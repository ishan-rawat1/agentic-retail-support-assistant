"""Unit tests: embeddings, vector store, and retrieval interfaces."""

import pytest

from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.retriever import Retriever
from app.rag.vectorstore import LocalVectorStore


def test_local_embeddings_are_deterministic_and_normalized():
    provider = LocalHashEmbeddingProvider(dimension=128)
    a = provider.embed_query("What is your return policy?")
    b = provider.embed_query("What is your return policy?")
    assert a == b
    assert len(a) == 128
    norm = sum(x * x for x in a) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-6)


def test_local_vector_store_upsert_query_and_overwrite(tmp_path):
    store = LocalVectorStore(tmp_path / "v.json")
    store.upsert(["a", "b"], [[1.0, 0.0], [0.0, 1.0]], [{"t": "a"}, {"t": "b"}])
    assert store.count() == 2

    matches = store.query([1.0, 0.1], top_k=2)
    assert matches[0].id == "a"
    assert matches[0].score > matches[1].score
    assert matches[0].metadata == {"t": "a"}

    # Upserting an existing id overwrites instead of duplicating.
    store.upsert(["a"], [[0.0, 1.0]], [{"t": "a2"}])
    assert store.count() == 2
    assert store.query([0.0, 1.0], top_k=1)[0].metadata["t"] in {"a2", "b"}

    # Persistence round-trip.
    reloaded = LocalVectorStore(tmp_path / "v.json")
    assert reloaded.count() == 2


def test_retriever_finds_correct_documents(_bootstrap):
    retriever = Retriever()
    cases = {
        "What is your return policy?": "returns-policy",
        "How long does standard shipping take?": "shipping-policy",
        "Do the Aurora headphones have noise cancellation?": "product-catalog",
        "What payment methods do you accept?": "payments-and-cancellations",
    }
    for query, expected_doc in cases.items():
        passages = retriever.retrieve(query)
        assert passages, f"no passages for {query!r}"
        assert passages[0].doc_id == expected_doc, (
            f"{query!r}: expected top doc {expected_doc}, got {passages[0].doc_id}"
        )
        # Scores are sorted descending and metadata survives the round trip.
        scores = [p.score for p in passages]
        assert scores == sorted(scores, reverse=True)
        assert passages[0].title and passages[0].source and passages[0].text


def test_retriever_respects_top_k_and_min_score(_bootstrap):
    retriever = Retriever()
    assert len(retriever.retrieve("shipping", top_k=2, min_score=0.0)) <= 2
    assert retriever.retrieve("completely unrelated quantum pets", min_score=0.9) == []
