"""Optional integration tests against REAL OpenAI / Pinecone services.

Skipped automatically unless credentials are present AND the
integration marker is requested:

    pytest -m integration

These verify the thin provider wrappers; all business logic is covered
by the offline unit suite.
"""

import os

import pytest

pytestmark = pytest.mark.integration

requires_openai = pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"), reason="OPENAI_API_KEY not set"
)
requires_pinecone = pytest.mark.skipif(
    not os.environ.get("PINECONE_API_KEY"), reason="PINECONE_API_KEY not set"
)


@requires_openai
def test_openai_embeddings_roundtrip():
    from app.rag.embeddings import OpenAIEmbeddingProvider

    provider = OpenAIEmbeddingProvider(
        model=os.environ.get("EMBEDDING_MODEL", "text-embedding-3-small"),
        api_key=os.environ["OPENAI_API_KEY"],
    )
    vectors = provider.embed_texts(["return policy", "shipping times"])
    assert len(vectors) == 2
    assert len(vectors[0]) == provider.dimension


@requires_openai
def test_openai_chat_model_responds():
    from langchain_core.messages import HumanMessage

    from app.agent.llm import get_chat_model
    from app.config import Settings

    settings = Settings(llm_provider="openai")
    llm = get_chat_model(settings)
    response = llm.invoke([HumanMessage(content="Reply with the single word: pong")])
    assert "pong" in response.content.lower()


@requires_pinecone
@requires_openai
def test_pinecone_upsert_and_query():
    from app.config import Settings
    from app.rag.embeddings import get_embedding_provider
    from app.rag.vectorstore import PineconeVectorStore

    settings = Settings(vector_backend="pinecone", embedding_provider="openai")
    embeddings = get_embedding_provider(settings)
    store = PineconeVectorStore(settings, dimension=embeddings.dimension)

    vector = embeddings.embed_query("integration test sentinel")
    store.upsert(["itest-1"], [vector], [{"text": "integration test sentinel", "doc_id": "itest"}])
    matches = store.query(vector, top_k=1)
    assert matches and matches[0].id == "itest-1"
