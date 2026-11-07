"""Embedding providers.

Two implementations behind one interface:

* ``OpenAIEmbeddingProvider`` — real ``text-embedding-3-*`` embeddings.
* ``LocalHashEmbeddingProvider`` — a deterministic, dependency-free
  fallback (hashed bag-of-words with L2 normalization). It is not
  semantically rich, but it is stable, fast, free, and good enough for
  keyword-heavy retail FAQ retrieval — which keeps the entire stack
  (demo, tests, evals) runnable without credentials.
"""

import hashlib
import math
import re
from typing import Protocol

from app.config import Settings, get_settings


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]: ...
    def embed_query(self, text: str) -> list[float]: ...


_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Trivial English stopwords; keeps the hashed space focused on content words.
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have how i if in is it its my of on or "
    "that the this to was what when where which who will with you your can do does".split()
)


# Words so frequent across retail policy docs that they barely
# discriminate between documents; downweighted, not removed.
_COMMON_RETAIL_TERMS = frozenset(
    "policy policie order item product customer support day business store "
    "contact email purchase account within".split()
)


class LocalHashEmbeddingProvider:
    """Deterministic hashed bag-of-words embeddings (offline fallback)."""

    def __init__(self, dimension: int = 512) -> None:
        self.dimension = dimension

    @staticmethod
    def _stem(word: str) -> str:
        """Crude suffix stripping so 'returns'/'returned'/'returning' collide."""
        for suffix in ("ing", "ies", "ed", "es", "s"):
            if word.endswith(suffix) and len(word) - len(suffix) >= 3:
                return word[: -len(suffix)] + ("y" if suffix == "ies" else "")
        return word

    def _tokens(self, text: str) -> list[tuple[str, float]]:
        words = [
            self._stem(w) for w in _TOKEN_RE.findall(text.lower()) if w not in _STOPWORDS
        ]
        weighted = [(w, 2.0) for w in words]
        weighted += [(f"{a}_{b}", 1.0) for a, b in zip(words, words[1:])]
        return weighted

    def _embed(self, text: str) -> list[float]:
        # Accumulate weighted term counts, then apply sublinear TF so
        # long chunks with many repeated terms don't drown the signal.
        counts: dict[int, float] = {}
        for token, weight in self._tokens(text):
            digest = hashlib.md5(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            common = token in _COMMON_RETAIL_TERMS
            counts[index] = counts.get(index, 0.0) + weight * (0.4 if common else 1.0)
        vec = [0.0] * self.dimension
        for index, count in counts.items():
            vec[index] = 1.0 + math.log(count) if count > 0 else 0.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbeddingProvider:
    """OpenAI embeddings via langchain-openai."""

    def __init__(self, model: str, api_key: str) -> None:
        from langchain_openai import OpenAIEmbeddings

        self._client = OpenAIEmbeddings(model=model, api_key=api_key)
        # text-embedding-3-small -> 1536, -3-large -> 3072
        self.dimension = 3072 if "large" in model else 1536

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._client.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._client.embed_query(text)


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    settings = settings or get_settings()
    if settings.resolved_embedding_provider == "openai":
        return OpenAIEmbeddingProvider(settings.embedding_model, settings.openai_api_key)
    return LocalHashEmbeddingProvider(settings.local_embedding_dim)
