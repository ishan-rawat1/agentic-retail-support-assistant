"""Centralized application configuration.

All runtime configuration comes from environment variables (or a local
``.env`` file) via pydantic-settings. Nothing in the codebase hardcodes
credentials; when OpenAI / Pinecone credentials are absent the app
transparently falls back to deterministic local providers so the full
stack stays runnable offline (demos, CI, unit tests).
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- General -----------------------------------------------------
    app_name: str = "Agentic Retail Support Assistant"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Database ----------------------------------------------------
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'retail_support.db').as_posix()}"

    # --- OpenAI ------------------------------------------------------
    openai_api_key: str = ""
    chat_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"

    # --- Pinecone ----------------------------------------------------
    pinecone_api_key: str = ""
    pinecone_index: str = "retail-support"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Providers ---------------------------------------------------
    # "auto" picks the real provider when credentials exist, otherwise a
    # deterministic local fallback. Explicit values force a choice.
    llm_provider: Literal["auto", "openai", "mock"] = "auto"
    vector_backend: Literal["auto", "pinecone", "local"] = "auto"
    embedding_provider: Literal["auto", "openai", "local"] = "auto"
    local_vector_store_path: str = str(BACKEND_DIR / "local_vector_store.json")
    local_embedding_dim: int = 512

    # --- RAG ---------------------------------------------------------
    docs_dir: str = str(BACKEND_DIR / "app" / "data" / "docs")
    chunk_size: int = 800
    chunk_overlap: int = 120
    top_k: int = 4
    min_retrieval_score: float = 0.15

    # --- Agent -------------------------------------------------------
    prompt_version: str = "v2"
    max_tool_iterations: int = 4

    # ------------------------------------------------------------------
    @property
    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        return "openai" if self.openai_api_key else "mock"

    @property
    def resolved_vector_backend(self) -> str:
        if self.vector_backend != "auto":
            return self.vector_backend
        return "pinecone" if self.pinecone_api_key else "local"

    @property
    def resolved_embedding_provider(self) -> str:
        if self.embedding_provider != "auto":
            return self.embedding_provider
        return "openai" if self.openai_api_key else "local"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
