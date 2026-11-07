"""Shared test fixtures.

Environment variables are pinned BEFORE any ``app`` module is imported
(pytest loads conftest first), so:

* the DB is a throwaway temp SQLite file,
* the vector store is a throwaway local JSON file,
* LLM / embeddings / vector backends are forced to the offline
  deterministic providers — the unit suite never performs network
  calls even if real credentials exist in ``.env``.
"""

import os
import tempfile
from pathlib import Path

_TMPDIR = Path(tempfile.mkdtemp(prefix="retail-support-tests-"))

os.environ.update(
    {
        "DATABASE_URL": f"sqlite:///{(_TMPDIR / 'test.db').as_posix()}",
        "LOCAL_VECTOR_STORE_PATH": str(_TMPDIR / "vectors.json"),
        "LLM_PROVIDER": "mock",
        "VECTOR_BACKEND": "local",
        "EMBEDDING_PROVIDER": "local",
        "ENVIRONMENT": "test",
        "LOG_LEVEL": "WARNING",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.config import get_settings  # noqa: E402

get_settings.cache_clear()

from app.db.base import SessionLocal  # noqa: E402
from app.db.seed import seed  # noqa: E402
from app.rag.ingest import ingest  # noqa: E402

ALICE = {"Authorization": "Bearer demo-alice"}
BOB = {"Authorization": "Bearer demo-bob"}
CAROL = {"Authorization": "Bearer demo-carol"}


@pytest.fixture(scope="session", autouse=True)
def _bootstrap():
    """Seed the temp DB and ingest docs into the temp vector store once."""
    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
    ingest(clear=True)
    yield


@pytest.fixture(scope="session")
def client(_bootstrap):
    from app.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def db_session(_bootstrap):
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture()
def alice_id(db_session):
    from sqlalchemy import select

    from app.db.models import Customer

    return db_session.execute(
        select(Customer.id).where(Customer.api_token == "demo-alice")
    ).scalar_one()


@pytest.fixture()
def bob_id(db_session):
    from sqlalchemy import select

    from app.db.models import Customer

    return db_session.execute(
        select(Customer.id).where(Customer.api_token == "demo-bob")
    ).scalar_one()
