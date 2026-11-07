"""FastAPI dependencies: DB session, authentication, shared singletons.

Authentication is a deliberately simple demo scheme: each seeded
customer has a static bearer token (``Authorization: Bearer demo-alice``).
Every data access downstream is scoped to the resolved customer, so
guessing another customer's order or session ID returns 404. In
production this dependency would validate an OIDC/JWT token instead —
the rest of the code would not change.
"""

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, Header
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db.base import get_db
from app.db.models import Customer
from app.rag.retriever import Retriever
from app.services.exceptions import AuthenticationError


def get_current_customer(
    db: Annotated[Session, Depends(get_db)],
    authorization: Annotated[str | None, Header()] = None,
) -> Customer:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise AuthenticationError(
            "Missing credentials. Send 'Authorization: Bearer <demo token>' "
            "(e.g. demo-alice — see README for demo identities)."
        )
    token = authorization.split(" ", 1)[1].strip()
    customer = db.execute(
        select(Customer).where(Customer.api_token == token)
    ).scalar_one_or_none()
    if customer is None:
        raise AuthenticationError("Invalid token.")
    return customer


@lru_cache
def get_retriever() -> Retriever:
    """Process-wide retriever (embedding provider + vector store)."""
    return Retriever(settings=get_settings())


SettingsDep = Annotated[Settings, Depends(get_settings)]
DbDep = Annotated[Session, Depends(get_db)]
CustomerDep = Annotated[Customer, Depends(get_current_customer)]
RetrieverDep = Annotated[Retriever, Depends(get_retriever)]
