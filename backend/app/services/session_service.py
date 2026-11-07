"""Chat session and message persistence, isolated per customer."""

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import ChatMessage, ChatSession, utcnow
from app.services.exceptions import NotFoundError


def create_session(db: Session, customer_id: int, title: str = "New conversation") -> ChatSession:
    session = ChatSession(customer_id=customer_id, title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session_for_customer(db: Session, customer_id: int, session_id: str) -> ChatSession:
    session = db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.customer_id == customer_id
        )
    ).scalar_one_or_none()
    if session is None:
        # Same error for "doesn't exist" and "not yours" — no enumeration.
        raise NotFoundError(f"No session {session_id} found for your account.")
    return session


def list_sessions(db: Session, customer_id: int) -> list[ChatSession]:
    return list(
        db.execute(
            select(ChatSession)
            .where(ChatSession.customer_id == customer_id)
            .order_by(ChatSession.updated_at.desc())
        ).scalars()
    )


def get_history(db: Session, customer_id: int, session_id: str) -> list[ChatMessage]:
    session = get_session_for_customer(db, customer_id, session_id)
    return list(session.messages)


def append_message(
    db: Session,
    session_id: str,
    role: str,
    content: str,
    *,
    route: str | None = None,
    sources: list | None = None,
    tool_calls: list | None = None,
) -> ChatMessage:
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
        route=route,
        sources=sources,
        tool_calls=tool_calls,
    )
    db.add(message)
    # Touch the parent session so updated_at reflects latest activity.
    session = db.get(ChatSession, session_id)
    if session is not None:
        if session.title == "New conversation" and role == "user":
            session.title = content[:60] + ("…" if len(content) > 60 else "")
        session.updated_at = utcnow()
    db.commit()
    db.refresh(message)
    return message
