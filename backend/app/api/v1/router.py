"""Versioned API router: all customer-facing endpoints under /api/v1."""

import json

from fastapi import APIRouter, Query

from app.agent.graph import AgentRuntime
from app.api.deps import CustomerDep, DbDep, RetrieverDep, SettingsDep
from app.schemas.api import (
    ChatRequest,
    ChatResponse,
    CustomerResponse,
    HealthResponse,
    MessageResponse,
    SessionCreateRequest,
    SessionResponse,
    ToolInfo,
    ToolInvokeRequest,
    ToolInvokeResponse,
)
from app.services import session_service
from app.services.tool_registry import TOOL_REGISTRY, execute_tool

router = APIRouter(prefix="/api/v1")


# --- Health / readiness -------------------------------------------------

@router.get("/health", response_model=HealthResponse, tags=["health"])
def health(settings: SettingsDep, retriever: RetrieverDep) -> HealthResponse:
    try:
        vector_count = retriever.store.count()
    except Exception:
        vector_count = None
    return HealthResponse(
        status="ok",
        llm_provider=settings.resolved_llm_provider,
        vector_backend=settings.resolved_vector_backend,
        embedding_provider=settings.resolved_embedding_provider,
        prompt_version=settings.prompt_version,
        vector_count=vector_count,
    )


@router.get("/ready", tags=["health"])
def ready(db: DbDep) -> dict:
    # Readiness: DB reachable.
    db.execute(__import__("sqlalchemy").text("SELECT 1"))
    return {"status": "ready"}


# --- Identity -------------------------------------------------------------

@router.get("/me", response_model=CustomerResponse, tags=["customers"])
def me(customer: CustomerDep) -> CustomerResponse:
    return CustomerResponse(
        external_id=customer.external_id, name=customer.name, email=customer.email
    )


# --- Chat ------------------------------------------------------------------

@router.post("/chat", response_model=ChatResponse, tags=["chat"])
def chat(
    body: ChatRequest,
    db: DbDep,
    customer: CustomerDep,
    retriever: RetrieverDep,
    settings: SettingsDep,
) -> ChatResponse:
    """Send a message to the agent within a (new or existing) session."""
    if body.session_id:
        session = session_service.get_session_for_customer(db, customer.id, body.session_id)
    else:
        session = session_service.create_session(db, customer.id)

    history = [
        {"role": m.role, "content": m.content}
        for m in session_service.get_history(db, customer.id, session.id)
    ]

    session_service.append_message(db, session.id, "user", body.message)

    runtime = AgentRuntime(
        db, customer.id, session.id, retriever=retriever, settings=settings
    )
    result = runtime.run(body.message, history)

    # Round-trip through JSON so DB columns hold plain serializable data.
    tool_calls = json.loads(json.dumps(result["tool_invocations"], default=str))
    session_service.append_message(
        db,
        session.id,
        "assistant",
        result["answer"],
        route=result["route"],
        sources=result["sources"],
        tool_calls=tool_calls,
    )

    return ChatResponse(
        session_id=session.id,
        answer=result["answer"],
        route=result["route"],
        route_reason=result["route_reason"],
        sources=result["sources"],
        tool_invocations=tool_calls,
    )


# --- Sessions ----------------------------------------------------------------

@router.post("/sessions", response_model=SessionResponse, status_code=201, tags=["sessions"])
def create_session(body: SessionCreateRequest, db: DbDep, customer: CustomerDep) -> SessionResponse:
    session = session_service.create_session(
        db, customer.id, title=body.title or "New conversation"
    )
    return SessionResponse.model_validate(session)


@router.get("/sessions", response_model=list[SessionResponse], tags=["sessions"])
def list_sessions(db: DbDep, customer: CustomerDep) -> list[SessionResponse]:
    return [
        SessionResponse.model_validate(s)
        for s in session_service.list_sessions(db, customer.id)
    ]


@router.get(
    "/sessions/{session_id}/history",
    response_model=list[MessageResponse],
    tags=["sessions"],
)
def session_history(session_id: str, db: DbDep, customer: CustomerDep) -> list[MessageResponse]:
    return [
        MessageResponse.model_validate(m)
        for m in session_service.get_history(db, customer.id, session_id)
    ]


# --- Tools ----------------------------------------------------------------------

@router.get("/tools", response_model=list[ToolInfo], tags=["tools"])
def list_tools() -> list[ToolInfo]:
    return [
        ToolInfo(
            name=spec.name,
            description=spec.description,
            parameters=spec.args_model.model_json_schema(),
        )
        for spec in TOOL_REGISTRY.values()
    ]


@router.post("/tools/invoke", response_model=ToolInvokeResponse, tags=["tools"])
def invoke_tool(
    body: ToolInvokeRequest,
    db: DbDep,
    customer: CustomerDep,
    session_id: str | None = Query(default=None),
) -> ToolInvokeResponse:
    """Execute an allowlisted tool directly (debugging / integration).

    Tool names outside the registry are rejected (422); arguments are
    validated against the tool's Pydantic schema; execution is scoped to
    the authenticated customer and audit-logged.
    """
    outcome = execute_tool(
        db,
        tool_name=body.tool,
        arguments=body.arguments,
        customer_id=customer.id,
        session_id=session_id,
    )
    return ToolInvokeResponse(tool=body.tool, **outcome)
