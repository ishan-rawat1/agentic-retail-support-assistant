"""Pydantic request/response contracts for the public API."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# --- Chat ------------------------------------------------------------

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000, description="The customer's message.")
    session_id: str | None = Field(
        default=None,
        description="Existing session id; omit to start a new conversation.",
    )


class SourceRef(BaseModel):
    doc_id: str
    title: str
    source: str
    chunk_index: int
    score: float
    snippet: str


class ToolInvocationRecord(BaseModel):
    name: str
    arguments: dict[str, Any]
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: str = Field(description="Which agent path handled the turn: direct | rag | tool.")
    route_reason: str = ""
    sources: list[SourceRef] = []
    tool_invocations: list[ToolInvocationRecord] = []


# --- Sessions ----------------------------------------------------------

class SessionCreateRequest(BaseModel):
    title: str | None = Field(default=None, max_length=200)


class SessionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class MessageResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    role: str
    content: str
    route: str | None = None
    sources: list[dict] | None = None
    tool_calls: list[dict] | None = None
    created_at: datetime


# --- Tools -------------------------------------------------------------

class ToolInvokeRequest(BaseModel):
    tool: str = Field(description="Allowlisted tool name, e.g. lookup_order.")
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolInvokeResponse(BaseModel):
    tool: str
    success: bool
    result: dict[str, Any] | None = None
    error: str | None = None


class ToolInfo(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any]


# --- Misc ----------------------------------------------------------------

class CustomerResponse(BaseModel):
    external_id: str
    name: str
    email: str


class HealthResponse(BaseModel):
    status: str
    llm_provider: str
    vector_backend: str
    embedding_provider: str
    prompt_version: str
    vector_count: int | None = None


class ErrorResponse(BaseModel):
    detail: str
    code: str = "error"
