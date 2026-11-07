"""LangGraph agent state and structured-output schemas."""

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class RouteDecision(BaseModel):
    """Structured output schema for the routing node."""

    route: Literal["direct", "rag", "tool"] = Field(
        description="Which path should handle the customer's latest message."
    )
    reason: str = Field(description="One-sentence justification for the chosen route.")


class AgentState(TypedDict, total=False):
    # Conversation messages (history + this turn). add_messages merges
    # node updates instead of overwriting.
    messages: Annotated[list[AnyMessage], add_messages]
    # Routing decision for this turn.
    route: str
    route_reason: str
    # RAG artifacts: serialized retrieved passages (with scores) for
    # grounding, citation, and debugging/eval.
    passages: list[dict]
    # Tool invocations executed this turn: {name, arguments, success, result|error}.
    tool_invocations: list[dict]
    # Loop guard for the tool-calling cycle.
    tool_iterations: int
