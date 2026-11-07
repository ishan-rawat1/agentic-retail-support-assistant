"""LangGraph agent workflow.

Graph topology (observable, no hidden reasoning):

    START -> route -+-> direct ----------------------------> END
                    +-> retrieve -> generate_rag ----------> END
                    +-> tool_agent <-> tool_executor (loop) -> END

* ``route``         LLM structured-output classification (direct|rag|tool).
* ``retrieve``      embeds the query and searches the vector store.
* ``generate_rag``  grounded answer with citations; explicit uncertainty
                    when retrieval returns nothing relevant.
* ``tool_agent``    LLM with bound tool schemas; may emit tool calls.
* ``tool_executor`` executes allowlisted tools through the audited
                    registry (customer-scoped, DB-backed) and feeds
                    ToolMessages back to the tool_agent.
"""

import json
import logging
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from sqlalchemy.orm import Session

from app.agent.llm import get_chat_model
from app.agent.prompts.registry import get_prompt_pack
from app.agent.state import AgentState, RouteDecision
from app.config import Settings, get_settings
from app.rag.retriever import Retriever
from app.services.tool_registry import TOOL_REGISTRY, execute_tool

logger = logging.getLogger(__name__)

MAX_HISTORY_MESSAGES = 12

NO_CONTEXT_ANSWER = (
    "I'm sorry — I couldn't find anything about that in our documentation, so I "
    "don't want to guess. Please reach out to support@example-retail.com and a "
    "human agent will help you within 1 business day."
)


def _tool_schemas() -> list[dict]:
    """OpenAI-style tool schemas derived from the allowlisted registry."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.args_model.model_json_schema(),
        }
        for spec in TOOL_REGISTRY.values()
    ]


def format_context(passages: list[dict]) -> str:
    if not passages:
        return "(no relevant documents found)"
    blocks = [
        f"[{p['title']}] (source: {p['source']}, chunk {p['chunk_index']}, score {p['score']:.3f})\n{p['text']}"
        for p in passages
    ]
    return "\n---\n".join(blocks)


class AgentRuntime:
    """Per-request runtime binding the graph to a DB session and customer."""

    def __init__(
        self,
        db: Session,
        customer_id: int,
        session_id: str | None = None,
        *,
        retriever: Retriever | None = None,
        llm=None,
        settings: Settings | None = None,
        prompt_version: str | None = None,
        top_k: int | None = None,
    ) -> None:
        self.db = db
        self.customer_id = customer_id
        self.session_id = session_id
        self.settings = settings or get_settings()
        self.retriever = retriever or Retriever(settings=self.settings)
        self.llm = llm or get_chat_model(self.settings)
        self.prompts = get_prompt_pack(prompt_version or self.settings.prompt_version)
        self.top_k = top_k or self.settings.top_k
        self.graph = self._build()

    # --- nodes --------------------------------------------------------

    def _route(self, state: AgentState) -> dict:
        router = self.llm.with_structured_output(RouteDecision)
        messages = [SystemMessage(self.prompts.router_system), *state["messages"]]
        decision: RouteDecision = router.invoke(messages)
        logger.info("route=%s reason=%s", decision.route, decision.reason)
        return {"route": decision.route, "route_reason": decision.reason}

    def _direct(self, state: AgentState) -> dict:
        response = self.llm.invoke(
            [SystemMessage(self.prompts.direct_system), *state["messages"]]
        )
        return {"messages": [response]}

    def _retrieve(self, state: AgentState) -> dict:
        query = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, HumanMessage):
                query = str(msg.content)
                break
        passages = self.retriever.retrieve(query, top_k=self.top_k)
        return {
            "passages": [
                {
                    "chunk_id": p.chunk_id,
                    "doc_id": p.doc_id,
                    "source": p.source,
                    "title": p.title,
                    "category": p.category,
                    "chunk_index": p.chunk_index,
                    "text": p.text,
                    "score": p.score,
                }
                for p in passages
            ]
        }

    def _generate_rag(self, state: AgentState) -> dict:
        passages = state.get("passages", [])
        if not passages:
            # Deterministic uncertainty path: no relevant context above
            # threshold, so refuse to guess regardless of provider.
            return {"messages": [AIMessage(content=NO_CONTEXT_ANSWER)]}
        system = self.prompts.rag_system_template.format(context=format_context(passages))
        response = self.llm.invoke([SystemMessage(system), *state["messages"]])
        return {"messages": [response]}

    def _tool_agent(self, state: AgentState) -> dict:
        llm_with_tools = self.llm.bind_tools(_tool_schemas())
        response = llm_with_tools.invoke(
            [SystemMessage(self.prompts.tool_system), *state["messages"]]
        )
        return {
            "messages": [response],
            "tool_iterations": state.get("tool_iterations", 0) + 1,
        }

    def _tool_executor(self, state: AgentState) -> dict:
        last = state["messages"][-1]
        tool_messages: list[ToolMessage] = []
        invocations = list(state.get("tool_invocations", []))
        for call in getattr(last, "tool_calls", []) or []:
            outcome = execute_tool(
                self.db,
                tool_name=call["name"],
                arguments=call.get("args") or {},
                customer_id=self.customer_id,
                session_id=self.session_id,
            )
            invocations.append(
                {
                    "name": call["name"],
                    "arguments": call.get("args") or {},
                    "success": outcome["success"],
                    "result": outcome["result"],
                    "error": outcome["error"],
                }
            )
            tool_messages.append(
                ToolMessage(content=json.dumps(outcome, default=str), tool_call_id=call["id"])
            )
        return {"messages": tool_messages, "tool_invocations": invocations}

    # --- edges --------------------------------------------------------

    def _route_edge(self, state: AgentState) -> Literal["direct", "retrieve", "tool_agent"]:
        return {"direct": "direct", "rag": "retrieve", "tool": "tool_agent"}[state["route"]]

    def _tools_edge(self, state: AgentState) -> Literal["tool_executor", "__end__"]:
        last = state["messages"][-1]
        has_calls = bool(getattr(last, "tool_calls", None))
        within_budget = state.get("tool_iterations", 0) <= self.settings.max_tool_iterations
        return "tool_executor" if has_calls and within_budget else END

    # --- graph --------------------------------------------------------

    def _build(self):
        graph = StateGraph(AgentState)
        graph.add_node("route", self._route)
        graph.add_node("direct", self._direct)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("generate_rag", self._generate_rag)
        graph.add_node("tool_agent", self._tool_agent)
        graph.add_node("tool_executor", self._tool_executor)

        graph.add_edge(START, "route")
        graph.add_conditional_edges("route", self._route_edge)
        graph.add_edge("direct", END)
        graph.add_edge("retrieve", "generate_rag")
        graph.add_edge("generate_rag", END)
        graph.add_conditional_edges("tool_agent", self._tools_edge)
        graph.add_edge("tool_executor", "tool_agent")
        return graph.compile()

    # --- entry point ----------------------------------------------------

    def run(self, user_message: str, history: list[dict] | None = None) -> dict:
        """Run one turn. ``history`` is [{role, content}, ...] from persistence."""
        messages = []
        for item in (history or [])[-MAX_HISTORY_MESSAGES:]:
            cls = HumanMessage if item["role"] == "user" else AIMessage
            messages.append(cls(content=item["content"]))
        messages.append(HumanMessage(content=user_message))

        final_state: AgentState = self.graph.invoke(
            {"messages": messages, "tool_invocations": [], "tool_iterations": 0}
        )

        answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and msg.content:
                answer = str(msg.content)
                break

        return {
            "answer": answer,
            "route": final_state.get("route", ""),
            "route_reason": final_state.get("route_reason", ""),
            "sources": [
                {
                    "doc_id": p["doc_id"],
                    "title": p["title"],
                    "source": p["source"],
                    "chunk_index": p["chunk_index"],
                    "score": round(p["score"], 4),
                    "snippet": p["text"][:240],
                }
                for p in final_state.get("passages", [])
            ],
            "tool_invocations": final_state.get("tool_invocations", []),
        }
