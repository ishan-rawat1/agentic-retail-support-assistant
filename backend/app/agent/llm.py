"""Chat model providers.

``get_chat_model`` returns either a real ``ChatOpenAI`` or a
``MockChatModel`` — a deterministic, rule-based ``BaseChatModel``
implementation. The mock honors the exact same interfaces the graph
uses (``invoke``, ``bind_tools``, ``with_structured_output``), so the
LangGraph workflow is identical in both modes; only the text/tool-call
generation strategy differs. That keeps demos, unit tests, and the
eval harness runnable and deterministic without API keys.
"""

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import RunnableLambda

from app.agent.state import RouteDecision
from app.config import Settings, get_settings

ORDER_ID_RE = re.compile(r"\bORD-\d{3,8}\b", re.IGNORECASE)
PRODUCT_ID_RE = re.compile(r"\bPROD-\d{3,8}\b", re.IGNORECASE)

_GREETING_RE = re.compile(
    r"^(hi|hiya|hello|hey|yo|good (morning|afternoon|evening)|thanks|thank you|ok|okay|bye|goodbye)\b",
    re.IGNORECASE,
)
_CAPABILITY_RE = re.compile(r"\b(what can you do|who are you|how can you help|help me)\b", re.IGNORECASE)
_RETURN_INTENT_RE = re.compile(r"\b(return|refund|send (it|this) back)\b", re.IGNORECASE)
_STATUS_INTENT_RE = re.compile(
    r"\b(where('?s| is)?|track|tracking|status|arrive|arriving|delivery|delivered|shipped|what('?s| is| was) in)\b",
    re.IGNORECASE,
)
_MY_ORDER_RE = re.compile(r"\b(my (order|package|delivery)|\bit\b|\bthat order\b)\b", re.IGNORECASE)
_POLICY_RE = re.compile(r"\b(policy|policies|how long|how do|what is|do you|can you|accepted|ship|warranty)\b", re.IGNORECASE)


def _last_human(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return str(msg.content)
    return ""


def _find_order_id(messages: list[BaseMessage]) -> str | None:
    """Find the most recently mentioned order id in the conversation.

    Only user/assistant messages count — system prompts contain few-shot
    examples with order ids that must never leak into tool arguments.
    """
    for msg in reversed(messages):
        if not isinstance(msg, (HumanMessage, AIMessage)):
            continue
        found = ORDER_ID_RE.findall(str(msg.content))
        if found:
            return found[-1].upper()
    return None


def classify_route(messages: list[BaseMessage]) -> RouteDecision:
    """Deterministic routing rules used by the mock provider."""
    text = _last_human(messages)

    if ORDER_ID_RE.search(text) or PRODUCT_ID_RE.search(text):
        return RouteDecision(route="tool", reason="Message references a specific order/product id.")

    if re.search(r"\b(policy|policies)\b", text, re.IGNORECASE):
        return RouteDecision(route="rag", reason="Question about store policy documentation.")

    if _RETURN_INTENT_RE.search(text):
        # "Can I return it / my order?" -> tool if an order is identifiable
        # from the message or conversation; otherwise it's a policy question.
        if _MY_ORDER_RE.search(text) or _find_order_id(messages):
            return RouteDecision(route="tool", reason="Return question about the customer's own order.")
        return RouteDecision(route="rag", reason="General return-policy question.")

    if _MY_ORDER_RE.search(text) and _STATUS_INTENT_RE.search(text):
        return RouteDecision(route="tool", reason="Status question about the customer's own order.")

    if _CAPABILITY_RE.search(text) or (_GREETING_RE.search(text) and len(text.split()) <= 6):
        return RouteDecision(route="direct", reason="Greeting or small talk.")

    return RouteDecision(route="rag", reason="General question answerable from documentation.")


# ---------------------------------------------------------------------------
# Mock generation helpers
# ---------------------------------------------------------------------------

def _mock_tool_calls(messages: list[BaseMessage]) -> list[dict]:
    """Decide which tool(s) to call, mimicking an LLM's tool selection."""
    text = _last_human(messages)
    calls: list[dict] = []

    product_ids = PRODUCT_ID_RE.findall(text)
    order_id = None
    found_in_text = ORDER_ID_RE.findall(text)
    if found_in_text:
        order_id = found_in_text[-1].upper()
    elif _MY_ORDER_RE.search(text) or _RETURN_INTENT_RE.search(text):
        order_id = _find_order_id(messages)  # resolve "it" from history

    wants_return = bool(_RETURN_INTENT_RE.search(text))
    wants_status = bool(_STATUS_INTENT_RE.search(text))

    if order_id:
        if wants_status or not wants_return:
            calls.append({"name": "lookup_order", "args": {"order_id": order_id}})
        if wants_return:
            calls.append({"name": "check_return_eligibility", "args": {"order_id": order_id}})
    for pid in product_ids:
        calls.append({"name": "get_product_details", "args": {"product_id": pid.upper()}})

    for i, call in enumerate(calls):
        call["id"] = f"call_{i}"
    return calls


def _summarize_order(result: dict) -> str:
    items = ", ".join(f"{i['name']} (x{i['quantity']})" for i in result.get("items", []))
    parts = [f"Your order {result['order_id']} is currently **{result['status']}**."]
    if result.get("tracking_number"):
        carrier = f" via {result['carrier']}" if result.get("carrier") else ""
        parts.append(f"Tracking number {result['tracking_number']}{carrier}.")
    if result.get("estimated_delivery") and result["status"] in ("processing", "shipped"):
        parts.append(f"Estimated delivery: {result['estimated_delivery']}.")
    if result.get("delivered_at"):
        parts.append(f"It was delivered on {result['delivered_at']}.")
    if items:
        parts.append(f"Items: {items}.")
    return " ".join(parts)


def _summarize_eligibility(result: dict) -> str:
    if not result.get("items"):
        return f"Order {result['order_id']}: {result['reason']}"
    lines = [f"Return check for order {result['order_id']} (delivered {result.get('delivered_at', 'n/a')}):"]
    for item in result["items"]:
        status = "eligible for return" if item["eligible"] else "not eligible"
        lines.append(f"- {item['name']}: {status} — {item['reason']}")
    if result.get("eligible"):
        lines.append(result.get("refund_info", ""))
    return "\n".join(filter(None, lines))


def _summarize_product(result: dict) -> str:
    specs = result.get("specs") or {}
    spec_text = "; ".join(f"{k.replace('_', ' ')}: {v}" for k, v in list(specs.items())[:4])
    stock = "in stock" if result.get("in_stock") else "out of stock"
    answer = (
        f"{result['name']} ({result['product_id']}) — ${result['price']:.2f}, {stock}. "
        f"{result.get('description', '')}"
    )
    if spec_text:
        answer += f" Key specs — {spec_text}."
    return answer


def _compose_tool_answer(messages: list[BaseMessage]) -> str:
    """Build the final answer from ToolMessage results of this turn."""
    summaries: list[str] = []
    # Collect trailing ToolMessages (results of the latest tool batch).
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                payload = json.loads(str(msg.content))
            except json.JSONDecodeError:
                payload = {"success": False, "error": str(msg.content)}
            if payload.get("success"):
                result = payload.get("result", {})
                if "items" in result and "eligible" in result:
                    summaries.append(_summarize_eligibility(result))
                elif "order_id" in result:
                    summaries.append(_summarize_order(result))
                elif "product_id" in result:
                    summaries.append(_summarize_product(result))
                else:
                    summaries.append(json.dumps(result))
            else:
                summaries.append(
                    f"I couldn't complete that lookup: {payload.get('error', 'unknown error')} "
                    "Please double-check the id (order ids look like ORD-1234) and try again."
                )
        elif isinstance(msg, AIMessage):
            break
    summaries.reverse()
    return "\n\n".join(summaries) if summaries else (
        "Could you share your order number (it looks like ORD-1234)? "
        "I'll look it up right away."
    )


_TITLE_RE = re.compile(r"^\[(?P<title>[^\]]+)\]", re.MULTILINE)


def _compose_rag_answer(system_text: str) -> str:
    """Answer grounded in the retrieved context embedded in the system prompt."""
    match = re.search(r"<retrieved_context>\n(.*?)\n</retrieved_context>", system_text, re.DOTALL)
    context = match.group(1).strip() if match else ""
    if not context or context == "(no relevant documents found)":
        return (
            "I'm sorry — I couldn't find anything about that in our documentation, so I "
            "don't want to guess. Please reach out to support@example-retail.com and a "
            "human agent will help you within 1 business day."
        )
    # Use the top two passages verbatim (they are policy text), cite titles.
    blocks = [b.strip() for b in context.split("\n---\n") if b.strip()]
    answer_parts: list[str] = []
    titles: list[str] = []
    for block in blocks[:2]:
        title_match = _TITLE_RE.match(block)
        body = block[title_match.end():].strip() if title_match else block
        if title_match:
            title = title_match.group("title")
            if title not in titles:
                titles.append(title)
        answer_parts.append(body[:600])
    citation = " ".join(f"[{t}]" for t in titles)
    return "Here's what our documentation says:\n\n" + "\n\n".join(answer_parts) + (
        f"\n\nSources: {citation}" if citation else ""
    )


def _compose_direct_answer(messages: list[BaseMessage]) -> str:
    text = _last_human(messages).lower()
    if "thank" in text:
        return "You're very welcome! Is there anything else I can help you with?"
    if _CAPABILITY_RE.search(text):
        return (
            "I can help you check order status and tracking, review return eligibility "
            "for your orders, and answer questions about our products, shipping, "
            "returns, warranty, and payment policies. What can I do for you?"
        )
    if "bye" in text or "goodbye" in text:
        return "Goodbye! Feel free to come back any time you need help with an order."
    return (
        "Hello! I'm your retail support assistant. I can track orders, check return "
        "eligibility, and answer product or policy questions. How can I help today?"
    )


class MockChatModel(BaseChatModel):
    """Deterministic rule-based chat model with tool-calling support."""

    @property
    def _llm_type(self) -> str:
        return "mock-retail-support"

    def bind_tools(self, tools: Any, **kwargs: Any):
        return self.bind(tools=tools, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any):
        if schema is RouteDecision:
            return RunnableLambda(lambda messages: classify_route(list(messages)))
        raise NotImplementedError(f"MockChatModel has no structured output for {schema}")

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        system_text = str(messages[0].content) if messages else ""
        tools_bound = bool(kwargs.get("tools"))

        if tools_bound:
            if isinstance(messages[-1], ToolMessage):
                message = AIMessage(content=_compose_tool_answer(messages))
            else:
                tool_calls = _mock_tool_calls(messages)
                if tool_calls:
                    message = AIMessage(content="", tool_calls=tool_calls)
                else:
                    message = AIMessage(content=_compose_tool_answer(messages))
        elif "<retrieved_context>" in system_text:
            message = AIMessage(content=_compose_rag_answer(system_text))
        else:
            message = AIMessage(content=_compose_direct_answer(messages))

        return ChatResult(generations=[ChatGeneration(message=message)])


def get_chat_model(settings: Settings | None = None) -> BaseChatModel:
    settings = settings or get_settings()
    if settings.resolved_llm_provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.chat_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    return MockChatModel()
