"""Agent workflow tests: routing, RAG grounding, tool loops, memory."""

from tests.conftest import ALICE, BOB


def chat(client, message, headers=ALICE, session_id=None):
    body = {"message": message}
    if session_id:
        body["session_id"] = session_id
    response = client.post("/api/v1/chat", json=body, headers=headers)
    assert response.status_code == 200, response.text
    return response.json()


class TestRouting:
    def test_greeting_routes_direct(self, client):
        data = chat(client, "Hi there!")
        assert data["route"] == "direct"
        assert data["sources"] == []
        assert data["tool_invocations"] == []

    def test_policy_question_routes_rag(self, client):
        data = chat(client, "What is your return policy?")
        assert data["route"] == "rag"
        assert data["sources"], "RAG answers must carry sources"

    def test_order_question_routes_tool(self, client):
        data = chat(client, "Where is my order ORD-1001?")
        assert data["route"] == "tool"
        assert data["tool_invocations"]


class TestRagAnswers:
    def test_answer_is_grounded_with_citations(self, client):
        data = chat(client, "How long does standard shipping take?")
        assert data["route"] == "rag"
        assert "5–7 business days" in data["answer"] or "5-7 business days" in data["answer"]
        assert any(s["doc_id"] == "shipping-policy" for s in data["sources"])
        # Scores exposed for debugging/evaluation.
        assert all(0.0 <= s["score"] <= 1.0 for s in data["sources"])

    def test_insufficient_context_acknowledged(self, client):
        data = chat(client, "What is your policy on quantum teleportation of pets?")
        assert data["route"] == "rag"
        assert data["sources"] == []
        assert "couldn't find" in data["answer"]


class TestToolUse:
    def test_order_lookup_returns_real_data(self, client):
        data = chat(client, "Where is my order ORD-1001?")
        invocation = data["tool_invocations"][0]
        assert invocation["name"] == "lookup_order"
        assert invocation["arguments"] == {"order_id": "ORD-1001"}
        assert invocation["success"] is True
        assert "TRK-84512093" in data["answer"]

    def test_invalid_order_id_handled_gracefully(self, client):
        data = chat(client, "Where is my order ORD-9999?")
        invocation = data["tool_invocations"][0]
        assert invocation["success"] is False
        assert "ORD-9999" not in str(invocation.get("result"))
        assert "couldn't" in data["answer"].lower() or "no order" in data["answer"].lower()

    def test_multi_tool_single_turn(self, client):
        data = chat(client, "What was in order ORD-1004 and can I return it?", headers=BOB)
        names = [t["name"] for t in data["tool_invocations"]]
        assert "lookup_order" in names
        assert "check_return_eligibility" in names

    def test_return_eligibility_is_policy_based(self, client):
        data = chat(client, "Can I return my order ORD-1002?")
        invocation = data["tool_invocations"][-1]
        assert invocation["name"] == "check_return_eligibility"
        assert invocation["result"]["eligible"] is True
        assert "eligible" in data["answer"]

    def test_cross_customer_access_blocked_through_agent(self, client):
        # Bob asks about Alice's order through the chat agent.
        data = chat(client, "Where is my order ORD-1002?", headers=BOB)
        invocation = data["tool_invocations"][0]
        assert invocation["success"] is False
        assert invocation["result"] is None
        # No leakage of Alice's order contents.
        assert "TrailBlaze" not in data["answer"]


class TestConversationMemory:
    def test_pronoun_resolves_to_previous_order(self, client):
        first = chat(client, "Where is my order ORD-1002?")
        second = chat(client, "Can I return it?", session_id=first["session_id"])
        assert second["route"] == "tool"
        invocation = second["tool_invocations"][-1]
        assert invocation["name"] == "check_return_eligibility"
        assert invocation["arguments"] == {"order_id": "ORD-1002"}
        assert invocation["result"]["eligible"] is True

    def test_memory_is_session_scoped(self, client):
        # Mention ORD-1002 in session A; a NEW session must not inherit it.
        session_a = chat(client, "Where is my order ORD-1002?")["session_id"]
        fresh = chat(client, "Can I return it?")
        assert fresh["session_id"] != session_a
        # With no order in this session's history, the agent must ask for
        # the order number rather than guessing one.
        assert not any(
            t["arguments"].get("order_id") == "ORD-1002"
            for t in fresh["tool_invocations"]
        )
