"""API tests: validation, auth, session isolation, tool endpoint."""

from tests.conftest import ALICE, BOB


class TestAuth:
    def test_missing_token_is_401(self, client):
        assert client.get("/api/v1/me").status_code == 401

    def test_invalid_token_is_401(self, client):
        response = client.get("/api/v1/me", headers={"Authorization": "Bearer nope"})
        assert response.status_code == 401

    def test_valid_token_resolves_identity(self, client):
        response = client.get("/api/v1/me", headers=ALICE)
        assert response.status_code == 200
        assert response.json()["email"] == "alice@example.com"


class TestHealth:
    def test_health_reports_providers(self, client):
        data = client.get("/api/v1/health").json()
        assert data["status"] == "ok"
        assert data["llm_provider"] == "mock"
        assert data["vector_backend"] == "local"
        assert data["vector_count"] > 0

    def test_ready(self, client):
        assert client.get("/api/v1/ready").json() == {"status": "ready"}


class TestSessions:
    def test_create_list_history_roundtrip(self, client):
        created = client.post("/api/v1/sessions", json={"title": "Test chat"}, headers=ALICE)
        assert created.status_code == 201
        session_id = created.json()["id"]

        listed = client.get("/api/v1/sessions", headers=ALICE).json()
        assert any(s["id"] == session_id for s in listed)

        history = client.get(f"/api/v1/sessions/{session_id}/history", headers=ALICE)
        assert history.status_code == 200
        assert history.json() == []

    def test_sessions_are_isolated_between_customers(self, client):
        session_id = client.post("/api/v1/sessions", json={}, headers=ALICE).json()["id"]

        # Bob cannot read Alice's history, chat into her session, or see it listed.
        assert (
            client.get(f"/api/v1/sessions/{session_id}/history", headers=BOB).status_code == 404
        )
        response = client.post(
            "/api/v1/chat", json={"message": "hi", "session_id": session_id}, headers=BOB
        )
        assert response.status_code == 404
        assert all(s["id"] != session_id for s in client.get("/api/v1/sessions", headers=BOB).json())

    def test_unknown_session_404(self, client):
        assert (
            client.get("/api/v1/sessions/does-not-exist/history", headers=ALICE).status_code == 404
        )


class TestChatValidation:
    def test_empty_message_rejected(self, client):
        response = client.post("/api/v1/chat", json={"message": ""}, headers=ALICE)
        assert response.status_code == 422

    def test_chat_requires_auth(self, client):
        assert client.post("/api/v1/chat", json={"message": "hi"}).status_code == 401

    def test_chat_persists_history(self, client):
        response = client.post("/api/v1/chat", json={"message": "Hello!"}, headers=ALICE)
        assert response.status_code == 200
        data = response.json()
        assert data["answer"]
        history = client.get(
            f"/api/v1/sessions/{data['session_id']}/history", headers=ALICE
        ).json()
        assert [m["role"] for m in history] == ["user", "assistant"]
        assert history[1]["route"] == data["route"]


class TestToolEndpoint:
    def test_lists_allowlisted_tools(self, client):
        names = {t["name"] for t in client.get("/api/v1/tools").json()}
        assert names == {"lookup_order", "check_return_eligibility", "get_product_details"}

    def test_invoke_valid_tool(self, client):
        response = client.post(
            "/api/v1/tools/invoke",
            json={"tool": "lookup_order", "arguments": {"order_id": "ORD-1002"}},
            headers=ALICE,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is True
        assert body["result"]["order_id"] == "ORD-1002"

    def test_unknown_tool_rejected(self, client):
        response = client.post(
            "/api/v1/tools/invoke", json={"tool": "os.system", "arguments": {}}, headers=ALICE
        )
        assert response.status_code == 422

    def test_bad_arguments_rejected(self, client):
        response = client.post(
            "/api/v1/tools/invoke",
            json={"tool": "lookup_order", "arguments": {"wrong_field": 1}},
            headers=ALICE,
        )
        assert response.status_code == 422

    def test_tool_respects_customer_scope(self, client):
        # ORD-1001 is Alice's; Bob gets a not-found failure, not her data.
        response = client.post(
            "/api/v1/tools/invoke",
            json={"tool": "lookup_order", "arguments": {"order_id": "ORD-1001"}},
            headers=BOB,
        )
        assert response.status_code == 200
        body = response.json()
        assert body["success"] is False
        assert body["result"] is None
