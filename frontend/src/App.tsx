import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "./api";
import { MessageBubble } from "./components/MessageBubble";
import {
  DEMO_CUSTOMERS,
  EXAMPLE_QUESTIONS,
  type SessionInfo,
  type UiMessage,
} from "./types";

export default function App() {
  const [customerToken, setCustomerToken] = useState(DEMO_CUSTOMERS[0].token);
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<UiMessage[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  const customer = DEMO_CUSTOMERS.find((c) => c.token === customerToken)!;

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions(customerToken));
    } catch (e) {
      setError((e as Error).message);
    }
  }, [customerToken]);

  useEffect(() => {
    setSessionId(null);
    setMessages([]);
    setError(null);
    void refreshSessions();
  }, [refreshSessions]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const openSession = async (id: string) => {
    setError(null);
    setSessionId(id);
    try {
      const history = await api.getHistory(customerToken, id);
      setMessages(
        history.map((m) => ({
          role: m.role,
          content: m.content,
          route: m.route,
          sources: m.sources,
          toolCalls: m.tool_calls,
        })),
      );
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const newChat = () => {
    setSessionId(null);
    setMessages([]);
    setError(null);
  };

  const send = async (text: string) => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    setError(null);
    setBusy(true);
    setInput("");
    setMessages((prev) => [
      ...prev,
      { role: "user", content: trimmed },
      { role: "assistant", content: "", pending: true },
    ]);
    try {
      const response = await api.sendMessage(customerToken, trimmed, sessionId);
      setSessionId(response.session_id);
      setMessages((prev) => [
        ...prev.slice(0, -1),
        {
          role: "assistant",
          content: response.answer,
          route: response.route,
          sources: response.sources,
          toolCalls: response.tool_invocations,
        },
      ]);
      void refreshSessions();
    } catch (e) {
      setMessages((prev) => prev.slice(0, -1));
      setError((e as Error).message);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h1>
          Retail<span>Support</span>
        </h1>

        <label className="field-label">Demo customer</label>
        <select
          value={customerToken}
          onChange={(e) => setCustomerToken(e.target.value)}
          className="customer-select"
        >
          {DEMO_CUSTOMERS.map((c) => (
            <option key={c.token} value={c.token}>
              {c.name}
            </option>
          ))}
        </select>
        <p className="customer-hint">{customer.hint}</p>

        <button className="new-chat" onClick={newChat}>
          + New conversation
        </button>

        <div className="session-list">
          {sessions.map((s) => (
            <button
              key={s.id}
              className={`session-item ${s.id === sessionId ? "active" : ""}`}
              onClick={() => void openSession(s.id)}
              title={s.title}
            >
              {s.title}
            </button>
          ))}
          {sessions.length === 0 && <p className="empty-note">No conversations yet.</p>}
        </div>
      </aside>

      <main className="chat">
        <div className="messages">
          {messages.length === 0 && (
            <div className="welcome">
              <h2>How can we help, {customer.label}?</h2>
              <p>
                Ask about orders, returns, shipping, products, or policies. The agent decides
                whether to answer directly, search the knowledge base, or call order tools.
              </p>
              <div className="examples">
                {EXAMPLE_QUESTIONS.map((q) => (
                  <button key={q} onClick={() => void send(q)} disabled={busy}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
          {messages.map((m, i) => (
            <MessageBubble key={i} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>

        {error && <div className="error-bar">⚠ {error}</div>}

        <form
          className="composer"
          onSubmit={(e) => {
            e.preventDefault();
            void send(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message… e.g. Where is my order ORD-1001?"
            disabled={busy}
          />
          <button type="submit" disabled={busy || !input.trim()}>
            {busy ? "Thinking…" : "Send"}
          </button>
        </form>
      </main>
    </div>
  );
}
