import { useState } from "react";
import type { SourceRef, ToolInvocation, UiMessage } from "../types";

const ROUTE_LABELS: Record<string, string> = {
  direct: "Direct answer",
  rag: "Knowledge base",
  tool: "Order tools",
};

function Sources({ sources }: { sources: SourceRef[] }) {
  const [open, setOpen] = useState(false);
  if (sources.length === 0) return null;
  return (
    <div className="sources">
      <button className="disclosure" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} {sources.length} source{sources.length > 1 ? "s" : ""}
      </button>
      {open && (
        <ul>
          {sources.map((s, i) => (
            <li key={i}>
              <strong>{s.title}</strong>{" "}
              <span className="meta">
                ({s.source} · chunk {s.chunk_index} · score {s.score.toFixed(3)})
              </span>
              <p>{s.snippet}…</p>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function ToolCalls({ calls }: { calls: ToolInvocation[] }) {
  const [open, setOpen] = useState(false);
  if (calls.length === 0) return null;
  return (
    <div className="tool-calls">
      <button className="disclosure" onClick={() => setOpen(!open)}>
        {open ? "▾" : "▸"} {calls.length} tool call{calls.length > 1 ? "s" : ""}
      </button>
      {open && (
        <ul>
          {calls.map((t, i) => (
            <li key={i}>
              <code>
                {t.name}({JSON.stringify(t.arguments)})
              </code>{" "}
              <span className={t.success ? "ok" : "fail"}>{t.success ? "✓ ok" : "✗ failed"}</span>
              {t.error && <p className="meta">{t.error}</p>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export function MessageBubble({ message }: { message: UiMessage }) {
  if (message.pending) {
    return (
      <div className="bubble assistant pending">
        <span className="dot" />
        <span className="dot" />
        <span className="dot" />
      </div>
    );
  }
  return (
    <div className={`bubble ${message.role}`}>
      {message.role === "assistant" && message.route && (
        <span className={`route-badge route-${message.route}`}>
          {ROUTE_LABELS[message.route] ?? message.route}
        </span>
      )}
      <div className="content">{message.content}</div>
      {message.role === "assistant" && message.sources && <Sources sources={message.sources} />}
      {message.role === "assistant" && message.toolCalls && (
        <ToolCalls calls={message.toolCalls} />
      )}
    </div>
  );
}
