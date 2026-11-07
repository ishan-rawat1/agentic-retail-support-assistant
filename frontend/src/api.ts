import type { ChatResponse, HistoryMessage, SessionInfo } from "./types";

const BASE = "/api/v1";

async function request<T>(path: string, token: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
      ...(init?.headers ?? {}),
    },
  });
  if (!response.ok) {
    let detail = `HTTP ${response.status}`;
    try {
      const body = await response.json();
      if (body?.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch {
      /* keep default detail */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export const api = {
  sendMessage(token: string, message: string, sessionId?: string | null): Promise<ChatResponse> {
    return request<ChatResponse>("/chat", token, {
      method: "POST",
      body: JSON.stringify({ message, session_id: sessionId ?? undefined }),
    });
  },
  listSessions(token: string): Promise<SessionInfo[]> {
    return request<SessionInfo[]>("/sessions", token);
  },
  getHistory(token: string, sessionId: string): Promise<HistoryMessage[]> {
    return request<HistoryMessage[]>(`/sessions/${sessionId}/history`, token);
  },
};
