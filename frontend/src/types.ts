export interface SourceRef {
  doc_id: string;
  title: string;
  source: string;
  chunk_index: number;
  score: number;
  snippet: string;
}

export interface ToolInvocation {
  name: string;
  arguments: Record<string, unknown>;
  success: boolean;
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  route: string;
  route_reason: string;
  sources: SourceRef[];
  tool_invocations: ToolInvocation[];
}

export interface SessionInfo {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface HistoryMessage {
  id: number;
  role: "user" | "assistant";
  content: string;
  route: string | null;
  sources: SourceRef[] | null;
  tool_calls: ToolInvocation[] | null;
  created_at: string;
}

export interface UiMessage {
  role: "user" | "assistant";
  content: string;
  route?: string | null;
  sources?: SourceRef[] | null;
  toolCalls?: ToolInvocation[] | null;
  pending?: boolean;
}

export interface DemoCustomer {
  token: string;
  name: string;
  label: string;
  hint: string;
}

export const DEMO_CUSTOMERS: DemoCustomer[] = [
  { token: "demo-alice", name: "Alice Johnson", label: "Alice", hint: "Orders ORD-1001 (shipped), ORD-1002 (returnable), ORD-1003 (window expired)" },
  { token: "demo-bob", name: "Bob Martinez", label: "Bob", hint: "Orders ORD-1004 (electronics + gift card), ORD-1005 (processing)" },
  { token: "demo-carol", name: "Carol Chen", label: "Carol", hint: "Orders ORD-1006 (final sale / expired), ORD-1007 (cancelled)" },
];

export const EXAMPLE_QUESTIONS = [
  "Where is my order ORD-1001?",
  "Can I return it?",
  "What is your return policy for electronics?",
  "How long does standard shipping take?",
  "Do the Aurora headphones have noise cancellation?",
  "What was in order ORD-1004 and can I return it?",
  "What payment methods do you accept?",
];
