# Resume-to-Code Traceability

Written for interview preparation: each resume claim mapped to the code that implements it, how to demo it, likely interview questions, and honest limitations.

Status legend: ✅ implemented & verified (tests/evals executed) · ⚠️ implemented, unverified against paid services.

---

## Bullet 1 — Agentic orchestration

> *"Built an agentic customer-support assistant using LangChain to orchestrate multi-step reasoning, deciding whether to answer directly, query a knowledge base, or invoke a tool (order lookup, return-policy check)."*

**Status: ✅**

### Where it lives

| Concern | Code |
|---|---|
| Graph topology (route → direct / retrieve+generate / tool loop) | `backend/app/agent/graph.py` (`AgentRuntime._build`) |
| Routing decision as structured output | `graph.py::_route` + `backend/app/agent/state.py` (`RouteDecision`) |
| Tool-call loop with iteration budget | `graph.py::_tool_agent`, `_tool_executor`, `_tools_edge` |
| Tool business logic (real DB queries) | `backend/app/services/order_service.py`, `returns_service.py`, `product_service.py` |
| Allowlist + validation + audit log | `backend/app/services/tool_registry.py` |
| LLM providers (OpenAI / deterministic mock) | `backend/app/agent/llm.py` |

### How it works

`StateGraph(AgentState)` with six nodes. State carries LangChain messages (merged via `add_messages`), the route decision, retrieved passages, and tool-invocation records. The **route** node calls `llm.with_structured_output(RouteDecision)` — a Pydantic schema constraining output to `direct|rag|tool` plus a reason. The **tool_agent** node binds OpenAI-style tool schemas (derived from the registry's Pydantic models) to the LLM; emitted tool calls are executed by a *custom* executor node — not LangGraph's prebuilt `ToolNode` — so every execution passes through customer authorization, argument validation, and audit logging, then returns `ToolMessage`s to the LLM until it answers (`MAX_TOOL_ITERATIONS` guard).

### Demo

Backend running → UI at `localhost:5173` → ask, in one session: "Hi!" (direct), "Where is my order ORD-1001?" (tool), "Can I return it?" (tool, pronoun resolution), "What's your return policy?" (RAG). The route badge and expandable tool-call/source panels make the workflow observable per message.

### Likely interview questions

- *Why LangGraph instead of a single ReAct agent?* Explicit nodes/edges make the control flow observable, testable, and cheap — direct/RAG paths never pay for tool schemas; routing is a classified decision I can measure (routing_accuracy = 1.0 on the eval set), not an emergent behavior.
- *Why a custom tool-executor node?* Cross-cutting concerns: customer scoping, Pydantic validation, audit logging into `tool_execution_logs`, and uniform error envelopes (`success/result/error`) the LLM can relay conversationally.
- *How do multi-step tool chains happen?* The tool_agent↔tool_executor cycle: the model can emit several calls per turn (e.g. `lookup_order` + `check_return_eligibility` for "what was in ORD-1004 and can I return it?") or sequential rounds until done.

### Limitations

The mock LLM routes with deterministic rules (regex/keywords), so mock-mode "reasoning" is shallow by design; with `OPENAI_API_KEY` the identical graph runs on `ChatOpenAI`. Real-LLM behavior is implemented but unverified without credentials (⚠️).

---

## Bullet 2 — RAG pipeline

> *"Implemented a RAG pipeline chunking and embedding product/FAQ docs into Pinecone, retrieving top-k context at query time to ground responses and reduce hallucination."*

**Status: ✅ (local vector store verified; Pinecone wrapper ⚠️ unverified without key)**

### Where it lives

| Concern | Code |
|---|---|
| Corpus (6 Markdown docs: catalog, shipping, returns, warranty, payments, FAQ) | `backend/app/data/docs/` |
| Loading + frontmatter + normalization | `backend/app/rag/loader.py` |
| Recursive, Markdown-aware chunking w/ overlap + deterministic IDs | `backend/app/rag/chunking.py` |
| Embeddings (OpenAI `text-embedding-3-*` / local hashed fallback) | `backend/app/rag/embeddings.py` |
| Vector stores (Pinecone serverless / local JSON cosine store) | `backend/app/rag/vectorstore.py` |
| Query-time retrieval with scores + metadata | `backend/app/rag/retriever.py` |
| CLI ingestion | `python -m app.rag.ingest [--clear --chunk-size --chunk-overlap]` |

### How it works

Ingestion: load → normalize → `RecursiveCharacterTextSplitter` (separators tuned for Markdown headings; configurable size/overlap) → embed → upsert with metadata `{doc_id, source, title, category, chunk_index, text}`. Chunk IDs are `doc_id::index::sha256(content)[:8]`, so re-ingestion is **duplicate-safe** (unchanged chunks overwrite themselves). Retrieval: embed query → top-k cosine search → threshold filter (`MIN_RETRIEVAL_SCORE`) → passages with scores returned to the agent *and* exposed in the API response for debugging/eval.

Hallucination reduction is layered: grounding instructions + citation requirement in the RAG prompt, a deterministic refusal path when no passage clears the threshold ("I couldn't find…"), and measured groundedness/citation metrics in the eval harness.

### Demo

```bash
cd backend
python -m app.rag.ingest --clear        # prints docs/chunks/vector counts
```
Then ask "How long does standard shipping take?" — answer quotes the shipping table and the UI shows source chunks with scores. Ask "What is your policy on quantum teleportation of pets?" — explicit uncertainty, zero sources.

### Likely interview questions

- *How did you pick chunk size?* Measured, not guessed: the eval harness at 800/120 vs 400/60 shows small chunks regress MRR 1.0→0.88, groundedness 0.90→0.70, and break 6 cases (`evals/results/compare_chunks800_vs_400.md`).
- *Why can you swap Pinecone for a local store?* Both implement the same 4-method protocol (`upsert/query/count/clear`); the retriever is backend-agnostic. Pinecone is config (`PINECONE_API_KEY`), including index auto-creation.
- *How do you prevent duplicate vectors on re-ingestion?* Deterministic content-addressed IDs; tested in `tests/test_chunking.py::test_reingest_is_duplicate_safe`.

### Limitations

The offline fallback embedding is lexical (stemmed hashed bag-of-words) — great for this corpus (MRR 1.0), not semantically general. `PineconeVectorStore` is implemented against the current SDK but has not been exercised against a live index in this environment (no key); the optional integration test `tests/test_integration_real_services.py::test_pinecone_upsert_and_query` covers it when credentials exist.

---

## Bullet 3 — FastAPI service + evaluation harness

> *"Exposed the agent as a RESTful FastAPI service with endpoints for chat sessions, tool invocation, and conversation history; built an evaluation harness to catch regressions in retrieval accuracy and response relevance."*

**Status: ✅**

### Where it lives

| Concern | Code |
|---|---|
| Endpoints (`/api/v1`: chat, sessions, history, tools, health/ready, me) | `backend/app/api/v1/router.py` |
| Pydantic contracts | `backend/app/schemas/api.py` |
| Auth + DI | `backend/app/api/deps.py` |
| Correlation IDs + access logs / centralized errors | `backend/app/api/middleware.py`, `errors.py` |
| Persistence (sessions/messages survive restarts) | `backend/app/services/session_service.py`, `app/db/models.py` |
| Eval dataset (38 cases, 10 categories) | `backend/evals/dataset.json` |
| Metrics: Recall@k, Hit@k, MRR, groundedness, citation, tool/routing accuracy | `backend/evals/metrics.py` |
| Runner + reports (JSON/Markdown) + config sweeps | `backend/evals/run_eval.py` |
| Run comparison (improved/regressed/still-failing) | `backend/evals/compare.py` |
| LLM-as-judge rubrics (optional) | `backend/evals/judges.py` |

### Demo

Swagger UI at `http://localhost:8000/docs`; scripted calls in `examples/api-requests.http`. One-command regression suite: `python -m pytest` (51 tests, 2s, offline). One-command eval: `python -m evals.run_eval --label demo`.

### Verified results

- **Tests:** 51 passed, 0 failed (offline deterministic providers), executed 2026-11-07.
- **Evals:** 38/38 cases; routing/tool-selection/tool-args accuracy 1.0; Recall@4 / Hit@4 / MRR 1.0; groundedness 0.90; citation rate 1.0; uncertainty handling 1.0. Regression sensitivity demonstrated by the chunking sweep (32/38 at 400/60).

### Likely interview questions

- *How does the harness catch regressions?* Deterministic assertions per turn (route, tool name/args/outcome, retrieval hits, answer keywords, uncertainty) + ranked-retrieval metrics against labeled relevant docs; `evals.compare` diffs two runs and names fixed/regressed cases.
- *Session isolation?* Every session/order query filters by the authenticated customer id; cross-customer access yields the same 404 as nonexistence. Tested at service, API, and agent levels.

### Limitations

LLM-as-judge and `--real-llm` eval paths require an OpenAI key (⚠️ implemented, unverified here). The eval client runs in-process (TestClient) — fine for logic regressions, not a latency/load benchmark.

---

## Bullet 4 (additional) — Prompt engineering & iteration

> *Modular prompt management: system prompts, few-shot examples, structured output, anti-hallucination and injection defenses, versioned templates compared experimentally.*

**Status: ✅ framework & versions implemented; cross-version behavioral deltas require a real LLM (⚠️)**

### Where it lives

`backend/app/agent/prompts/` — `base.py` (PromptPack), `v1.py` (terse baseline), `v2.py` (few-shot routing examples, grounding + "say you don't know" rules, `[Title]` citation format, injection defense over `<retrieved_context>`, tone guidance), `registry.py` (lookup). Selected via `PROMPT_VERSION` env or `--prompt-version` eval flag. Structured outputs: `RouteDecision` (routing) and per-tool Pydantic argument schemas.

### Demo & results

`python -m evals.run_eval --prompt-version v1 --label v1` vs `--prompt-version v2`. Recorded result: **identical metrics in mock mode** (the deterministic mock ignores prompt text) — an honest harness: it measures the pipeline, and the per-version comparison becomes meaningful with `--real-llm`. The v1→v2 design rationale (what each added instruction is supposed to move: tool-selection accuracy, groundedness, citation/format compliance, tone) is documented in `v2.py`'s docstring and measurable per-metric in the report.

### Interview talking point

Prompt iteration is treated as an experiment: fixed dataset, one variable (prompt version), metrics per dimension, comparison report. That's the methodology story even where the offline run can't show a delta.

---

## Bullet 5 (additional) — Microservice-ready architecture

> *Independently deployable FastAPI service suitable for integration with a larger retail/POS platform.*

**Status: ✅**

Versioned endpoints (`/api/v1`), Pydantic contracts at the boundary, DI for DB/retriever/settings (`api/deps.py`), clear service boundaries (agent / rag / services / db / api are separately importable layers), request+correlation IDs (`X-Request-ID` honored and returned), centralized exception handling, `/health` + `/ready` probes, env-based config (`app/config.py`), containerization (`backend/Dockerfile`, `docker-compose.yml` with healthcheck-gated frontend), OpenAPI auto-docs. A POS platform would integrate by calling the same REST surface with per-customer tokens from its own IdP.

---

## Five demo questions for interviews

Run as Alice (`demo-alice`) in one session, in order:

1. **"Where is my order ORD-1001?"** → tool route, live DB lookup, tracking + items.
2. **"Can I return it?"** → conversation memory resolves "it"; policy-based refusal (in transit).
3. **"What's your return policy for electronics?"** → RAG with citation of the Returns & Refunds Policy (15-day window).
4. **"What was in order ORD-1004 and can I return it?"** *(switch to Bob)* → two tools in one turn; mixed per-item eligibility (headphones yes, gift cards never).
5. **"What is your policy on quantum teleportation of pets?"** → below-threshold retrieval → explicit "I don't know" instead of hallucination.

Bonus security demo: as Bob, ask "Where is my order ORD-1001?" (Alice's) → tool fails with not-found; no data leaks.
