"""Evaluation harness runner.

Runs the full dataset through the real application stack (FastAPI +
LangGraph + services + vector store) in-process, against a throwaway
DB and vector index, then writes a JSON + Markdown report.

Usage (from the backend/ directory):

    python -m evals.run_eval                               # defaults
    python -m evals.run_eval --prompt-version v1 --label v1-baseline
    python -m evals.run_eval --top-k 2 --label topk2
    python -m evals.run_eval --chunk-size 400 --chunk-overlap 60 --label small-chunks
    python -m evals.run_eval --judge                       # add LLM-as-judge (needs OPENAI_API_KEY)
    python -m evals.run_eval --real-llm                    # use the OpenAI agent instead of the mock

Compare two runs:

    python -m evals.compare evals/results/<a>/report.json evals/results/<b>/report.json
"""

import argparse
import json
import os
import statistics
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

EVALS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVALS_DIR / "results"


def _bootstrap_environment(args) -> None:
    """Pin env config BEFORE importing any app module."""
    tmpdir = Path(tempfile.mkdtemp(prefix="retail-eval-"))
    env = {
        "DATABASE_URL": f"sqlite:///{(tmpdir / 'eval.db').as_posix()}",
        "LOCAL_VECTOR_STORE_PATH": str(tmpdir / "vectors.json"),
        "VECTOR_BACKEND": "local",
        "EMBEDDING_PROVIDER": "local",
        "LLM_PROVIDER": "openai" if args.real_llm else "mock",
        "PROMPT_VERSION": args.prompt_version,
        "TOP_K": str(args.top_k),
        "CHUNK_SIZE": str(args.chunk_size),
        "CHUNK_OVERLAP": str(args.chunk_overlap),
        "LOG_LEVEL": "WARNING",
    }
    if args.real_retrieval:
        env.pop("VECTOR_BACKEND")
        env.pop("EMBEDDING_PROVIDER")
    os.environ.update(env)


def _mean(values: list[float]) -> float | None:
    values = [v for v in values if v is not None]
    return round(statistics.mean(values), 4) if values else None


def evaluate_turn(
    turn_result: dict, expect: dict, top_k: int, doc_texts: dict[str, str] | None = None
) -> dict:
    """Apply all assertions/metrics configured for one turn."""
    from evals import metrics

    answer = turn_result["answer"]
    checks: dict[str, bool] = {}
    turn_metrics: dict[str, float | None] = {}

    if "route" in expect:
        checks["route"] = turn_result["route"] == expect["route"]

    if expect.get("no_tools"):
        checks["no_tools"] = not turn_result["tool_invocations"]
    if expect.get("no_sources"):
        checks["no_sources"] = not turn_result["sources"]

    if "tools" in expect:
        tool_check = metrics.check_tool_calls(
            expect["tools"],
            turn_result["tool_invocations"],
            exact_order=expect.get("tools_exact_order", False),
        )
        checks["tool_selection"] = tool_check["selection_ok"]
        checks["tool_args"] = tool_check["args_ok"]

    if "tool_success" in expect:
        invocations = turn_result["tool_invocations"]
        if expect["tool_success"]:
            checks["tool_success"] = bool(invocations) and all(t["success"] for t in invocations)
        else:
            checks["tool_success"] = bool(invocations) and any(not t["success"] for t in invocations)

    if "retrieval_docs" in expect:
        retrieved = metrics.ranked_docs(turn_result["sources"])
        expected_docs = expect["retrieval_docs"]
        turn_metrics["recall_at_k"] = metrics.recall_at_k(expected_docs, retrieved, top_k)
        turn_metrics["hit_at_k"] = metrics.hit_at_k(expected_docs, retrieved, top_k)
        turn_metrics["mrr"] = metrics.mrr(expected_docs, retrieved)
        checks["retrieval_hit"] = turn_metrics["hit_at_k"] == 1.0

    if "answer_contains_any" in expect:
        checks["answer_contains_any"] = any(
            needle.lower() in answer.lower() for needle in expect["answer_contains_any"]
        )
    if "answer_contains_all" in expect:
        checks["answer_contains_all"] = all(
            needle.lower() in answer.lower() for needle in expect["answer_contains_all"]
        )
    if "answer_not_contains" in expect:
        checks["answer_not_contains"] = not any(
            needle.lower() in answer.lower() for needle in expect["answer_not_contains"]
        )

    if expect.get("uncertainty"):
        checks["uncertainty_acknowledged"] = any(
            phrase in answer.lower()
            for phrase in ("couldn't find", "don't want to guess", "don't have", "not sure")
        )

    if expect.get("grounded") and turn_result["sources"]:
        # Faithfulness is measured against the full text of the cited
        # documents (API snippets are truncated for payload size).
        doc_texts = doc_texts or {}
        context_texts = [
            doc_texts.get(s["doc_id"], s["snippet"]) for s in turn_result["sources"]
        ]
        turn_metrics["groundedness"] = metrics.groundedness(answer, context_texts)
        titles = [s["title"] for s in turn_result["sources"]]
        checks["citation_present"] = metrics.citation_present(answer, titles) or "Sources:" in answer

    return {"checks": checks, "metrics": turn_metrics, "passed": all(checks.values())}


def run(args) -> Path:
    _bootstrap_environment(args)

    # Imports AFTER env pinning.
    sys.path.insert(0, str(EVALS_DIR.parent))
    from fastapi.testclient import TestClient

    from app.config import get_settings

    get_settings.cache_clear()
    from app.db.base import SessionLocal
    from app.db.seed import seed
    from app.rag.ingest import ingest

    db = SessionLocal()
    try:
        seed(db)
    finally:
        db.close()
    ingest_stats = ingest(clear=True)

    from app.main import app
    from app.rag.loader import load_documents

    doc_texts = {d.doc_id: d.text for d in load_documents(get_settings().docs_dir)}

    dataset = json.loads((EVALS_DIR / "dataset.json").read_text(encoding="utf-8"))
    cases = dataset["cases"]
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    judge = None
    if args.judge:
        if not os.environ.get("OPENAI_API_KEY"):
            print("WARNING: --judge requested but OPENAI_API_KEY is not set; skipping judges.")
        else:
            from evals.judges import LLMJudge

            judge = LLMJudge(get_settings().chat_model, os.environ["OPENAI_API_KEY"])

    client = TestClient(app)
    case_results = []
    for case in cases:
        headers = {"Authorization": f"Bearer {case['customer_token']}"}
        session_id = None
        turn_results = []
        for turn in case["turns"]:
            body = {"message": turn["query"]}
            if session_id:
                body["session_id"] = session_id
            response = client.post("/api/v1/chat", json=body, headers=headers)
            if response.status_code != 200:
                turn_results.append(
                    {
                        "query": turn["query"],
                        "error": f"HTTP {response.status_code}: {response.text[:200]}",
                        "evaluation": {"checks": {"http_ok": False}, "metrics": {}, "passed": False},
                    }
                )
                continue
            data = response.json()
            session_id = data["session_id"]
            evaluation = evaluate_turn(data, turn.get("expect", {}), args.top_k, doc_texts)

            if judge is not None:
                judge_scores = {"relevance": judge.relevance(turn["query"], data["answer"])}
                if data["sources"]:
                    judge_scores["groundedness"] = judge.groundedness(
                        data["answer"], [s["snippet"] for s in data["sources"]]
                    )
                judge_scores["tone"] = judge.tone(data["answer"])
                evaluation["judge"] = judge_scores

            turn_results.append(
                {
                    "query": turn["query"],
                    "route": data["route"],
                    "answer": data["answer"],
                    "sources": [
                        {"doc_id": s["doc_id"], "score": s["score"]} for s in data["sources"]
                    ],
                    "tools": [
                        {"name": t["name"], "arguments": t["arguments"], "success": t["success"]}
                        for t in data["tool_invocations"]
                    ],
                    "evaluation": evaluation,
                }
            )

        case_results.append(
            {
                "id": case["id"],
                "category": case["category"],
                "passed": all(t["evaluation"]["passed"] for t in turn_results),
                "turns": turn_results,
            }
        )

    report = _build_report(args, ingest_stats, case_results)
    out_dir = _write_report(report, args.label)
    _print_summary(report, out_dir)
    return out_dir


def _build_report(args, ingest_stats: dict, case_results: list[dict]) -> dict:
    all_turns = [t for c in case_results for t in c["turns"]]

    def metric_values(name: str) -> list[float]:
        return [
            t["evaluation"]["metrics"][name]
            for t in all_turns
            if name in t["evaluation"].get("metrics", {})
        ]

    def check_rate(name: str) -> float | None:
        values = [
            1.0 if t["evaluation"]["checks"][name] else 0.0
            for t in all_turns
            if name in t["evaluation"].get("checks", {})
        ]
        return _mean(values)

    judge_means = {}
    for judge_metric in ("relevance", "groundedness", "tone"):
        scores = [
            t["evaluation"]["judge"][judge_metric]["score"]
            for t in all_turns
            if "judge" in t["evaluation"] and judge_metric in t["evaluation"]["judge"]
            and t["evaluation"]["judge"][judge_metric]["score"] is not None
        ]
        if scores:
            judge_means[f"judge_{judge_metric}_mean"] = _mean(scores)

    aggregates = {
        "cases_total": len(case_results),
        "cases_passed": sum(1 for c in case_results if c["passed"]),
        "turns_total": len(all_turns),
        "turns_passed": sum(1 for t in all_turns if t["evaluation"]["passed"]),
        "routing_accuracy": check_rate("route"),
        "tool_selection_accuracy": check_rate("tool_selection"),
        "tool_args_accuracy": check_rate("tool_args"),
        "tool_outcome_accuracy": check_rate("tool_success"),
        "recall_at_k": _mean(metric_values("recall_at_k")),
        "hit_rate_at_k": _mean(metric_values("hit_at_k")),
        "mrr": _mean(metric_values("mrr")),
        "groundedness_mean": _mean(metric_values("groundedness")),
        "citation_rate": check_rate("citation_present"),
        "uncertainty_handling": check_rate("uncertainty_acknowledged"),
        **judge_means,
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "prompt_version": args.prompt_version,
            "top_k": args.top_k,
            "chunk_size": args.chunk_size,
            "chunk_overlap": args.chunk_overlap,
            "llm_provider": "openai" if args.real_llm else "mock",
            "judge": bool(args.judge and os.environ.get("OPENAI_API_KEY")),
            "ingest_stats": ingest_stats,
        },
        "aggregates": aggregates,
        "cases": case_results,
    }


def _write_report(report: dict, label: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    out_dir = RESULTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    agg = report["aggregates"]
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Configuration",
        "",
        "| Setting | Value |",
        "|---|---|",
    ]
    for key, value in report["config"].items():
        if key != "ingest_stats":
            lines.append(f"| {key} | {value} |")
    lines += [
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
    ]
    for key, value in agg.items():
        lines.append(f"| {key} | {value if value is not None else 'n/a'} |")

    failures = [c for c in report["cases"] if not c["passed"]]
    lines += ["", f"## Failures ({len(failures)})", ""]
    if not failures:
        lines.append("All cases passed.")
    for case in failures:
        lines.append(f"### {case['id']} ({case['category']})")
        for turn in case["turns"]:
            failed_checks = [
                name for name, ok in turn["evaluation"]["checks"].items() if not ok
            ]
            if failed_checks:
                lines.append(f"- Query: {turn['query']!r}")
                lines.append(f"  - Failed checks: {', '.join(failed_checks)}")
                lines.append(f"  - Answer: {turn.get('answer', turn.get('error', ''))[:300]!r}")
        lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return out_dir


def _print_summary(report: dict, out_dir: Path) -> None:
    agg = report["aggregates"]
    print(f"\n=== Evaluation complete: {agg['cases_passed']}/{agg['cases_total']} cases passed ===")
    for key in (
        "routing_accuracy", "tool_selection_accuracy", "tool_args_accuracy",
        "recall_at_k", "hit_rate_at_k", "mrr", "groundedness_mean",
        "citation_rate", "uncertainty_handling",
    ):
        print(f"  {key:26s} {agg[key] if agg[key] is not None else 'n/a'}")
    print(f"Report written to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the evaluation suite.")
    parser.add_argument("--prompt-version", default="v2", help="Prompt pack to evaluate (v1, v2)")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--label", default=None, help="Label for the results directory")
    parser.add_argument("--only", default=None, help="Comma-separated case ids to run")
    parser.add_argument("--judge", action="store_true", help="Enable LLM-as-judge scoring")
    parser.add_argument("--real-llm", action="store_true", help="Use OpenAI instead of the mock LLM")
    parser.add_argument(
        "--real-retrieval", action="store_true",
        help="Use configured (possibly real) embedding/vector providers",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
