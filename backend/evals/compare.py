"""Compare two evaluation reports (e.g. prompt v1 vs v2, top-k 2 vs 4).

Usage:

    python -m evals.compare evals/results/<runA>/report.json evals/results/<runB>/report.json
"""

import argparse
import json
from pathlib import Path


def load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compare(report_a: dict, report_b: dict) -> str:
    label_a = _label(report_a)
    label_b = _label(report_b)

    lines = [
        "# Evaluation comparison",
        "",
        f"- **A**: {label_a}",
        f"- **B**: {label_b}",
        "",
        "## Aggregate metrics",
        "",
        f"| Metric | A | B | Delta (B-A) | Verdict |",
        "|---|---|---|---|---|",
    ]

    keys = [k for k in report_a["aggregates"] if not k.startswith("cases") and not k.startswith("turns")]
    for key in keys:
        a = report_a["aggregates"].get(key)
        b = report_b["aggregates"].get(key)
        if a is None and b is None:
            continue
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            delta = round(b - a, 4)
            verdict = "improved" if delta > 0 else ("regressed" if delta < 0 else "unchanged")
            lines.append(f"| {key} | {a} | {b} | {delta:+} | {verdict} |")
        else:
            lines.append(f"| {key} | {a} | {b} |  |  |")

    status_a = {c["id"]: c["passed"] for c in report_a["cases"]}
    status_b = {c["id"]: c["passed"] for c in report_b["cases"]}
    fixed = sorted(cid for cid in status_a if not status_a[cid] and status_b.get(cid))
    broke = sorted(cid for cid in status_a if status_a[cid] and status_b.get(cid) is False)

    lines += ["", "## Case-level changes", ""]
    lines.append(f"- Fixed in B ({len(fixed)}): {', '.join(fixed) or 'none'}")
    lines.append(f"- Regressed in B ({len(broke)}): {', '.join(broke) or 'none'}")

    failures_b = sorted(cid for cid, ok in status_b.items() if not ok)
    lines.append(f"- Still failing in B ({len(failures_b)}): {', '.join(failures_b) or 'none'}")
    return "\n".join(lines)


def _label(report: dict) -> str:
    config = report["config"]
    return (
        f"prompt={config['prompt_version']} top_k={config['top_k']} "
        f"chunks={config['chunk_size']}/{config['chunk_overlap']} llm={config['llm_provider']} "
        f"({report['generated_at']})"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report_a")
    parser.add_argument("report_b")
    parser.add_argument("--out", default=None, help="Optional path to write the markdown diff")
    args = parser.parse_args()

    markdown = compare(load(args.report_a), load(args.report_b))
    print(markdown)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
