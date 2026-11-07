"""Deterministic evaluation metrics.

Retrieval metrics operate at the *document* level: a turn's expectation
lists relevant doc ids, and the ranked retrieved passages are collapsed
to a ranked list of distinct documents (best rank wins).
"""

import re

_WORD_RE = re.compile(r"[a-z0-9$%.\-–]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have how i if in is it its of on or "
    "that the this to was what when where which who will with you your here's our "
    "says documentation sources".split()
)


def ranked_docs(sources: list[dict]) -> list[str]:
    """Collapse ranked passages to ranked distinct doc ids."""
    seen: list[str] = []
    for source in sources:  # sources arrive sorted by score desc
        doc_id = source["doc_id"]
        if doc_id not in seen:
            seen.append(doc_id)
    return seen


def recall_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 1.0
    top = set(retrieved[:k])
    return sum(1 for d in expected if d in top) / len(expected)


def hit_at_k(expected: list[str], retrieved: list[str], k: int) -> float:
    if not expected:
        return 1.0
    return 1.0 if any(d in retrieved[:k] for d in expected) else 0.0


def mrr(expected: list[str], retrieved: list[str]) -> float:
    for rank, doc in enumerate(retrieved, start=1):
        if doc in expected:
            return 1.0 / rank
    return 0.0


def groundedness(answer: str, context_texts: list[str]) -> float:
    """Heuristic faithfulness: fraction of the answer's content tokens
    that literally appear in the retrieved context. Exact-match token
    overlap is crude but deterministic; the optional LLM judge provides
    a semantic version of the same measurement."""
    answer_tokens = [t for t in _WORD_RE.findall(answer.lower()) if t not in _STOP and len(t) > 2]
    if not answer_tokens:
        return 0.0
    context = " ".join(context_texts).lower()
    context_tokens = set(_WORD_RE.findall(context))
    hits = sum(1 for t in answer_tokens if t in context_tokens)
    return hits / len(answer_tokens)


def citation_present(answer: str, source_titles: list[str]) -> bool:
    """Format compliance for RAG answers: cites at least one source title."""
    return any(f"[{title}]" in answer for title in source_titles)


def check_tool_calls(
    expected: list[dict],
    actual: list[dict],
    *,
    exact_order: bool = False,
) -> dict:
    """Compare expected tool calls against actual invocations."""
    actual_names = [a["name"] for a in actual]
    expected_names = [e["name"] for e in expected]

    if exact_order:
        selection_ok = actual_names == expected_names
    else:
        selection_ok = all(name in actual_names for name in expected_names)

    args_ok = True
    for exp in expected:
        exp_args = exp.get("args")
        if exp_args is None:
            continue
        matching = [
            a for a in actual
            if a["name"] == exp["name"]
            and all(a["arguments"].get(k) == v for k, v in exp_args.items())
        ]
        if not matching:
            args_ok = False
    return {"selection_ok": selection_ok, "args_ok": args_ok}
