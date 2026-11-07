# Evaluation comparison

- **A**: prompt=v2 top_k=4 chunks=800/120 llm=mock (2026-11-07T21:00:39.555458+00:00)
- **B**: prompt=v2 top_k=4 chunks=400/60 llm=mock (2026-11-07T21:00:56.519850+00:00)

## Aggregate metrics

| Metric | A | B | Delta (B-A) | Verdict |
|---|---|---|---|---|
| routing_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_selection_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_args_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_outcome_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| recall_at_k | 1.0 | 1.0 | +0.0 | unchanged |
| hit_rate_at_k | 1.0 | 1.0 | +0.0 | unchanged |
| mrr | 1.0 | 0.8846 | -0.1154 | regressed |
| groundedness_mean | 0.8967 | 0.7029 | -0.1938 | regressed |
| citation_rate | 1.0 | 1.0 | +0.0 | unchanged |
| uncertainty_handling | 1.0 | 0.5 | -0.5 | regressed |

## Case-level changes

- Fixed in B (0): none
- Regressed in B (6): CTX-02, RAG-01, RAG-04, RAG-05, RAG-08, RAG-09
- Still failing in B (6): CTX-02, RAG-01, RAG-04, RAG-05, RAG-08, RAG-09