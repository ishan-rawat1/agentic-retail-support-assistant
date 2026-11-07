# Evaluation comparison

- **A**: prompt=v2 top_k=4 chunks=800/120 llm=mock (2026-11-07T21:00:39.555458+00:00)
- **B**: prompt=v2 top_k=2 chunks=800/120 llm=mock (2026-11-07T21:00:50.085819+00:00)

## Aggregate metrics

| Metric | A | B | Delta (B-A) | Verdict |
|---|---|---|---|---|
| routing_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_selection_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_args_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| tool_outcome_accuracy | 1.0 | 1.0 | +0.0 | unchanged |
| recall_at_k | 1.0 | 0.9615 | -0.0385 | regressed |
| hit_rate_at_k | 1.0 | 1.0 | +0.0 | unchanged |
| mrr | 1.0 | 1.0 | +0.0 | unchanged |
| groundedness_mean | 0.8967 | 0.8967 | +0.0 | unchanged |
| citation_rate | 1.0 | 1.0 | +0.0 | unchanged |
| uncertainty_handling | 1.0 | 1.0 | +0.0 | unchanged |

## Case-level changes

- Fixed in B (0): none
- Regressed in B (0): none
- Still failing in B (0): none