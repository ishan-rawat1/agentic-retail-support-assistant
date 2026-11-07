# Evaluation Report

Generated: 2026-11-07T21:00:56.519850+00:00

## Configuration

| Setting | Value |
|---|---|
| prompt_version | v2 |
| top_k | 4 |
| chunk_size | 400 |
| chunk_overlap | 60 |
| llm_provider | mock |
| judge | False |

## Aggregate metrics

| Metric | Value |
|---|---|
| cases_total | 38 |
| cases_passed | 32 |
| turns_total | 42 |
| turns_passed | 36 |
| routing_accuracy | 1.0 |
| tool_selection_accuracy | 1.0 |
| tool_args_accuracy | 1.0 |
| tool_outcome_accuracy | 1.0 |
| recall_at_k | 1.0 |
| hit_rate_at_k | 1.0 |
| mrr | 0.8846 |
| groundedness_mean | 0.7029 |
| citation_rate | 1.0 |
| uncertainty_handling | 0.5 |

## Failures (6)

### RAG-01 (shipping_rag)
- Query: 'How long does standard shipping take?'
  - Failed checks: answer_contains_any
  - Answer: "Here's what our documentation says:\n\n(source: warranty-policy.md, chunk 1, score 0.290)\n## Coverage by category\n\n(source: shipping-policy.md, chunk 0, score 0.261)\n# Shipping & Delivery Policy\n\nSources: [Warranty Policy] [Shipping & Delivery Policy]"

### RAG-04 (returns_rag)
- Query: 'What is your return policy?'
  - Failed checks: answer_contains_any
  - Answer: "Here's what our documentation says:\n\n(source: returns-policy.md, chunk 0, score 0.568)\n# Returns & Refunds Policy\n\n(source: returns-policy.md, chunk 1, score 0.523)\n## Return windows\n\nSources: [Returns & Refunds Policy]"

### RAG-05 (returns_rag)
- Query: "What's the return policy for electronics?"
  - Failed checks: answer_contains_any
  - Answer: "Here's what our documentation says:\n\n(source: returns-policy.md, chunk 0, score 0.344)\n# Returns & Refunds Policy\n\n(source: returns-policy.md, chunk 1, score 0.316)\n## Return windows\n\nSources: [Returns & Refunds Policy]"

### RAG-08 (product_rag)
- Query: 'How long is the battery life on the Aurora wireless headphones?'
  - Failed checks: answer_contains_any
  - Answer: "Here's what our documentation says:\n\n(source: product-catalog.md, chunk 1, score 0.441)\n## Aurora Wireless Headphones (PROD-101) — $129.99\n\n(source: warranty-policy.md, chunk 2, score 0.218)\n- **Electronics** (Aurora Wireless Headphones, LumenGlow Smart Lamp): **1-year\n  limited warranty** covering "

### RAG-09 (warranty_rag)
- Query: 'What warranty comes with the standing desk?'
  - Failed checks: answer_contains_any
  - Answer: "Here's what our documentation says:\n\n(source: warranty-policy.md, chunk 0, score 0.356)\n# Warranty Policy\n\n(source: product-catalog.md, chunk 4, score 0.347)\n## NimbusDesk Standing Desk (PROD-103) — $449.00\n\nSources: [Warranty Policy] [Product Catalog]"

### CTX-02 (insufficient_context)
- Query: 'Do you sell motorcycles with rocket boosters?'
  - Failed checks: answer_contains_any, uncertainty_acknowledged
  - Answer: "Here's what our documentation says:\n\n(source: warranty-policy.md, chunk 5, score 0.164)\n## Making a warranty claim\n\n1. Contact support with your order number and photos or a video of the defect.\n2. Claims are reviewed within **2 business days**.\n3. Approved claims receive a repair, an identical repl"
