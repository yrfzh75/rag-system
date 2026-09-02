# Retrieval Evaluation

Dataset: 12 cases; top_k=3

| Configuration | Hit@k | MRR | Context precision | Refusal accuracy | p50 ms | p95 ms |
|---|---:|---:|---:|---:|---:|---:|
| vector_only | 0.9000 | 0.9000 | 0.9000 | 1.0000 | 2.553 | 4.211 |
| hybrid | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.362 | 3.898 |
| hybrid_rerank | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 2.464 | 2.759 |

## Conclusion

Recommended default: **hybrid_rerank**. Compared with vector-only, it improved Hit@k by 10.0 percentage points and context precision by 10.0 percentage points on this dataset.

The dataset is intentionally small and should be expanded before production. Latency values measure local retrieval only and should be rerun on deployment hardware.
