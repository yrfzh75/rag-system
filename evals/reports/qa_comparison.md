# QA hardening comparison

| Metric | Initial live run | Hardened run | Required | Result |
|---|---:|---:|---:|---|
| Faithfulness | 0.770 | 0.870 | >= 0.85 | Pass |
| Answer compliance | 0.833 | 1.000 | >= 0.90 | Pass |
| Style consistency | 1.000 | 1.000 | >= 0.85 | Pass |
| Refusal appropriateness | 1.000 | 1.000 | >= 0.90 | Pass |
| Requests within 10 seconds | 1.000 | 1.000 | >= 0.90 | Pass |

The first run exposed a mislabeled retention question whose expected answer was absent from the
sample corpus, plus overlong model output that added unsupported commentary. The hardened run uses
a corpus-backed retention case and requires at most three cited factual statements. Both changes
are recorded because evaluation-data quality and prompt behavior are separate failure modes. Full
per-case responses and metric inputs are in `qa_before/results.json` and
`qa_baseline/results.json`.

The final load check used five distinct simultaneous calls with caching disabled. It measured p50
`4.68s`, p95 `6.76s`, zero cache hits, and 100% within 10 seconds. Re-run on production-like
hardware for capacity planning.
