# Reproducible issue diagnosis

## Issue 1: dense-only retrieval missed a relevant bilingual case

- Evidence: the committed 12-case retrieval evaluation measured vector-only Hit@3 and context
  precision at `0.90`; hybrid measured `1.00` for both.
- Cause: exact bilingual terms and identifiers were not always ranked strongly by dense similarity.
- Fix: fuse multilingual vector search with BM25 using reciprocal-rank fusion.
- Improvement: `0.90 -> 1.00`, an absolute 10 percentage-point and relative 11.1% improvement.
- Reproduce: `rag-evaluate --dataset evals/retrieval_cases.json --output /tmp/retrieval-eval --top-k 3`.

## Issue 2: unsafe prompt-override requests could reach generation

- Evidence: before hardening, `GroundedQAService.answer` performed retrieval immediately and had no
  query safety gate, so the prompt-injection test's pre-generation refusal rate was `0.00`.
- Cause: context-level grounding instructions existed, but input-level override detection did not.
- Fix: deterministic English/Chinese injection patterns now produce `safety_policy` refusal before
  retrieval or generation; a regression test verifies the generator is never called.
- Improvement: targeted safety refusal accuracy `0.00 -> 1.00` (100 percentage points).
- Reproduce: `python -m pytest tests/qa/test_service.py -k prompt_injection -q`.
- Machine-readable before/after metrics are in `issue_diagnosis_metrics.json`; the hardened QA
  results also contain the safety case and its emitted request trace.

These small diagnostic sets establish reproducibility, not statistical certainty. Expand both
datasets with production-like failure examples before setting a production SLO.
