from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from statistics import median
from time import perf_counter

import httpx

from rag_mvp.evaluation.qa import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate QA concurrency and latency")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/qa")
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--expect-cold-cache", action="store_true")
    args = parser.parse_args()

    queries = [
        "How many weeks of parental leave are available?",
        "员工有多少周育儿假？",
        "When should customer export files be deleted?",
        "安全事件应该如何上报？",
        "Which component selects vector-only or hybrid search?",
    ]

    def call(index: int) -> tuple[float, bool]:
        started = perf_counter()
        with httpx.Client(timeout=180) as client:
            response = client.post(
                args.endpoint,
                json={
                    "query": queries[index % len(queries)],
                    "session_id": f"load-{index}",
                    "top_k": 3,
                },
            )
            response.raise_for_status()
            cache_hit = bool(response.json()["cache_hit"])
        return (perf_counter() - started) * 1000, cache_hit

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(call, range(args.requests)))
    latencies = [result[0] for result in results]
    cache_hit_rate = sum(result[1] for result in results) / len(results)
    latency_pass = sum(value <= 10_000 for value in latencies) / len(latencies) >= 0.9
    cache_pass = not args.expect_cold_cache or cache_hit_rate == 0
    summary = {
        "concurrency": args.concurrency,
        "requests": args.requests,
        "p50_ms": median(latencies),
        "p95_ms": percentile(latencies, 0.95),
        "requests_within_10s": sum(value <= 10_000 for value in latencies) / len(latencies),
        "cache_hit_rate": cache_hit_rate,
        "cold_cache_verified": args.expect_cold_cache and cache_hit_rate == 0,
        "passes": args.concurrency >= 5 and latency_pass and cache_pass,
    }
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["passes"] else 1)


if __name__ == "__main__":
    main()
