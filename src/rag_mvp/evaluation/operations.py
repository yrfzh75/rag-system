from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import median

from rag_mvp.evaluation.qa import percentile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an operations report from QA JSONL logs")
    parser.add_argument("--logs", default="artifacts/qa_events.jsonl")
    parser.add_argument("--output", default="artifacts/operations.csv")
    parser.add_argument("--qa-results", default="evals/reports/qa_latest/results.json")
    args = parser.parse_args()
    events = [
        json.loads(line)
        for line in Path(args.logs).read_text(encoding="utf-8").splitlines()
        if line.strip() and json.loads(line).get("event") == "qa.request_completed"
    ]
    if not events:
        raise SystemExit("No qa.request_completed events found")
    latencies = [float(event["total_ms"]) for event in events]
    qa_results = Path(args.qa_results)
    compliance: float | str = "not_measured"
    if qa_results.exists():
        compliance = float(
            json.loads(qa_results.read_text(encoding="utf-8"))["summary"]["answer_compliance"]
        )
    row = {
        "request_count": len(events),
        "p50_latency_ms": round(median(latencies), 3),
        "p95_latency_ms": round(percentile(latencies, 0.95), 3),
        "token_usage": sum(int(event["total_tokens"]) for event in events),
        "cache_hit_rate": sum(bool(event["cache_hit"]) for event in events) / len(events),
        "refusal_rate": sum(bool(event["refused"]) for event in events) / len(events),
        "answer_compliance_rate": compliance,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, indent=2))


if __name__ == "__main__":
    main()
