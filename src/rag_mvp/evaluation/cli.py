from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rag_mvp.evaluation.retrieval import RetrievalEvaluator
from rag_mvp.retrieval.api import get_retriever


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare retrieval configurations")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("evals/retrieval_cases.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/retrieval_evaluation"))
    parser.add_argument("--top-k", type=int, default=3)
    return parser


def write_reports(report: dict[str, object], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "results.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    configurations = report["configurations"]
    fields = [
        "name",
        "mode",
        "reranker_enabled",
        "hit_rate_at_k",
        "mrr",
        "context_precision",
        "refusal_accuracy",
        "latency_p50_ms",
        "latency_p95_ms",
    ]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for configuration in configurations:  # type: ignore[union-attr]
            writer.writerow({field: configuration[field] for field in fields})
    lines = [
        "# Retrieval Evaluation",
        "",
        f"Dataset: {report['dataset_size']} cases; top_k={report['top_k']}",
        "",
        "| Configuration | Hit@k | MRR | Context precision | Refusal accuracy | p50 ms | p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in configurations:  # type: ignore[union-attr]
        lines.append(
            f"| {item['name']} | {item['hit_rate_at_k']:.4f} | {item['mrr']:.4f} | "
            f"{item['context_precision']:.4f} | {item['refusal_accuracy']:.4f} | "
            f"{item['latency_p50_ms']:.3f} | {item['latency_p95_ms']:.3f} |"
        )
    baseline = configurations[0]  # type: ignore[index]
    best = max(
        configurations,  # type: ignore[arg-type]
        key=lambda item: (
            item["hit_rate_at_k"],
            item["context_precision"],
            item["refusal_accuracy"],
            -item["latency_p95_ms"],
        ),
    )
    hit_delta = (best["hit_rate_at_k"] - baseline["hit_rate_at_k"]) * 100
    precision_delta = (best["context_precision"] - baseline["context_precision"]) * 100
    lines.extend(
        [
            "",
            "## Conclusion",
            "",
            (
                f"Recommended default: **{best['name']}**. Compared with vector-only, it improved "
                f"Hit@k by {hit_delta:.1f} percentage points and context precision by "
                f"{precision_delta:.1f} percentage points on this dataset."
            ),
            "",
            (
                "The dataset is intentionally small and should be expanded before production. "
                "Latency values measure local retrieval only and should be rerun on deployment hardware."
            ),
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    evaluator = RetrievalEvaluator(get_retriever(), top_k=args.top_k)
    cases = evaluator.load_cases(args.dataset)
    report = evaluator.evaluate(cases)
    write_reports(report, args.output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
