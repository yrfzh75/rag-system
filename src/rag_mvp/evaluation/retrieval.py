from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Protocol

from rag_mvp.retrieval.service import RetrievalMode


class EvaluatedRetriever(Protocol):
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        mode: RetrievalMode | None = None,
        reranker_enabled: bool | None = None,
    ) -> list[dict[str, object]]: ...


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    language: str
    query: str
    expected_sources: list[str]


@dataclass(frozen=True)
class RetrievalConfiguration:
    name: str
    mode: RetrievalMode
    reranker_enabled: bool


CONFIGURATIONS = (
    RetrievalConfiguration("vector_only", "vector_only", False),
    RetrievalConfiguration("hybrid", "hybrid", False),
    RetrievalConfiguration("hybrid_rerank", "hybrid", True),
)


def percentile(values: list[float], percentile_value: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile_value * len(ordered)) - 1)
    return ordered[index]


class RetrievalEvaluator:
    def __init__(self, retriever: EvaluatedRetriever, *, top_k: int = 3) -> None:
        self.retriever = retriever
        self.top_k = top_k

    @staticmethod
    def load_cases(path: Path) -> list[EvaluationCase]:
        raw_cases = json.loads(path.read_text(encoding="utf-8"))
        return [EvaluationCase(**case) for case in raw_cases]

    def evaluate(
        self,
        cases: list[EvaluationCase],
        configurations: tuple[RetrievalConfiguration, ...] = CONFIGURATIONS,
    ) -> dict[str, object]:
        reports = [self._evaluate_configuration(cases, config) for config in configurations]
        return {
            "dataset_size": len(cases),
            "top_k": self.top_k,
            "configurations": reports,
        }

    def _evaluate_configuration(
        self, cases: list[EvaluationCase], configuration: RetrievalConfiguration
    ) -> dict[str, object]:
        details: list[dict[str, object]] = []
        latencies: list[float] = []
        hit_rates: list[float] = []
        reciprocal_ranks: list[float] = []
        precisions: list[float] = []
        refusal_scores: list[float] = []

        for case in cases:
            started = perf_counter()
            results = self.retriever.retrieve(
                case.query,
                top_k=self.top_k,
                mode=configuration.mode,
                reranker_enabled=configuration.reranker_enabled,
            )
            latency_ms = (perf_counter() - started) * 1000
            latencies.append(latency_ms)
            returned_sources = [str(result["source_name"]) for result in results]
            expected = set(case.expected_sources)
            relevant_positions = [
                index
                for index, source in enumerate(returned_sources, start=1)
                if source in expected
            ]
            if expected:
                hit = float(bool(relevant_positions))
                reciprocal_rank = 1 / relevant_positions[0] if relevant_positions else 0.0
                seen_relevant_sources: set[str] = set()
                precision_sum = 0.0
                for rank, source in enumerate(returned_sources, start=1):
                    if source in expected and source not in seen_relevant_sources:
                        seen_relevant_sources.add(source)
                        precision_sum += len(seen_relevant_sources) / rank
                precision = precision_sum / len(expected)
                hit_rates.append(hit)
                reciprocal_ranks.append(reciprocal_rank)
                precisions.append(precision)
                refusal_correct = None
            else:
                hit = None
                reciprocal_rank = None
                precision = None
                refusal_correct = float(not results)
                refusal_scores.append(refusal_correct)
            details.append(
                {
                    **asdict(case),
                    "returned_sources": returned_sources,
                    "latency_ms": round(latency_ms, 3),
                    "hit": hit,
                    "reciprocal_rank": reciprocal_rank,
                    "context_precision": precision,
                    "refusal_correct": refusal_correct,
                }
            )

        return {
            "name": configuration.name,
            "mode": configuration.mode,
            "reranker_enabled": configuration.reranker_enabled,
            "hit_rate_at_k": round(mean(hit_rates), 4) if hit_rates else 0.0,
            "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
            "context_precision": round(mean(precisions), 4) if precisions else 0.0,
            "refusal_accuracy": round(mean(refusal_scores), 4) if refusal_scores else 0.0,
            "latency_p50_ms": round(median(latencies), 3) if latencies else 0.0,
            "latency_p95_ms": round(percentile(latencies, 0.95), 3),
            "cases": details,
        }
