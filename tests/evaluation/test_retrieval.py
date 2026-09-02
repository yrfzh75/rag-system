from rag_mvp.evaluation.retrieval import EvaluationCase, RetrievalConfiguration, RetrievalEvaluator


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        mode: str | None = None,
        reranker_enabled: bool | None = None,
    ) -> list[dict[str, object]]:
        if query == "unknown":
            return []
        return [
            {"source_name": "handbook.md", "chunk_id": "1", "score": 0.9},
            {"source_name": "other.md", "chunk_id": "2", "score": 0.5},
        ]


def test_evaluator_calculates_retrieval_and_refusal_metrics() -> None:
    evaluator = RetrievalEvaluator(FakeRetriever(), top_k=2)  # type: ignore[arg-type]
    report = evaluator.evaluate(
        [
            EvaluationCase("known", "en", "known", ["handbook.md"]),
            EvaluationCase("unknown", "en", "unknown", []),
        ],
        (RetrievalConfiguration("vector", "vector_only", False),),
    )

    metrics = report["configurations"][0]  # type: ignore[index]
    assert metrics["hit_rate_at_k"] == 1.0
    assert metrics["mrr"] == 1.0
    assert metrics["context_precision"] == 1.0
    assert metrics["refusal_accuracy"] == 1.0
