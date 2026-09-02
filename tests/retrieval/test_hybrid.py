from collections.abc import Sequence

from rag_mvp.retrieval.service import BM25Index, ConfigurableRetriever

CHUNKS = [
    {
        "chunk_id": "leave",
        "source_name": "handbook.md",
        "text": "Employees receive sixteen weeks parental leave. 员工享受十六周育儿假。",
    },
    {
        "chunk_id": "latency",
        "source_name": "spec.txt",
        "text": "QA latency must be less than ten seconds.",
    },
]


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[1.0, 0.0] for _ in texts]


class FakeStore:
    def list_chunks(self) -> list[dict[str, object]]:
        return CHUNKS

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        return [
            {"score": 0.8, **CHUNKS[1]},
            {"score": 0.7, **CHUNKS[0]},
        ][:top_k]


def test_bm25_supports_english_and_chinese_terms() -> None:
    index = BM25Index(CHUNKS)

    assert index.search("parental leave", top_k=1)[0]["chunk_id"] == "leave"
    assert index.search("育儿假", top_k=1)[0]["chunk_id"] == "leave"


def test_hybrid_and_reranker_promote_exact_match() -> None:
    retriever = ConfigurableRetriever(
        embedder=FakeEmbedder(),
        store=FakeStore(),
        default_score_threshold=0.0,
        candidate_k=2,
    )

    results = retriever.retrieve(
        "parental leave",
        mode="hybrid",
        reranker_enabled=True,
        top_k=1,
    )

    assert results[0]["chunk_id"] == "leave"
