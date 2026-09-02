from collections.abc import Sequence

from rag_mvp.retrieval.service import VectorRetriever


class FakeEmbedder:
    @property
    def dimension(self) -> int:
        return 3

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        assert texts == ["parental leave"]
        return [[1.0, 0.0, 0.0]]


class FakeStore:
    def __init__(self) -> None:
        self.arguments: tuple[list[float], int, float | None] | None = None

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        self.arguments = (vector, top_k, score_threshold)
        return [{"chunk_id": "chunk-1", "score": 0.91}]


def test_retriever_embeds_normalized_query_and_searches_store() -> None:
    store = FakeStore()
    retriever = VectorRetriever(
        embedder=FakeEmbedder(),
        store=store,
        default_top_k=5,
        default_score_threshold=0.3,
    )

    results = retriever.retrieve("  parental leave  ", top_k=3, score_threshold=0.5)

    assert results == [{"chunk_id": "chunk-1", "score": 0.91}]
    assert store.arguments == ([1.0, 0.0, 0.0], 3, 0.5)
