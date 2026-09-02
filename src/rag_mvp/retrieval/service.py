from __future__ import annotations

from typing import Protocol

from rag_mvp.ingestion.embedding import Embedder


class SearchStore(Protocol):
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]: ...


class VectorRetriever:
    """Embed a query and retrieve matching chunks from the vector store."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: SearchStore,
        default_top_k: int = 5,
        default_score_threshold: float | None = 0.3,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.default_top_k = default_top_k
        self.default_score_threshold = default_score_threshold

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be blank")
        vector = self.embedder.embed([normalized_query])[0]
        return self.store.search(
            vector,
            top_k=top_k if top_k is not None else self.default_top_k,
            score_threshold=(
                score_threshold if score_threshold is not None else self.default_score_threshold
            ),
        )
