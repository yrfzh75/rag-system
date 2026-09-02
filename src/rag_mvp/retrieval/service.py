from __future__ import annotations

import math
import re
from collections import Counter
from typing import Literal, Protocol

from rag_mvp.ingestion.embedding import Embedder


class SearchStore(Protocol):
    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]: ...

    def list_chunks(self) -> list[dict[str, object]]: ...


RetrievalMode = Literal["vector_only", "hybrid"]

ENGLISH_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "of",
    "the",
    "to",
    "what",
    "when",
    "which",
    "will",
    "with",
}


def tokenize(text: str) -> list[str]:
    """Tokenize English words plus Chinese characters and adjacent Chinese pairs."""
    lowered = text.lower()
    english = [
        token for token in re.findall(r"[a-z0-9]+", lowered) if token not in ENGLISH_STOPWORDS
    ]
    chinese_runs = re.findall(r"[\u3400-\u9fff]+", lowered)
    chinese: list[str] = []
    for run in chinese_runs:
        chinese.extend(run)
        chinese.extend(run[index : index + 2] for index in range(len(run) - 1))
    return english + chinese


class BM25Index:
    """Small in-memory bilingual BM25 index for the MVP corpus."""

    def __init__(
        self, chunks: list[dict[str, object]], *, k1: float = 1.5, b: float = 0.75
    ) -> None:
        self.chunks = chunks
        self.k1 = k1
        self.b = b
        self.documents = [tokenize(str(chunk.get("text", ""))) for chunk in chunks]
        self.term_frequencies = [Counter(document) for document in self.documents]
        self.average_length = (
            sum(len(document) for document in self.documents) / len(self.documents)
            if self.documents
            else 0.0
        )
        document_frequency: Counter[str] = Counter()
        for document in self.documents:
            document_frequency.update(set(document))
        total = len(self.documents)
        self.idf = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def search(self, query: str, *, top_k: int) -> list[dict[str, object]]:
        query_terms = tokenize(query)
        scored: list[tuple[float, dict[str, object]]] = []
        for chunk, document, frequencies in zip(
            self.chunks, self.documents, self.term_frequencies, strict=True
        ):
            score = 0.0
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if not frequency:
                    continue
                length_normalization = 1 - self.b
                if self.average_length:
                    length_normalization += self.b * len(document) / self.average_length
                score += self.idf.get(term, 0.0) * (
                    frequency * (self.k1 + 1) / (frequency + self.k1 * length_normalization)
                )
            if score > 0:
                scored.append((score, chunk))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [{"score": score, **chunk} for score, chunk in scored[:top_k]]


class LexicalOverlapReranker:
    """Deterministic bilingual second-stage reranker with no additional model download."""

    def rerank(
        self, query: str, candidates: list[dict[str, object]], *, top_k: int
    ) -> list[dict[str, object]]:
        query_terms = set(tokenize(query))
        rescored: list[dict[str, object]] = []
        for candidate in candidates:
            document_terms = set(tokenize(str(candidate.get("text", ""))))
            coverage = len(query_terms & document_terms) / max(len(query_terms), 1)
            score = 0.7 * float(candidate["score"]) + 0.3 * coverage
            rescored.append({**candidate, "score": score})
        return sorted(rescored, key=lambda item: float(item["score"]), reverse=True)[:top_k]


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


class ConfigurableRetriever(VectorRetriever):
    """Select vector or hybrid retrieval and optional reranking through configuration."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: SearchStore,
        default_top_k: int = 5,
        default_score_threshold: float | None = 0.3,
        default_mode: RetrievalMode = "vector_only",
        candidate_k: int = 10,
        reranker_enabled: bool = False,
    ) -> None:
        super().__init__(
            embedder=embedder,
            store=store,
            default_top_k=default_top_k,
            default_score_threshold=default_score_threshold,
        )
        if default_mode not in {"vector_only", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {default_mode}")
        self.default_mode = default_mode
        self.candidate_k = candidate_k
        self.default_reranker_enabled = reranker_enabled
        self.lexical_index = BM25Index(store.list_chunks())
        self.reranker = LexicalOverlapReranker()

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        mode: RetrievalMode | None = None,
        reranker_enabled: bool | None = None,
    ) -> list[dict[str, object]]:
        normalized_query = query.strip()
        if not normalized_query:
            raise ValueError("Query must not be blank")
        result_limit = top_k if top_k is not None else self.default_top_k
        selected_mode = mode or self.default_mode
        use_reranker = (
            reranker_enabled if reranker_enabled is not None else self.default_reranker_enabled
        )
        if selected_mode == "vector_only":
            results = super().retrieve(
                normalized_query,
                top_k=max(result_limit, self.candidate_k) if use_reranker else result_limit,
                score_threshold=score_threshold,
            )
        elif selected_mode == "hybrid":
            vector = self.embedder.embed([normalized_query])[0]
            vector_results = self.store.search(
                vector,
                top_k=self.candidate_k,
                score_threshold=None,
            )
            lexical_results = self.lexical_index.search(
                normalized_query,
                top_k=self.candidate_k,
            )
            confidence_threshold = (
                score_threshold if score_threshold is not None else self.default_score_threshold
            )
            dense_is_confident = bool(vector_results) and float(vector_results[0]["score"]) >= (
                confidence_threshold or 0.0
            )
            if not lexical_results and not dense_is_confident:
                return []
            results = self._reciprocal_rank_fusion(vector_results, lexical_results)
        else:
            raise ValueError(f"Unsupported retrieval mode: {selected_mode}")

        if use_reranker:
            results = self.reranker.rerank(normalized_query, results, top_k=result_limit)
        else:
            results = results[:result_limit]
        threshold = score_threshold if score_threshold is not None else self.default_score_threshold
        return [result for result in results if float(result["score"]) >= (threshold or 0.0)]

    @staticmethod
    def _reciprocal_rank_fusion(
        vector_results: list[dict[str, object]],
        lexical_results: list[dict[str, object]],
        *,
        rank_constant: int = 60,
    ) -> list[dict[str, object]]:
        fused: dict[str, dict[str, object]] = {}
        scores: dict[str, float] = {}
        for results in (vector_results, lexical_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = str(result["chunk_id"])
                fused[chunk_id] = result
                scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rank_constant + rank)
        maximum = max(scores.values(), default=1.0)
        ranked = [
            {**fused[chunk_id], "score": score / maximum} for chunk_id, score in scores.items()
        ]
        return sorted(ranked, key=lambda item: float(item["score"]), reverse=True)
