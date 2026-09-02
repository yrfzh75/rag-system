from __future__ import annotations

from time import perf_counter
from uuid import uuid4

from rag_mvp.qa.generation import AnswerGenerator
from rag_mvp.qa.models import QAResponse, SourceCitation, TokenUsage
from rag_mvp.retrieval.service import VectorRetriever


class GroundedQAService:
    """Retrieve evidence, refuse unsupported questions, and generate grounded answers."""

    def __init__(self, *, retriever: VectorRetriever, generator: AnswerGenerator) -> None:
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, *, top_k: int | None = None) -> QAResponse:
        request_id = str(uuid4())
        started = perf_counter()
        retrieval_started = perf_counter()
        contexts = self.retriever.retrieve(query, top_k=top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        if not contexts:
            total_ms = (perf_counter() - started) * 1000
            return QAResponse(
                answer=(
                    "I cannot answer from the available documents. "
                    "Try rephrasing the question or contact the document owner."
                ),
                sources=[],
                refused=True,
                refusal_reason="no_relevant_context",
                request_id=request_id,
                model=None,
                retrieval_ms=retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
                token_usage=TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
            )

        generation_started = perf_counter()
        generated = self.generator.generate(query, contexts)
        generation_ms = (perf_counter() - generation_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        sources = [
            SourceCitation(
                citation_id=index,
                source_name=str(item["source_name"]),
                source_path=str(item["source_path"]),
                page_number=(int(item["page_number"]) if item.get("page_number") else None),
                chunk_id=str(item["chunk_id"]),
                score=float(item["score"]),
            )
            for index, item in enumerate(contexts, start=1)
        ]
        return QAResponse(
            answer=generated.text,
            sources=sources,
            refused=False,
            refusal_reason=None,
            request_id=request_id,
            model=generated.model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            token_usage=TokenUsage(
                input_tokens=generated.input_tokens,
                output_tokens=generated.output_tokens,
                total_tokens=generated.input_tokens + generated.output_tokens,
            ),
        )
