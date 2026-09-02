from __future__ import annotations

from collections import defaultdict, deque
from hashlib import sha256
from threading import Lock
from time import perf_counter
from uuid import uuid4

from rag_mvp.qa.cache import TTLAnswerCache
from rag_mvp.qa.generation import AnswerGenerator
from rag_mvp.qa.grounding import validate_grounding
from rag_mvp.qa.models import QAResponse, SourceCitation, TokenUsage
from rag_mvp.qa.observability import EventSink, NullEventSink
from rag_mvp.qa.security import contains_prompt_injection, redact_pii
from rag_mvp.retrieval.service import VectorRetriever


class GroundedQAService:
    """Retrieve evidence, refuse unsupported questions, and generate grounded answers."""

    def __init__(
        self,
        *,
        retriever: VectorRetriever,
        generator: AnswerGenerator,
        cache: TTLAnswerCache | None = None,
        event_sink: EventSink | None = None,
        history_max_turns: int = 3,
        grounding_min_support: float = 0.5,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.cache = cache
        self.event_sink = event_sink or NullEventSink()
        self.history_max_turns = history_max_turns
        self.grounding_min_support = grounding_min_support
        self._history: dict[str, deque[tuple[str, str]]] = defaultdict(
            lambda: deque(maxlen=history_max_turns)
        )
        self._history_lock = Lock()

    def answer(
        self, query: str, *, top_k: int | None = None, session_id: str | None = None
    ) -> QAResponse:
        request_id = str(uuid4())
        trace_id = uuid4().hex
        started = perf_counter()
        query_redaction = redact_pii(query)
        query_hash = sha256(query.encode("utf-8")).hexdigest()
        history = self._get_history(session_id)

        if contains_prompt_injection(query):
            return self._refusal(
                request_id=request_id,
                trace_id=trace_id,
                session_id=session_id,
                started=started,
                reason="safety_policy",
                answer=(
                    "I cannot follow instructions that attempt to override system safety. "
                    "Ask a factual question about the available internal documents."
                ),
                query_hash=query_hash,
                pii_redacted=query_redaction.redacted,
            )

        cache_key = self._cache_key(query_redaction.text, top_k, history)
        cached = self.cache.get(cache_key) if self.cache and not query_redaction.redacted else None
        if cached is not None:
            cached.request_id = request_id
            cached.trace_id = trace_id
            cached.session_id = session_id
            cached.cache_hit = True
            cached.retrieval_ms = 0.0
            cached.generation_ms = 0.0
            cached.total_ms = (perf_counter() - started) * 1000
            self._record(request_id, trace_id, query_hash, cached)
            return cached

        retrieval_started = perf_counter()
        retrieval_query = " ".join([*(turn[0] for turn in history[-1:]), query])
        contexts = self.retriever.retrieve(retrieval_query, top_k=top_k)
        retrieval_ms = (perf_counter() - retrieval_started) * 1000

        if not contexts:
            return self._refusal(
                request_id=request_id,
                trace_id=trace_id,
                session_id=session_id,
                started=started,
                reason="no_relevant_context",
                answer=(
                    "I cannot answer from the available documents. "
                    "Try rephrasing the question or contact the document owner."
                ),
                retrieval_ms=retrieval_ms,
                query_hash=query_hash,
                pii_redacted=query_redaction.redacted,
            )

        generation_started = perf_counter()
        generated = self.generator.generate(query_redaction.text, contexts, history=history)
        generation_ms = (perf_counter() - generation_started) * 1000
        total_ms = (perf_counter() - started) * 1000
        source_redactions = [redact_pii(str(item["text"])) for item in contexts]
        sources = [
            SourceCitation(
                citation_id=index,
                source_name=str(item["source_name"]),
                source_path=str(item["source_path"]),
                page_number=(int(item["page_number"]) if item.get("page_number") else None),
                chunk_id=str(item["chunk_id"]),
                score=float(item["score"]),
                text=source_redactions[index - 1].text,
            )
            for index, item in enumerate(contexts, start=1)
        ]
        answer_redaction = redact_pii(generated.text)
        grounding = validate_grounding(
            answer_redaction.text,
            [redaction.text for redaction in source_redactions],
            min_support=self.grounding_min_support,
        )
        pii_redacted = (
            query_redaction.redacted
            or answer_redaction.redacted
            or any(redaction.redacted for redaction in source_redactions)
        )
        if not grounding.valid:
            return self._refusal(
                request_id=request_id,
                trace_id=trace_id,
                session_id=session_id,
                started=started,
                reason="ungrounded_generation",
                answer=(
                    "I cannot provide a sufficiently grounded answer from the retrieved documents. "
                    "Try rephrasing the question or contact the document owner."
                ),
                retrieval_ms=retrieval_ms,
                generation_ms=generation_ms,
                query_hash=query_hash,
                pii_redacted=pii_redacted,
                model=generated.model,
                token_usage=TokenUsage(
                    input_tokens=generated.input_tokens,
                    output_tokens=generated.output_tokens,
                    total_tokens=generated.input_tokens + generated.output_tokens,
                ),
                grounding_score=grounding.support_score,
            )
        response = QAResponse(
            answer=answer_redaction.text,
            sources=sources,
            refused=False,
            refusal_reason=None,
            request_id=request_id,
            trace_id=trace_id,
            session_id=session_id,
            cache_hit=False,
            pii_redacted=pii_redacted,
            grounding_score=grounding.support_score,
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
        if session_id:
            with self._history_lock:
                self._history[session_id].append((query_redaction.text, response.answer))
        if self.cache and not response.pii_redacted:
            self.cache.set(cache_key, response)
        self._record(request_id, trace_id, query_hash, response)
        return response

    def _get_history(self, session_id: str | None) -> list[tuple[str, str]]:
        if not session_id:
            return []
        with self._history_lock:
            return list(self._history[session_id])

    @staticmethod
    def _cache_key(query: str, top_k: int | None, history: list[tuple[str, str]]) -> str:
        material = repr((query.casefold().strip(), top_k, history)).encode("utf-8")
        return sha256(material).hexdigest()

    def _refusal(
        self,
        *,
        request_id: str,
        trace_id: str,
        session_id: str | None,
        started: float,
        reason: str,
        answer: str,
        query_hash: str,
        pii_redacted: bool,
        retrieval_ms: float = 0.0,
        generation_ms: float = 0.0,
        model: str | None = None,
        token_usage: TokenUsage | None = None,
        grounding_score: float | None = None,
    ) -> QAResponse:
        response = QAResponse(
            answer=answer,
            sources=[],
            refused=True,
            refusal_reason=reason,
            request_id=request_id,
            trace_id=trace_id,
            session_id=session_id,
            cache_hit=False,
            pii_redacted=pii_redacted,
            grounding_score=grounding_score,
            model=model,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=(perf_counter() - started) * 1000,
            token_usage=token_usage
            or TokenUsage(input_tokens=0, output_tokens=0, total_tokens=0),
        )
        self._record(request_id, trace_id, query_hash, response)
        return response

    def _record(
        self, request_id: str, trace_id: str, query_hash: str, response: QAResponse
    ) -> None:
        self.event_sink.emit(
            {
                "event": "qa.request_completed",
                "request_id": request_id,
                "trace_id": trace_id,
                "retrieval_span_id": sha256(f"{trace_id}:retrieval".encode()).hexdigest()[:16],
                "generation_span_id": (
                    sha256(f"{trace_id}:generation".encode()).hexdigest()[:16]
                    if response.generation_ms > 0
                    else None
                ),
                "session_id": response.session_id,
                "query_sha256": query_hash,
                "retrieval_mode": getattr(self.retriever, "default_mode", "vector_only"),
                "reranker_enabled": getattr(
                    self.retriever, "default_reranker_enabled", False
                ),
                "model": response.model,
                "refused": response.refused,
                "refusal_reason": response.refusal_reason,
                "cache_hit": response.cache_hit,
                "pii_redacted": response.pii_redacted,
                "grounding_score": response.grounding_score,
                "source_count": len(response.sources),
                "input_tokens": response.token_usage.input_tokens,
                "output_tokens": response.token_usage.output_tokens,
                "total_tokens": response.token_usage.total_tokens,
                "retrieval_ms": round(response.retrieval_ms, 3),
                "generation_ms": round(response.generation_ms, 3),
                "total_ms": round(response.total_ms, 3),
            }
        )
