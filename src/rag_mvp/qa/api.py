from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends

from rag_mvp.config import get_settings
from rag_mvp.qa.cache import TTLAnswerCache
from rag_mvp.qa.generation import OllamaAnswerGenerator
from rag_mvp.qa.models import QARequest, QAResponse
from rag_mvp.qa.observability import JsonLineEventSink
from rag_mvp.qa.service import GroundedQAService
from rag_mvp.retrieval.api import get_retriever

router = APIRouter(tags=["question answering"])
_qa_service: GroundedQAService | None = None
_qa_service_lock = Lock()


def get_qa_service() -> GroundedQAService:
    global _qa_service
    if _qa_service is None:
        with _qa_service_lock:
            if _qa_service is None:
                settings = get_settings()
                generator = OllamaAnswerGenerator(
                    model=settings.llm_model,
                    base_url=settings.ollama_base_url,
                    max_output_tokens=settings.llm_max_output_tokens,
                    timeout_seconds=settings.ollama_timeout_seconds,
                )
                cache = (
                    TTLAnswerCache(
                        max_entries=settings.qa_cache_max_entries,
                        ttl_seconds=settings.qa_cache_ttl_seconds,
                    )
                    if settings.qa_cache_enabled
                    else None
                )
                _qa_service = GroundedQAService(
                    retriever=get_retriever(),
                    generator=generator,
                    cache=cache,
                    event_sink=JsonLineEventSink(settings.qa_log_path),
                    history_max_turns=settings.qa_history_max_turns,
                    grounding_min_support=settings.qa_grounding_min_support,
                )
    return _qa_service


@router.post("/qa", response_model=QAResponse)
def answer(
    request: QARequest,
    service: Annotated[GroundedQAService, Depends(get_qa_service)],
) -> QAResponse:
    return service.answer(request.query, top_k=request.top_k, session_id=request.session_id)
