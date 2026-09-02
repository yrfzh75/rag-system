from functools import lru_cache
from typing import Annotated

from fastapi import APIRouter, Depends

from rag_mvp.config import get_settings
from rag_mvp.qa.generation import OllamaAnswerGenerator
from rag_mvp.qa.models import QARequest, QAResponse
from rag_mvp.qa.service import GroundedQAService
from rag_mvp.retrieval.api import get_retriever

router = APIRouter(tags=["question answering"])


@lru_cache
def get_qa_service() -> GroundedQAService:
    settings = get_settings()
    generator = OllamaAnswerGenerator(
        model=settings.llm_model,
        base_url=settings.ollama_base_url,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    return GroundedQAService(retriever=get_retriever(), generator=generator)


@router.post("/qa", response_model=QAResponse)
def answer(
    request: QARequest,
    service: Annotated[GroundedQAService, Depends(get_qa_service)],
) -> QAResponse:
    return service.answer(request.query, top_k=request.top_k)
