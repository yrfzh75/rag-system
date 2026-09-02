from pathlib import Path
from threading import Lock
from typing import Annotated

from fastapi import APIRouter, Depends

from rag_mvp.config import get_settings
from rag_mvp.ingestion.embedding import FastEmbedder
from rag_mvp.ingestion.store import QdrantChunkStore
from rag_mvp.retrieval.models import RetrievalRequest, RetrievalResponse
from rag_mvp.retrieval.service import ConfigurableRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])
_retriever: ConfigurableRetriever | None = None
_retriever_lock = Lock()


def get_retriever() -> ConfigurableRetriever:
    """Build the local retrieval dependencies once, on the first query."""
    global _retriever
    if _retriever is None:
        with _retriever_lock:
            if _retriever is None:
                settings = get_settings()
                embedder = FastEmbedder(
                    settings.ingest_embedding_model,
                    cache_dir=Path(settings.ingest_model_cache),
                )
                store = QdrantChunkStore(
                    path=Path(settings.ingest_qdrant_path),
                    collection=settings.ingest_collection,
                    vector_size=embedder.dimension,
                )
                _retriever = ConfigurableRetriever(
                    embedder=embedder,
                    store=store,
                    default_top_k=settings.retrieval_top_k,
                    default_score_threshold=settings.retrieval_score_threshold,
                    default_mode=settings.retrieval_mode,
                    candidate_k=settings.retrieval_candidate_k,
                    reranker_enabled=settings.retrieval_reranker_enabled,
                )
    return _retriever


@router.post("/search", response_model=RetrievalResponse)
def search(
    request: RetrievalRequest,
    retriever: Annotated[ConfigurableRetriever, Depends(get_retriever)],
) -> RetrievalResponse:
    results = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        mode=request.mode,
        reranker_enabled=request.reranker_enabled,
    )
    return RetrievalResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )
