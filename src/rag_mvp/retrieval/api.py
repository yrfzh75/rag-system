from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, Depends

from rag_mvp.config import get_settings
from rag_mvp.ingestion.embedding import FastEmbedder
from rag_mvp.ingestion.store import QdrantChunkStore
from rag_mvp.retrieval.models import RetrievalRequest, RetrievalResponse
from rag_mvp.retrieval.service import VectorRetriever

router = APIRouter(prefix="/retrieval", tags=["retrieval"])


@lru_cache
def get_retriever() -> VectorRetriever:
    """Build the local retrieval dependencies once, on the first query."""
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
    return VectorRetriever(
        embedder=embedder,
        store=store,
        default_top_k=settings.retrieval_top_k,
        default_score_threshold=settings.retrieval_score_threshold,
    )


@router.post("/search", response_model=RetrievalResponse)
def search(
    request: RetrievalRequest,
    retriever: Annotated[VectorRetriever, Depends(get_retriever)],
) -> RetrievalResponse:
    results = retriever.retrieve(
        request.query,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
    )
    return RetrievalResponse(
        query=request.query,
        result_count=len(results),
        results=results,
    )
