from fastapi import FastAPI

from rag_mvp.config import get_settings
from rag_mvp.qa.api import router as qa_router
from rag_mvp.retrieval.api import router as retrieval_router

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Bilingual RAG MVP API",
)
app.include_router(retrieval_router)
app.include_router(qa_router)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "ok", "environment": settings.app_env}
