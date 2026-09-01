from fastapi import FastAPI

from rag_mvp.config import get_settings

settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Bilingual RAG MVP API",
)


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    """Report that the API process is ready to accept requests."""
    return {"status": "ok", "environment": settings.app_env}

