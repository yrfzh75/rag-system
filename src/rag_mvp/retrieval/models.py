from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    score_threshold: float | None = Field(default=None, ge=-1.0, le=1.0)


class RetrievalHit(BaseModel):
    chunk_id: str
    document_id: str
    text: str
    score: float
    source_path: str
    source_name: str
    source_type: str
    page_number: int | None
    chunk_index: int
    language: str
    extraction_method: str


class RetrievalResponse(BaseModel):
    query: str
    result_count: int
    results: list[RetrievalHit]
