from pydantic import BaseModel, Field


class QARequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceCitation(BaseModel):
    citation_id: int
    source_name: str
    source_path: str
    page_number: int | None
    chunk_id: str
    score: float


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int


class QAResponse(BaseModel):
    answer: str
    sources: list[SourceCitation]
    refused: bool
    refusal_reason: str | None
    request_id: str
    model: str | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    token_usage: TokenUsage
