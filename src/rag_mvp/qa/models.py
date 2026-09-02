from pydantic import BaseModel, Field


class QARequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    session_id: str | None = Field(default=None, min_length=1, max_length=128)


class SourceCitation(BaseModel):
    citation_id: int
    source_name: str
    source_path: str
    page_number: int | None
    chunk_id: str
    score: float
    text: str | None = None


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
    trace_id: str
    session_id: str | None = None
    cache_hit: bool = False
    pii_redacted: bool = False
    grounding_score: float | None = None
    model: str | None
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    token_usage: TokenUsage
