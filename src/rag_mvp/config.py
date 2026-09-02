from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "bilingual-rag-mvp"
    app_env: str = "development"
    ingest_source_dir: str = "./documents"
    ingest_qdrant_path: str = "./data/qdrant"
    ingest_collection: str = "knowledge_chunks"
    ingest_embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ingest_model_cache: str = "./data/models"
    ingest_chunk_size: int = 1000
    ingest_chunk_overlap: int = 150
    ingest_ocr_enabled: bool = True
    ingest_ocr_languages: str = "eng+chi_sim"
    ingest_ocr_min_text_chars: int = 40
    retrieval_top_k: int = 5
    retrieval_score_threshold: float = 0.3
    retrieval_mode: Literal["vector_only", "hybrid"] = "hybrid"
    retrieval_candidate_k: int = 10
    retrieval_reranker_enabled: bool = False
    llm_model: str = "qwen3:4b-instruct-2507-q4_K_M"
    llm_max_output_tokens: int = 500
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_timeout_seconds: float = 120.0

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
