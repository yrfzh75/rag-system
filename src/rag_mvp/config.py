from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "bilingual-rag-mvp"
    app_env: str = "development"
    ingest_source_dir: str = "./documents"
    ingest_qdrant_path: str = "./data/qdrant"
    ingest_collection: str = "knowledge_chunks"
    ingest_embedding_model: str = (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )
    ingest_model_cache: str = "./data/models"
    ingest_chunk_size: int = 1000
    ingest_chunk_overlap: int = 150
    ingest_ocr_enabled: bool = True
    ingest_ocr_languages: str = "eng+chi_sim"
    ingest_ocr_min_text_chars: int = 40

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    return Settings()
