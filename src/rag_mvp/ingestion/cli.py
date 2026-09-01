from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_mvp.config import get_settings
from rag_mvp.ingestion.chunking import TextChunker
from rag_mvp.ingestion.embedding import FastEmbedder
from rag_mvp.ingestion.parsers import DocumentParser
from rag_mvp.ingestion.pipeline import IngestionPipeline
from rag_mvp.ingestion.store import QdrantChunkStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Ingest internal documents into Qdrant")
    parser.add_argument("source", nargs="?", help="Corpus directory; defaults to INGEST_SOURCE_DIR")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    settings = get_settings()
    source_dir = Path(args.source or settings.ingest_source_dir).resolve()
    if not source_dir.is_dir():
        raise SystemExit(f"Source directory does not exist: {source_dir}")

    embedder = FastEmbedder(
        settings.ingest_embedding_model,
        cache_dir=Path(settings.ingest_model_cache),
    )
    store = QdrantChunkStore(
        path=Path(settings.ingest_qdrant_path),
        collection=settings.ingest_collection,
        vector_size=embedder.dimension,
    )
    pipeline = IngestionPipeline(
        parser=DocumentParser(
            corpus_root=source_dir,
            ocr_enabled=settings.ingest_ocr_enabled,
            ocr_languages=settings.ingest_ocr_languages,
            ocr_min_text_chars=settings.ingest_ocr_min_text_chars,
        ),
        chunker=TextChunker(
            chunk_size=settings.ingest_chunk_size,
            overlap=settings.ingest_chunk_overlap,
        ),
        embedder=embedder,
        store=store,
        progress=lambda event: print(json.dumps(event, ensure_ascii=False)),
    )
    try:
        report = pipeline.ingest_directory(source_dir)
    finally:
        store.close()

    print(json.dumps({"event": "ingestion.completed", **report.as_dict()}, ensure_ascii=False))
    if report.failed_files:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
