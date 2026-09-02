from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from rag_mvp.ingestion.chunking import TextChunker
from rag_mvp.ingestion.embedding import Embedder
from rag_mvp.ingestion.models import Chunk, IngestionReport
from rag_mvp.ingestion.parsers import SUPPORTED_SUFFIXES, DocumentParser


class ChunkStore(Protocol):
    def replace_document(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...


class IngestionPipeline:
    def __init__(
        self,
        *,
        parser: DocumentParser,
        chunker: TextChunker,
        embedder: Embedder,
        store: ChunkStore,
        progress: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.parser = parser
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.progress = progress or (lambda _: None)

    def ingest_directory(self, source_dir: Path) -> IngestionReport:
        source_dir = source_dir.resolve()
        files = sorted(
            path
            for path in source_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
        )
        report = IngestionReport(discovered_files=len(files))
        for path in files:
            try:
                self._ingest_file(path, report)
            except Exception as exc:  # noqa: BLE001 -- isolate failures to one source document
                report.failed_files += 1
                report.errors.append({"source_path": str(path), "error": str(exc)})
                self.progress(
                    {
                        "event": "ingestion.document_failed",
                        "source_path": str(path),
                        "error": str(exc),
                    }
                )
        return report

    def _ingest_file(self, path: Path, report: IngestionReport) -> None:
        document = self.parser.parse(path)
        chunks = self.chunker.chunk(document)
        if not chunks:
            raise ValueError("Document produced no non-empty chunks")
        vectors = self.embedder.embed([chunk.text for chunk in chunks])
        self.store.replace_document(chunks, vectors)

        report.ingested_files += 1
        report.chunks_written += len(chunks)
        report.native_pages += sum(page.extraction_method == "native" for page in document.pages)
        report.ocr_pages += sum(page.extraction_method == "ocr" for page in document.pages)
        self.progress(
            {
                "event": "ingestion.document_completed",
                "source_path": document.source_path.as_posix(),
                "content_sha256": document.content_sha256,
                "chunks_written": len(chunks),
            }
        )
