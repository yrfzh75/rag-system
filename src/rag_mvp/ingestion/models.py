from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourcePage:
    """Text extracted from one source unit, normally a document page."""

    text: str
    page_number: int | None
    extraction_method: str


@dataclass(frozen=True)
class Document:
    """A parsed source document with stable identity and version metadata."""

    document_id: str
    content_sha256: str
    source_path: Path
    source_name: str
    source_type: str
    pages: tuple[SourcePage, ...]


@dataclass(frozen=True)
class Chunk:
    """A searchable text chunk and its retrieval payload."""

    chunk_id: str
    document_id: str
    content_sha256: str
    text: str
    source_path: str
    source_name: str
    source_type: str
    page_number: int | None
    chunk_index: int
    language: str
    extraction_method: str

    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content_sha256": self.content_sha256,
            "text": self.text,
            "source_path": self.source_path,
            "source_name": self.source_name,
            "source_type": self.source_type,
            "page_number": self.page_number,
            "chunk_index": self.chunk_index,
            "language": self.language,
            "extraction_method": self.extraction_method,
        }


@dataclass
class IngestionReport:
    """Machine-readable summary of one ingestion run."""

    discovered_files: int = 0
    ingested_files: int = 0
    failed_files: int = 0
    chunks_written: int = 0
    native_pages: int = 0
    ocr_pages: int = 0
    errors: list[dict[str, str]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "discovered_files": self.discovered_files,
            "ingested_files": self.ingested_files,
            "failed_files": self.failed_files,
            "chunks_written": self.chunks_written,
            "native_pages": self.native_pages,
            "ocr_pages": self.ocr_pages,
            "errors": self.errors,
        }

