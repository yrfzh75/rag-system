from __future__ import annotations

import re
import uuid

from rag_mvp.ingestion.language import detect_language
from rag_mvp.ingestion.models import Chunk, Document

_BREAK_CHARS = frozenset(" \t\n。！？；.!?;")


class TextChunker:
    """Character-window chunker that handles unspaced Chinese and spaced English text."""

    def __init__(self, *, chunk_size: int = 1000, overlap: int = 150) -> None:
        if chunk_size < 100:
            raise ValueError("chunk_size must be at least 100 characters")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must be non-negative and smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks: list[Chunk] = []
        chunk_index = 0
        for page in document.pages:
            for text in self._split(page.text):
                identity = f"{document.document_id}:{page.page_number}:{chunk_index}:{text}"
                chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, identity))
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        content_sha256=document.content_sha256,
                        text=text,
                        source_path=document.source_path.as_posix(),
                        source_name=document.source_name,
                        source_type=document.source_type,
                        page_number=page.page_number,
                        chunk_index=chunk_index,
                        language=detect_language(text),
                        extraction_method=page.extraction_method,
                    )
                )
                chunk_index += 1
        return chunks

    def _split(self, text: str) -> list[str]:
        normalized = re.sub(r"[ \t]+", " ", text.replace("\r\n", "\n").replace("\r", "\n"))
        normalized = re.sub(r"\n{3,}", "\n\n", normalized).strip()
        if not normalized:
            return []

        pieces: list[str] = []
        start = 0
        while start < len(normalized):
            proposed_end = min(start + self.chunk_size, len(normalized))
            end = proposed_end
            if proposed_end < len(normalized):
                search_floor = start + max(self.chunk_size // 2, self.chunk_size - 200)
                for cursor in range(proposed_end, search_floor, -1):
                    if normalized[cursor - 1] in _BREAK_CHARS:
                        end = cursor
                        break
            piece = normalized[start:end].strip()
            if piece:
                pieces.append(piece)
            if end >= len(normalized):
                break
            next_start = max(start + 1, end - self.overlap)
            while next_start < end and normalized[next_start] not in _BREAK_CHARS:
                next_start += 1
            start = min(next_start + 1, end)
        return pieces
