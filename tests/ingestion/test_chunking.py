from pathlib import Path

from rag_mvp.ingestion.chunking import TextChunker
from rag_mvp.ingestion.models import Document, SourcePage


def test_chunks_bilingual_text_and_preserves_metadata() -> None:
    document = Document(
        document_id="doc-1",
        content_sha256="version-1",
        source_path=Path("handbooks/leave.txt"),
        source_name="leave.txt",
        source_type="txt",
        pages=(SourcePage(("Parental leave policy. 员工育儿假政策。 " * 20), 3, "native"),),
    )

    chunks = TextChunker(chunk_size=120, overlap=20).chunk(document)

    assert len(chunks) > 1
    assert all(chunk.page_number == 3 for chunk in chunks)
    assert all(chunk.source_path == "handbooks/leave.txt" for chunk in chunks)
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert len({chunk.chunk_id for chunk in chunks}) == len(chunks)


def test_chunk_ids_are_deterministic() -> None:
    document = Document(
        document_id="doc-1",
        content_sha256="version-1",
        source_path=Path("guide.md"),
        source_name="guide.md",
        source_type="md",
        pages=(SourcePage("A sufficiently long compliance guide section." * 5, None, "native"),),
    )
    chunker = TextChunker(chunk_size=100, overlap=10)

    assert [item.chunk_id for item in chunker.chunk(document)] == [
        item.chunk_id for item in chunker.chunk(document)
    ]
