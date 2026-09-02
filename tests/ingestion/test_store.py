from pathlib import Path

from rag_mvp.ingestion.models import Chunk
from rag_mvp.ingestion.store import QdrantChunkStore


def make_chunk(chunk_id: str, document_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content_sha256="version-1",
        text=text,
        source_path="handbook.txt",
        source_name="handbook.txt",
        source_type="txt",
        page_number=None,
        chunk_index=0,
        language="en",
        extraction_method="native",
    )


def test_replace_document_removes_old_chunks(tmp_path: Path) -> None:
    store = QdrantChunkStore(path=tmp_path / "qdrant", collection="test", vector_size=3)
    try:
        first = make_chunk("0ed8d4ed-f63e-45e4-a70b-8dbca95fe7f4", "doc-1", "old")
        replacement = make_chunk("826e6ce7-f11c-4635-82d4-3b3d68ddeaea", "doc-1", "new")
        store.replace_document([first], [[1.0, 0.0, 0.0]])
        store.replace_document([replacement], [[0.0, 1.0, 0.0]])

        points, _ = store.client.scroll(collection_name="test", with_payload=True, limit=10)
        assert len(points) == 1
        assert points[0].payload["text"] == "new"
    finally:
        store.close()


def test_search_returns_ranked_payloads(tmp_path: Path) -> None:
    store = QdrantChunkStore(path=tmp_path / "qdrant", collection="test", vector_size=3)
    try:
        first = make_chunk("0ed8d4ed-f63e-45e4-a70b-8dbca95fe7f4", "doc-1", "parental leave")
        second = make_chunk("826e6ce7-f11c-4635-82d4-3b3d68ddeaea", "doc-2", "API latency")
        store.replace_document([first], [[1.0, 0.0, 0.0]])
        store.replace_document([second], [[0.0, 1.0, 0.0]])

        results = store.search([1.0, 0.0, 0.0], top_k=1, score_threshold=0.5)

        assert len(results) == 1
        assert results[0]["text"] == "parental leave"
        assert results[0]["score"] == 1.0
    finally:
        store.close()
