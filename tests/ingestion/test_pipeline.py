from pathlib import Path

from rag_mvp.ingestion.chunking import TextChunker
from rag_mvp.ingestion.parsers import DocumentParser
from rag_mvp.ingestion.pipeline import IngestionPipeline


class FakeEmbedder:
    dimension = 3

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(text)), 1.0, 0.0] for text in texts]


class RecordingStore:
    def __init__(self) -> None:
        self.documents: list[tuple[list[object], list[list[float]]]] = []

    def replace_document(self, chunks: list[object], vectors: list[list[float]]) -> None:
        self.documents.append((chunks, vectors))


def test_pipeline_continues_after_a_bad_document(tmp_path: Path) -> None:
    good = tmp_path / "good.txt"
    good.write_text("A valid employee handbook section. " * 10, encoding="utf-8")
    bad = tmp_path / "bad.txt"
    bad.write_text("", encoding="utf-8")
    store = RecordingStore()
    events: list[dict[str, object]] = []
    pipeline = IngestionPipeline(
        parser=DocumentParser(corpus_root=tmp_path),
        chunker=TextChunker(chunk_size=120, overlap=20),
        embedder=FakeEmbedder(),
        store=store,
        progress=events.append,
    )

    report = pipeline.ingest_directory(tmp_path)

    assert report.discovered_files == 2
    assert report.ingested_files == 1
    assert report.failed_files == 1
    assert report.chunks_written > 0
    assert len(store.documents) == 1
    assert {event["event"] for event in events} == {
        "ingestion.document_completed",
        "ingestion.document_failed",
    }

