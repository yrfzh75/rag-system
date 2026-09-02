from fastapi.testclient import TestClient

from rag_mvp.main import app
from rag_mvp.retrieval.api import get_retriever


class FakeRetriever:
    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
        mode: str | None = None,
        reranker_enabled: bool | None = None,
    ) -> list[dict[str, object]]:
        assert query == "How much parental leave is available?"
        assert top_k == 2
        assert score_threshold is None
        assert mode is None
        assert reranker_enabled is None
        return [
            {
                "chunk_id": "chunk-1",
                "document_id": "doc-1",
                "text": "Eligible employees receive 16 weeks of parental leave.",
                "score": 0.91,
                "source_path": "employee_handbook.md",
                "source_name": "employee_handbook.md",
                "source_type": "md",
                "page_number": None,
                "chunk_index": 0,
                "language": "en",
                "extraction_method": "native",
            }
        ]


def test_vector_search_endpoint_returns_ranked_evidence() -> None:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    try:
        response = TestClient(app).post(
            "/retrieval/search",
            json={"query": "How much parental leave is available?", "top_k": 2},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["result_count"] == 1
    assert body["results"][0]["source_name"] == "employee_handbook.md"
    assert body["results"][0]["score"] == 0.91


def test_vector_search_rejects_invalid_top_k() -> None:
    app.dependency_overrides[get_retriever] = lambda: FakeRetriever()
    try:
        response = TestClient(app).post(
            "/retrieval/search",
            json={"query": "test", "top_k": 0},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 422
