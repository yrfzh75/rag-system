from fastapi.testclient import TestClient

from rag_mvp.main import app
from rag_mvp.qa.api import get_qa_service
from rag_mvp.qa.models import QAResponse, SourceCitation, TokenUsage


class FakeQAService:
    def answer(self, query: str, *, top_k: int | None = None) -> QAResponse:
        assert query == "How much parental leave is available?"
        assert top_k == 3
        return QAResponse(
            answer="Employees receive 16 weeks of paid parental leave. [1]",
            sources=[
                SourceCitation(
                    citation_id=1,
                    source_name="employee_handbook.md",
                    source_path="employee_handbook.md",
                    page_number=None,
                    chunk_id="chunk-1",
                    score=0.9,
                )
            ],
            refused=False,
            refusal_reason=None,
            request_id="request-1",
            model="test-model",
            retrieval_ms=5,
            generation_ms=20,
            total_ms=25,
            token_usage=TokenUsage(input_tokens=100, output_tokens=12, total_tokens=112),
        )


def test_qa_endpoint_returns_grounded_answer() -> None:
    app.dependency_overrides[get_qa_service] = lambda: FakeQAService()
    try:
        response = TestClient(app).post(
            "/qa",
            json={"query": "How much parental leave is available?", "top_k": 3},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["answer"].endswith("[1]")
    assert response.json()["sources"][0]["source_name"] == "employee_handbook.md"
