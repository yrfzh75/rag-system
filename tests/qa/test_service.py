from collections.abc import Sequence

from rag_mvp.qa.generation import GeneratedAnswer
from rag_mvp.qa.service import GroundedQAService


class FakeRetriever:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results

    def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        return self.results


class FakeGenerator:
    def __init__(self) -> None:
        self.called = False

    def generate(
        self,
        query: str,
        contexts: Sequence[dict[str, object]],
    ) -> GeneratedAnswer:
        self.called = True
        return GeneratedAnswer(
            text="Employees receive 16 weeks of paid parental leave. [1]",
            model="test-model",
            input_tokens=100,
            output_tokens=12,
        )


def make_context() -> dict[str, object]:
    return {
        "chunk_id": "chunk-1",
        "document_id": "doc-1",
        "text": "Employees receive 16 weeks of paid parental leave.",
        "score": 0.9,
        "source_path": "employee_handbook.md",
        "source_name": "employee_handbook.md",
        "source_type": "md",
        "page_number": None,
        "chunk_index": 0,
        "language": "en",
        "extraction_method": "native",
    }


def test_generates_answer_with_sources_and_usage() -> None:
    generator = FakeGenerator()
    service = GroundedQAService(
        retriever=FakeRetriever([make_context()]),  # type: ignore[arg-type]
        generator=generator,
    )

    response = service.answer("How much parental leave is available?")

    assert response.refused is False
    assert response.sources[0].source_name == "employee_handbook.md"
    assert response.token_usage.total_tokens == 112
    assert generator.called is True


def test_refuses_without_relevant_context_and_skips_llm() -> None:
    generator = FakeGenerator()
    service = GroundedQAService(
        retriever=FakeRetriever([]),  # type: ignore[arg-type]
        generator=generator,
    )

    response = service.answer("What is the office Wi-Fi password?")

    assert response.refused is True
    assert response.refusal_reason == "no_relevant_context"
    assert response.sources == []
    assert response.model is None
    assert generator.called is False
