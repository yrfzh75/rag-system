from rag_mvp.qa.generation import OllamaAnswerGenerator


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, object]:
        return {
            "model": "qwen3:4b-instruct-2507-q4_K_M",
            "response": "Employees receive 16 weeks of paid parental leave. [1]",
            "prompt_eval_count": 140,
            "eval_count": 15,
        }


class FakeClient:
    def __init__(self) -> None:
        self.path = ""
        self.payload: dict[str, object] = {}

    def post(self, path: str, *, json: dict[str, object]) -> FakeResponse:
        self.path = path
        self.payload = json
        return FakeResponse()


def test_ollama_generator_sends_non_streaming_grounded_request() -> None:
    client = FakeClient()
    generator = OllamaAnswerGenerator(model="qwen3:4b-instruct-2507-q4_K_M", client=client)

    result = generator.generate(
        "How much parental leave is available?",
        [
            {
                "source_name": "handbook.md",
                "page_number": None,
                "text": "Employees receive 16 weeks of paid parental leave.",
            }
        ],
    )

    assert client.path == "/api/generate"
    assert client.payload["stream"] is False
    assert client.payload["think"] is False
    assert "<context>" in str(client.payload["prompt"])
    assert result.model == "qwen3:4b-instruct-2507-q4_K_M"
    assert result.input_tokens == 140
    assert result.output_tokens == 15
