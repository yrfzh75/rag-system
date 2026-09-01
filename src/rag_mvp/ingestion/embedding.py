from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol


class Embedder(Protocol):
    @property
    def dimension(self) -> int: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


class FastEmbedder:
    """Local ONNX embeddings with an explicit, versioned model name."""

    def __init__(self, model_name: str, *, cache_dir: Path) -> None:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "Embedding support requires qdrant-client[fastembed]; run 'uv sync'"
            ) from exc
        self.model_name = model_name
        cache_dir.mkdir(parents=True, exist_ok=True)
        self._model = TextEmbedding(model_name=model_name, cache_dir=str(cache_dir))
        model_info = next(
            (
                item
                for item in TextEmbedding.list_supported_models()
                if item["model"] == model_name
            ),
            None,
        )
        if model_info is None:
            raise ValueError(f"Unsupported FastEmbed model: {model_name}")
        self._dimension = int(model_info["dim"])

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [vector.tolist() for vector in self._model.embed(list(texts))]
