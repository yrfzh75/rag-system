from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from rag_mvp.ingestion.models import Chunk


class QdrantChunkStore:
    """Persistent local Qdrant collection for dense chunk vectors."""

    def __init__(self, *, path: Path, collection: str, vector_size: int) -> None:
        from qdrant_client import QdrantClient, models

        path.parent.mkdir(parents=True, exist_ok=True)
        self.client = QdrantClient(path=str(path))
        self.collection = collection
        self.models = models
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            existing_size = self.client.get_collection(collection).config.params.vectors.size
            if existing_size != vector_size:
                raise ValueError(
                    f"Collection '{collection}' has vector size {existing_size}; expected {vector_size}"
                )

    def replace_document(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("Each chunk must have exactly one vector")

        document_id = chunks[0].document_id
        self.client.delete(
            collection_name=self.collection,
            points_selector=self.models.FilterSelector(
                filter=self.models.Filter(
                    must=[
                        self.models.FieldCondition(
                            key="document_id",
                            match=self.models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
            wait=True,
        )
        self.client.upsert(
            collection_name=self.collection,
            points=[
                self.models.PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.payload())
                for chunk, vector in zip(chunks, vectors, strict=True)
            ],
            wait=True,
        )

    def search(
        self,
        vector: list[float],
        *,
        top_k: int,
        score_threshold: float | None = None,
    ) -> list[dict[str, object]]:
        """Return the highest-scoring chunk payloads for one dense query vector."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=top_k,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
        )
        return [{"score": float(point.score), **(point.payload or {})} for point in response.points]

    def list_chunks(self) -> list[dict[str, object]]:
        """Return all stored chunk payloads for local lexical indexing."""
        chunks: list[dict[str, object]] = []
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.collection,
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            chunks.extend(dict(point.payload or {}) for point in points)
            if offset is None:
                return chunks

    def close(self) -> None:
        self.client.close()
