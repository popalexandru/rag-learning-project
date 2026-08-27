from asyncio import Lock
from dataclasses import dataclass
from uuid import uuid4

from app.schemas import MetadataValue, SearchResult
from app.vector_math import cosine_similarity


@dataclass(frozen=True)
class VectorRecord:
    id: str
    text: str
    embedding: list[float]
    metadata: dict[str, MetadataValue]


class InMemoryVectorStore:
    """Educational brute-force vector store; data disappears when the process stops."""

    def __init__(self) -> None:
        self._records: list[VectorRecord] = []
        self._lock = Lock()

    async def add(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadata: list[dict[str, MetadataValue]],
    ) -> list[str]:
        if not (len(texts) == len(embeddings) == len(metadata)):
            raise ValueError("Texts, embeddings, and metadata must have equal lengths")

        records = [
            VectorRecord(
                id=str(uuid4()),
                text=text,
                embedding=embedding,
                metadata=item_metadata,
            )
            for text, embedding, item_metadata in zip(
                texts, embeddings, metadata, strict=True
            )
        ]
        async with self._lock:
            self._records.extend(records)
        return [record.id for record in records]

    async def search(self, query_embedding: list[float], top_k: int) -> list[SearchResult]:
        async with self._lock:
            records_snapshot = list(self._records)

        ranked = sorted(
            (
                SearchResult(
                    id=record.id,
                    text=record.text,
                    metadata=record.metadata,
                    score=cosine_similarity(query_embedding, record.embedding),
                )
                for record in records_snapshot
            ),
            key=lambda result: result.score,
            reverse=True,
        )
        return ranked[:top_k]
