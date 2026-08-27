from fastapi.testclient import TestClient

from app.chunking import chunk_text
from app.config import Settings, get_settings
from app.main import app, get_openai_service, get_vector_store
from app.vector_math import cosine_similarity
from app.vector_store import InMemoryVectorStore

client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_root_links_to_docs() -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_chunk_text_uses_overlapping_windows() -> None:
    chunks = chunk_text("abcdefghij", chunk_size=6, chunk_overlap=2)

    assert [chunk.text for chunk in chunks] == ["abcdef", "efghij"]
    assert [(chunk.char_start, chunk.char_end) for chunk in chunks] == [(0, 6), (4, 10)]


def test_chat_explains_when_key_is_missing() -> None:
    app.dependency_overrides[get_settings] = lambda: Settings(openai_api_key=None)
    try:
        response = client.post("/chat", json={"message": "Hello"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["detail"]


class FakeOpenAIService:
    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        assert texts == ["cat", "kitten"]
        return [[1.0, 0.0], [0.8, 0.6]], 2


def test_similarity_endpoint_uses_embeddings() -> None:
    app.dependency_overrides[get_openai_service] = lambda: FakeOpenAIService()
    try:
        response = client.post("/similarity", json={"text_a": "cat", "text_b": "kitten"})
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "similarity": 0.8,
        "dimensions": 2,
        "model": "text-embedding-3-small",
        "tokens": 2,
    }


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


class FakeSearchService:
    vectors = {
        "Cats sleep for many hours.": [1.0, 0.0],
        "PostgreSQL stores relational data.": [0.0, 1.0],
        "How do cats rest?": [0.9, 0.1],
        "How long do cats sleep?": [0.95, 0.05],
    }

    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [self.vectors[text] for text in texts], len(texts)

    async def answer_from_context(
        self,
        question: str,
        context_blocks: list[str],
    ) -> tuple[str, int]:
        assert question == "How long do cats sleep?"
        assert context_blocks[0] == "[Source 1]\nCats sleep for many hours."
        return "Cats sleep for many hours. [Source 1]", 12


def test_documents_can_be_indexed_and_searched() -> None:
    store = InMemoryVectorStore()
    app.dependency_overrides[get_openai_service] = lambda: FakeSearchService()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        add_response = client.post(
            "/documents",
            json={
                "documents": [
                    {"text": "Cats sleep for many hours.", "metadata": {"topic": "animals"}},
                    {
                        "text": "PostgreSQL stores relational data.",
                        "metadata": {"topic": "databases"},
                    },
                ]
            },
        )
        search_response = client.post(
            "/search",
            json={"query": "How do cats rest?", "top_k": 2},
        )
    finally:
        app.dependency_overrides.clear()

    assert add_response.status_code == 200
    assert add_response.json()["stored_count"] == 2
    assert search_response.status_code == 200
    results = search_response.json()["results"]
    assert len(results) == 2
    assert results[0]["text"] == "Cats sleep for many hours."
    assert results[0]["metadata"] == {"topic": "animals"}
    assert results[0]["score"] > results[1]["score"]


def test_rag_retrieves_context_and_generates_grounded_answer() -> None:
    store = InMemoryVectorStore()
    app.dependency_overrides[get_openai_service] = lambda: FakeSearchService()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        client.post(
            "/documents",
            json={
                "documents": [
                    {"text": "Cats sleep for many hours.", "metadata": {"source": "cats.md"}},
                    {
                        "text": "PostgreSQL stores relational data.",
                        "metadata": {"source": "postgres.md"},
                    },
                ]
            },
        )
        response = client.post(
            "/rag/ask",
            json={"question": "How long do cats sleep?", "top_k": 1},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "Cats sleep for many hours. [Source 1]"
    assert body["sources"][0]["text"] == "Cats sleep for many hours."
    assert body["sources"][0]["metadata"] == {"source": "cats.md"}
    assert body["retrieval_tokens"] == 1
    assert body["generation_tokens"] == 12


def test_rag_does_not_call_generation_when_store_is_empty() -> None:
    store = InMemoryVectorStore()
    app.dependency_overrides[get_openai_service] = lambda: FakeSearchService()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        response = client.post(
            "/rag/ask",
            json={"question": "How long do cats sleep?", "top_k": 1},
        )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["sources"] == []
    assert response.json()["generation_tokens"] == 0


class FakeUploadService:
    async def embed(self, texts: list[str]) -> tuple[list[list[float]], int]:
        return [[1.0, 0.0] for _ in texts], len(texts)


def test_text_file_is_chunked_indexed_and_searchable() -> None:
    store = InMemoryVectorStore()
    app.dependency_overrides[get_openai_service] = lambda: FakeUploadService()
    app.dependency_overrides[get_vector_store] = lambda: store
    try:
        upload_response = client.post(
            "/documents/upload",
            files={"file": ("guide.md", b"A" * 150, "text/markdown")},
            data={"chunk_size": "100", "chunk_overlap": "20"},
        )
        search_response = client.post(
            "/search",
            json={"query": "guide", "top_k": 2},
        )
    finally:
        app.dependency_overrides.clear()

    assert upload_response.status_code == 200
    assert upload_response.json()["chunks_created"] == 2
    assert upload_response.json()["filename"] == "guide.md"
    results = search_response.json()["results"]
    assert len(results) == 2
    assert results[0]["metadata"]["filename"] == "guide.md"
    assert results[0]["metadata"]["chunk_index"] == 0


def test_upload_rejects_unsupported_file_type() -> None:
    response = client.post(
        "/documents/upload",
        files={"file": ("guide.pdf", b"not-a-pdf", "application/pdf")},
    )

    assert response.status_code == 415
