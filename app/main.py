from functools import lru_cache
from pathlib import Path
from typing import Annotated
from uuid import uuid4

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status

from app.chunking import chunk_text
from app.config import Settings, get_settings
from app.openai_service import OpenAIService
from app.schemas import (
    AddDocumentsRequest,
    AddDocumentsResponse,
    ChatRequest,
    ChatResponse,
    EmbeddingRequest,
    EmbeddingResponse,
    RagRequest,
    RagResponse,
    SearchRequest,
    SearchResponse,
    SimilarityRequest,
    SimilarityResponse,
    UploadDocumentResponse,
)
from app.vector_math import cosine_similarity
from app.vector_store import InMemoryVectorStore

app = FastAPI(
    title="RAG Learning Project",
    description="Step-by-step foundation for embeddings, vector search, and RAG.",
    version="0.1.0",
)

MAX_UPLOAD_BYTES = 2 * 1024 * 1024
SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


@app.get("/")
async def root() -> dict[str, str]:
    return {
        "name": "RAG Learning Project",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


def get_openai_service(settings: Settings = Depends(get_settings)) -> OpenAIService:
    if settings.openai_api_key is None or not settings.openai_api_key.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENAI_API_KEY is not configured. Add it to the local .env file.",
        )

    return OpenAIService(
        api_key=settings.openai_api_key.get_secret_value(),
        model=settings.openai_model,
        embedding_model=settings.openai_embedding_model,
    )


@lru_cache
def get_vector_store() -> InMemoryVectorStore:
    return InMemoryVectorStore()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
) -> ChatResponse:
    answer = await service.answer(request.message)
    return ChatResponse(answer=answer, model=settings.openai_model)


@app.post("/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    request: EmbeddingRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
) -> EmbeddingResponse:
    vectors, tokens = await service.embed([request.text])
    vector = vectors[0]
    return EmbeddingResponse(
        embedding=vector,
        dimensions=len(vector),
        model=settings.openai_embedding_model,
        tokens=tokens,
    )


@app.post("/similarity", response_model=SimilarityResponse)
async def compare_texts(
    request: SimilarityRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
) -> SimilarityResponse:
    vectors, tokens = await service.embed([request.text_a, request.text_b])
    return SimilarityResponse(
        similarity=cosine_similarity(vectors[0], vectors[1]),
        dimensions=len(vectors[0]),
        model=settings.openai_embedding_model,
        tokens=tokens,
    )


@app.post("/documents", response_model=AddDocumentsResponse)
async def add_documents(
    request: AddDocumentsRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
    store: InMemoryVectorStore = Depends(get_vector_store),
) -> AddDocumentsResponse:
    texts = [document.text for document in request.documents]
    metadata = [document.metadata for document in request.documents]
    embeddings, tokens = await service.embed(texts)
    ids = await store.add(texts, embeddings, metadata)

    return AddDocumentsResponse(
        ids=ids,
        stored_count=len(ids),
        dimensions=len(embeddings[0]),
        model=settings.openai_embedding_model,
        tokens=tokens,
    )


@app.post("/documents/upload", response_model=UploadDocumentResponse)
async def upload_document(
    file: Annotated[UploadFile, File(description="A UTF-8 .txt or .md file")],
    chunk_size: Annotated[int, Form(ge=100, le=5_000)] = 1_000,
    chunk_overlap: Annotated[int, Form(ge=0, le=1_000)] = 200,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
    store: InMemoryVectorStore = Depends(get_vector_store),
) -> UploadDocumentResponse:
    filename = Path(file.filename or "").name
    if Path(filename).suffix.lower() not in SUPPORTED_TEXT_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only .txt and .md files are supported.",
        )
    if chunk_overlap >= chunk_size:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="chunk_overlap must be smaller than chunk_size.",
        )

    try:
        raw_content = await file.read(MAX_UPLOAD_BYTES + 1)
    finally:
        await file.close()

    if len(raw_content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="The maximum upload size is 2 MiB.",
        )
    try:
        text = raw_content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The file must contain valid UTF-8 text.",
        ) from error

    chunks = chunk_text(text, chunk_size, chunk_overlap)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file contains no text.",
        )

    document_id = str(uuid4())
    texts = [chunk.text for chunk in chunks]
    metadata = [
        {
            "document_id": document_id,
            "filename": filename,
            "content_type": file.content_type or "text/plain",
            "chunk_index": chunk.index,
            "char_start": chunk.char_start,
            "char_end": chunk.char_end,
        }
        for chunk in chunks
    ]
    embeddings, tokens = await service.embed(texts)
    ids = await store.add(texts, embeddings, metadata)

    return UploadDocumentResponse(
        document_id=document_id,
        filename=filename,
        ids=ids,
        chunks_created=len(chunks),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        dimensions=len(embeddings[0]),
        model=settings.openai_embedding_model,
        tokens=tokens,
    )


@app.post("/search", response_model=SearchResponse)
async def search_documents(
    request: SearchRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
    store: InMemoryVectorStore = Depends(get_vector_store),
) -> SearchResponse:
    embeddings, tokens = await service.embed([request.query])
    results = await store.search(embeddings[0], request.top_k)
    return SearchResponse(
        results=results,
        model=settings.openai_embedding_model,
        tokens=tokens,
    )


@app.post("/rag/ask", response_model=RagResponse)
async def rag_ask(
    request: RagRequest,
    settings: Settings = Depends(get_settings),
    service: OpenAIService = Depends(get_openai_service),
    store: InMemoryVectorStore = Depends(get_vector_store),
) -> RagResponse:
    query_embeddings, retrieval_tokens = await service.embed([request.question])
    sources = await store.search(query_embeddings[0], request.top_k)

    if not sources:
        return RagResponse(
            answer="I do not know because no documents are indexed.",
            sources=[],
            model=settings.openai_model,
            retrieval_tokens=retrieval_tokens,
            generation_tokens=0,
        )

    context_blocks = [
        f"[Source {index}]\n{source.text}"
        for index, source in enumerate(sources, start=1)
    ]
    answer, generation_tokens = await service.answer_from_context(
        question=request.question,
        context_blocks=context_blocks,
    )
    return RagResponse(
        answer=answer,
        sources=sources,
        model=settings.openai_model,
        retrieval_tokens=retrieval_tokens,
        generation_tokens=generation_tokens,
    )
