from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=10_000)


class ChatResponse(BaseModel):
    answer: str
    model: str


class EmbeddingRequest(BaseModel):
    text: str = Field(min_length=1, max_length=30_000)


class EmbeddingResponse(BaseModel):
    embedding: list[float]
    dimensions: int
    model: str
    tokens: int


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=30_000)
    text_b: str = Field(min_length=1, max_length=30_000)


class SimilarityResponse(BaseModel):
    similarity: float
    dimensions: int
    model: str
    tokens: int


class DocumentInput(BaseModel):
    text: str = Field(min_length=1, max_length=30_000)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class AddDocumentsRequest(BaseModel):
    documents: list[DocumentInput] = Field(min_length=1, max_length=100)


class AddDocumentsResponse(BaseModel):
    ids: list[str]
    stored_count: int
    dimensions: int
    model: str
    tokens: int


class UploadDocumentResponse(BaseModel):
    document_id: str
    filename: str
    ids: list[str]
    chunks_created: int
    chunk_size: int
    chunk_overlap: int
    dimensions: int
    model: str
    tokens: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=30_000)
    top_k: int = Field(default=3, ge=1, le=20)


class SearchResult(BaseModel):
    id: str
    text: str
    metadata: dict[str, MetadataValue]
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResult]
    model: str
    tokens: int


class RagRequest(BaseModel):
    question: str = Field(min_length=1, max_length=30_000)
    top_k: int = Field(default=3, ge=1, le=20)


class RagResponse(BaseModel):
    answer: str
    sources: list[SearchResult]
    model: str
    retrieval_tokens: int
    generation_tokens: int
