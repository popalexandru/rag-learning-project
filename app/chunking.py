from dataclasses import dataclass


@dataclass(frozen=True)
class TextChunk:
    index: int
    text: str
    char_start: int
    char_end: int


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[TextChunk]:
    """Split text using transparent fixed-size character windows."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    normalized_text = text.strip()
    if not normalized_text:
        return []

    step = chunk_size - chunk_overlap
    chunks: list[TextChunk] = []
    for start in range(0, len(normalized_text), step):
        end = min(start + chunk_size, len(normalized_text))
        chunks.append(
            TextChunk(
                index=len(chunks),
                text=normalized_text[start:end],
                char_start=start,
                char_end=end,
            )
        )
        if end == len(normalized_text):
            break

    return chunks
