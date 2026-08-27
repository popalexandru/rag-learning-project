from math import sqrt


def cosine_similarity(vector_a: list[float], vector_b: list[float]) -> float:
    """Measure vector direction similarity, with a result between -1 and 1."""
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions")
    if not vector_a:
        raise ValueError("Vectors must not be empty")

    dot_product = sum(a * b for a, b in zip(vector_a, vector_b, strict=True))
    magnitude_a = sqrt(sum(value * value for value in vector_a))
    magnitude_b = sqrt(sum(value * value for value in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity is undefined for a zero vector")

    return dot_product / (magnitude_a * magnitude_b)
