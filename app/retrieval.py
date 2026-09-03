from __future__ import annotations

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class Chunk:
    document_name: str
    page: int
    text: str


def chunk_text(text: str, document_name: str, page: int, size: int = 700, overlap: int = 120) -> list[Chunk]:
    clean = " ".join(text.split())
    if not clean:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(clean):
        end = min(len(clean), start + size)
        if end < len(clean):
            boundary = clean.rfind(" ", start, end)
            if boundary > start + (size // 2):
                end = boundary
        chunks.append(Chunk(document_name=document_name, page=page, text=clean[start:end]))
        if end == len(clean):
            break
        start = max(end - overlap, start + 1)
    return chunks


class Retriever:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def add_chunks(self, chunks: list[Chunk]) -> None:
        self.chunks.extend(chunks)

    def search(self, query: str, limit: int = 3) -> list[tuple[Chunk, float]]:
        if not query.strip() or not self.chunks:
            return []
        corpus = [chunk.text for chunk in self.chunks]
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform(corpus + [query])
        scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
        ranked = scores.argsort()[::-1][:limit]
        return [(self.chunks[index], float(scores[index])) for index in ranked if scores[index] > 0]
