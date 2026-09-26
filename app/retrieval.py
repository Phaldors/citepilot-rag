from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

from dataclasses import dataclass

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

client = OpenAI()


def embed_text(text: str) -> list[float]:
    response = client.embeddings.create(model="text-embedding-3-small", input=text)
    return response.data[0].embedding


@dataclass(frozen=True)
class Chunk:
    document_name: str
    page: int
    text: str
    embedding: list[float]


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
        chunks.append(Chunk(document_name=document_name, page=page, text=clean[start:end], embedding=embed_text(clean[start:end])))
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
        query_vector = embed_text(query)
        chunk_vectors = [chunk.embedding for chunk in self.chunks]
        scores = cosine_similarity([query_vector], chunk_vectors).flatten()
        ranked = scores.argsort()[::-1][:limit]
        return [(self.chunks[index], float(scores[index])) for index in ranked if scores[index] > 0]


def synthesize_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    sources = "\n\n".join(f"[{chunk.document_name}, page {chunk.page}] {chunk.text}" for chunk, _ in results)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer using only the provided sources. Cite each claim like [document, page]. "
                    "If the sources don't contain the answer, say so. Always answer in the same "
                    "language as the question, regardless of the sources' or your own language."
                ),
            },
            {"role": "user", "content": f"Sources:\n{sources}\n\nQuestion: {question}"},
        ],
    )
    return response.choices[0].message.content