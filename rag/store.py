"""In-memory vector store with cosine-similarity ranking."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass
class ScoredChunk:
    source: str
    text: str
    score: float  # cosine similarity in [0, 1] for normalized vectors


def _cosine(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class VectorStore:
    """Minimal in-memory store: add embedded chunks, search by query vector."""

    def __init__(self) -> None:
        self._sources: List[str] = []
        self._texts: List[str] = []
        self._vectors: List[List[float]] = []

    def add(self, source: str, chunks: List[str], vectors: List[List[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must align")
        for chunk, vec in zip(chunks, vectors):
            self._sources.append(source)
            self._texts.append(chunk)
            self._vectors.append(vec)

    def __len__(self) -> int:
        return len(self._texts)

    def search(self, query_vector: List[float], top_k: int = 4) -> List[ScoredChunk]:
        scored = [
            ScoredChunk(source=s, text=t, score=_cosine(query_vector, v))
            for s, t, v in zip(self._sources, self._texts, self._vectors)
        ]
        scored.sort(key=lambda c: c.score, reverse=True)
        return scored[:max(1, top_k)]

    def clear(self) -> None:
        self._sources.clear(); self._texts.clear(); self._vectors.clear()
