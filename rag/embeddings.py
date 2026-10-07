"""Pluggable embedding interface with a lightweight local baseline.

:class:`Embedder` is the contract — implement it to plug in OpenAI embeddings,
sentence-transformers, or any other model. :class:`HashEmbedder` is a
deterministic, dependency-free baseline (character-trigram hashing into a
fixed-dimension vector, L2-normalized) that works fully offline. It is honest
about being a baseline: fine for keyword-ish similarity in this repo's demos
and tests, not a substitute for a real embedding model in production.
"""
from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from typing import List


class Embedder(ABC):
    """Contract for text embedding models."""

    @abstractmethod
    def embed(self, texts: List[str]) -> List[List[float]]:
        """Return one L2-normalized vector per input text."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


def _normalize(vec: List[float]) -> List[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


class HashEmbedder(Embedder):
    """Deterministic trigram-hash embeddings (offline baseline).

    Each character trigram of the lowercased text hashes into one of ``dim``
    buckets; counts are L2-normalized. Similar texts share trigrams, so cosine
    similarity approximates textual overlap. No downloads, no API calls.
    """

    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def _trigrams(self, text: str):
        text = re.sub(r"\s+", " ", text.lower()).strip()
        padded = f"  {text}  "
        for i in range(len(padded) - 2):
            yield padded[i:i + 3]

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = []
        for text in texts:
            vec = [0.0] * self._dim
            for tri in self._trigrams(text):
                bucket = int(hashlib.md5(tri.encode()).hexdigest(), 16) % self._dim
                vec[bucket] += 1.0
            vectors.append(_normalize(vec))
        return vectors


# Example of how to plug in a real model (requires sentence-transformers):
#
#   class SentenceTransformerEmbedder(Embedder):
#       def __init__(self, model_name="all-MiniLM-L6-v2"):
#           from sentence_transformers import SentenceTransformer
#           self._model = SentenceTransformer(model_name)
#       @property
#       def dimension(self): return self._model.get_sentence_embedding_dimension()
#       def embed(self, texts): return [list(map(float, v)) for v in self._model.encode(texts, normalize_embeddings=True)]
