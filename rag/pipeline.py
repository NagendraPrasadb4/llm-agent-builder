"""End-to-end RAG pipeline: ingest -> chunk -> embed -> retrieve -> grounded answer.

``RAGPipeline.query`` retrieves the top-k chunks, asks the LLM to answer using
ONLY those sources, and parses ``[n]`` citations back to source files — the
basis for measuring groundedness (hallucination rate) in eval/.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

from providers.base import ChatMessage
from providers.failover import MultiProviderClient
from .chunking import chunk_text
from .embeddings import Embedder, HashEmbedder
from .ingest import Document, load_documents
from .store import ScoredChunk, VectorStore

log = logging.getLogger(__name__)

RAG_PROMPT = """Answer the question using ONLY the sources below. If the sources do not contain the answer, say so explicitly.

Cite every factual claim with the source number in square brackets, e.g. [1], [2].

SOURCES:
{sources}

QUESTION: {question}

Answer with citations:"""


@dataclass
class RAGAnswer:
    answer: str
    citations: List[str]          # source files cited, in order
    retrieved: List[ScoredChunk] = field(default_factory=list)


class RAGPipeline:
    """Retrieval-augmented generation over a directory of txt/md documents."""

    def __init__(self, client: MultiProviderClient,
                 embedder: Optional[Embedder] = None,
                 chunk_size: int = 600, overlap: int = 80) -> None:
        self.client = client
        self.embedder = embedder or HashEmbedder()
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.store = VectorStore()
        self._built = False

    def build(self, directory: str) -> "RAGPipeline":
        """Ingest, chunk, and embed every document in ``directory``."""
        docs: List[Document] = load_documents(directory)
        for doc in docs:
            chunks = chunk_text(doc.text, self.chunk_size, self.overlap)
            if not chunks:
                continue
            vectors = self.embedder.embed(chunks)
            self.store.add(doc.source, chunks, vectors)
        self._built = True
        log.info("RAG index built: %d chunks from %d documents", len(self.store), len(docs))
        return self

    def retrieve(self, question: str, top_k: int = 4) -> List[ScoredChunk]:
        if not self._built:
            raise RuntimeError("call build() before retrieve()")
        qvec = self.embedder.embed([question])[0]
        return self.store.search(qvec, top_k=top_k)

    def query(self, question: str, top_k: int = 4) -> RAGAnswer:
        """Retrieve context and generate a citation-grounded answer."""
        hits = self.retrieve(question, top_k=top_k)
        sources_block = "\n\n".join(
            f"[{i + 1}] (source: {h.source})\n{h.text}" for i, h in enumerate(hits)
        )
        prompt = RAG_PROMPT.format(sources=sources_block, question=question)
        resp = self.client.chat([ChatMessage(role="user", content=prompt)])
        cited = self._parse_citations(resp.content, hits)
        return RAGAnswer(answer=resp.content, citations=cited, retrieved=hits)

    @staticmethod
    def _parse_citations(answer: str, hits: List[ScoredChunk]) -> List[str]:
        cited: List[str] = []
        for num in re.findall(r"\[(\d+)\]", answer):
            idx = int(num) - 1
            if 0 <= idx < len(hits) and hits[idx].source not in cited:
                cited.append(hits[idx].source)
        return cited
