"""Chunking strategies for the RAG pipeline.

``chunk_text`` splits hierarchically — paragraphs, then sentences, then words —
so chunks stay semantically coherent while respecting the size budget. A
configurable character overlap preserves context across chunk boundaries.
"""
from __future__ import annotations

import re
from typing import List


def _split_sentences(paragraph: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\s+", paragraph.strip())
    return [p for p in parts if p]


def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> List[str]:
    """Split ``text`` into overlapping chunks.

    Args:
        text: input text.
        chunk_size: target max characters per chunk.
        overlap: characters of overlap between consecutive chunks.

    Returns:
        List of chunk strings, in document order.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size // 2))
    text = text.strip()
    if not text:
        return []

    # Build atomic units: sentences (paragraphs split first for coherence).
    units: List[str] = []
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) <= chunk_size:
            units.append(paragraph)
        else:
            units.extend(_split_sentences(paragraph))

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for unit in units:
        # A single unit longer than chunk_size: hard-split on words.
        words = unit.split()
        pieces = [unit]
        if len(unit) > chunk_size:
            pieces, buf, blen = [], [], 0
            for w in words:
                if blen + len(w) + 1 > chunk_size and buf:
                    pieces.append(" ".join(buf)); buf, blen = [], 0
                buf.append(w); blen += len(w) + 1
            if buf:
                pieces.append(" ".join(buf))
        for piece in pieces:
            if current and current_len + len(piece) + 1 > chunk_size:
                chunks.append(" ".join(current))
                # Overlap: carry the tail of the finished chunk forward.
                tail, tail_len = [], 0
                for u in reversed(current):
                    if tail_len + len(u) >= overlap:
                        break
                    tail.insert(0, u); tail_len += len(u) + 1
                current, current_len = tail, tail_len
            current.append(piece)
            current_len += len(piece) + 1
    if current:
        chunks.append(" ".join(current))
    return chunks
