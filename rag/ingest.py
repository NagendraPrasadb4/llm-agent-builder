"""Document ingestion for the RAG pipeline (txt/md files)."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

log = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = {".txt", ".md", ".markdown"}


@dataclass
class Document:
    source: str  # file path / identifier used in citations
    text: str


def _strip_markdown(text: str) -> str:
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"(\*\*|__)(.*?)\1", r"\2", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)
    return text


def load_documents(directory: str, strip_md: bool = True) -> List[Document]:
    """Recursively load supported text documents from ``directory``."""
    root = Path(directory)
    if not root.is_dir():
        raise ValueError(f"not a directory: {directory}")
    docs: List[Document] = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError as exc:
                log.warning("skipping unreadable file %s: %s", path, exc)
                continue
            if strip_md and path.suffix.lower() != ".txt":
                text = _strip_markdown(text)
            text = text.strip()
            if text:
                docs.append(Document(source=str(path), text=text))
    log.info("ingested %d documents from %s", len(docs), directory)
    return docs
