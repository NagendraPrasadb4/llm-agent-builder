"""Tests for the RAG pipeline: chunking, embeddings, search ranking, citations."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providers import MockProvider, MultiProviderClient
from rag import HashEmbedder, RAGPipeline, VectorStore, chunk_text, load_documents


def test_chunking_respects_size_and_overlap():
    text = " ".join(f"Sentence number {i} about agents." for i in range(50))
    chunks = chunk_text(text, chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(len(c) <= 260 for c in chunks)  # size + small tolerance
    # overlap: some content shared between consecutive chunks
    assert any(w in chunks[1] for w in chunks[0].split()[-5:])


def test_chunking_empty():
    assert chunk_text("   ") == []


def test_hash_embedder_deterministic_and_normalized():
    e = HashEmbedder(dim=64)
    a, b = e.embed(["hello world", "hello world"])[0], e.embed(["hello world"])[0]
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-6


def test_vector_store_ranks_relevant_chunk_first():
    e = HashEmbedder()
    store = VectorStore()
    texts = ["the agent loop detects repeated tool calls",
             "completely unrelated text about gardening and soil"]
    store.add("s", texts, e.embed(texts))
    hits = store.search(e.embed(["repeated tool calls"])[0], top_k=2)
    assert hits[0].text.startswith("the agent loop")


def test_ingest_loads_sample_docs():
    docs = load_documents(str(ROOT / "data" / "sample_docs"))
    assert len(docs) >= 3
    assert all(d.source and d.text for d in docs)


def test_rag_query_cites_retrieved_source():
    client = MultiProviderClient([MockProvider(
        ["The agent uses fingerprints of tool calls to detect loops [1]."])])
    pipe = RAGPipeline(client).build(str(ROOT / "data" / "sample_docs"))
    ans = pipe.query("How does the agent detect repeated tool calls?", top_k=3)
    assert ans.citations, "expected at least one parsed citation"
    assert any("agent_architecture" in c for c in ans.citations)


def test_rag_retrieve_before_build_raises():
    client = MultiProviderClient([MockProvider(["x"])])
    with pytest.raises(RuntimeError):
        RAGPipeline(client).retrieve("q")
