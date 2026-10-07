"""Tests for providers: failover order, context trimming, usage/cost tracking."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from providers import (
    ChatMessage, LLMProvider, MockProvider, MultiProviderClient, ProviderError, UsageTracker,
)


class BoomProvider(LLMProvider):
    name = "boom"

    def chat(self, messages, **kw):
        raise ProviderError("boom: always fails")

    def stream(self, messages, **kw):
        raise ProviderError("boom: always fails")
        yield  # make it a generator


def msgs(n=3):
    return [ChatMessage(role="user", content=f"message {i}") for i in range(n)]


def test_failover_uses_next_provider_on_error():
    client = MultiProviderClient([BoomProvider(), MockProvider(['{"final": "ok"}'])])
    resp = client.chat(msgs())
    assert resp.provider == "mock"
    assert client.usage.by_provider["mock"]["calls"] == 1


def test_all_providers_failing_raises():
    client = MultiProviderClient([BoomProvider(), BoomProvider()])
    with pytest.raises(ProviderError, match="all 2 providers failed"):
        client.chat(msgs())


def test_context_trimming_keeps_system_and_latest():
    big = "x" * 4000
    messages = [ChatMessage(role="system", content="SYS"),
                ChatMessage(role="user", content=big),
                ChatMessage(role="assistant", content=big),
                ChatMessage(role="user", content="latest")]
    provider = MockProvider(['{"final": "ok"}'])
    client = MultiProviderClient([provider], context_limits={"mock": 1200}, reserve_tokens=100)
    client.chat(messages)
    sent = provider.calls[0]
    assert sent[0].role == "system" and sent[0].content == "SYS"
    assert sent[-1].content == "latest"


def test_usage_tracker_cost_math():
    u = UsageTracker()
    u.cost_per_1k["mock"] = (0.001, 0.002)
    from providers import ChatResponse
    u.record(ChatResponse(content="x", provider="mock", model="m",
                          prompt_tokens=1000, completion_tokens=1000))
    assert u.total_tokens == 2000
    assert abs(u.estimated_cost_usd() - 0.003) < 1e-9


def test_mock_stream_yields_words():
    chunks = list(MockProvider(["hello world"]).stream(msgs()))
    assert "".join(chunks).strip() == "hello world"


def test_mock_exhaustion_raises_provider_error():
    p = MockProvider(["only one"])
    p.chat(msgs())
    with pytest.raises(ProviderError, match="script exhausted"):
        p.chat(msgs())
