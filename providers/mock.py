"""Deterministic mock provider for offline development, tests, demos, and eval.

``MockProvider`` replays a scripted list of responses in order, so agent
behaviour is fully reproducible without API keys or network access.
"""
from __future__ import annotations

import time
from typing import Iterator, List

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderError


class MockProvider(LLMProvider):
    """Replays scripted responses; raises when the script is exhausted."""

    name = "mock"

    def __init__(self, responses: List[str], model: str = "mock-1") -> None:
        if not responses:
            raise ValueError("MockProvider needs at least one scripted response")
        self._responses = list(responses)
        self._cursor = 0
        self.model = model
        self.calls: List[List[ChatMessage]] = []

    def _next(self) -> str:
        if self._cursor >= len(self._responses):
            raise ProviderError("mock: script exhausted — agent took more steps than scripted")
        text = self._responses[self._cursor]
        self._cursor += 1
        return text

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        self.calls.append(list(messages))
        start = time.time()
        content = self._next()
        return self._fill_usage(ChatResponse(
            content=content, provider=self.name, model=self.model,
            latency_s=time.time() - start), messages)

    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        self.calls.append(list(messages))
        for word in self._next().split(" "):
            yield word + " "
