"""Provider-agnostic LLM interfaces.

Every provider implements :class:`LLMProvider` with two verbs:

- :meth:`chat` — one full completion plus usage accounting.
- :meth:`stream` — SSE-style token-by-token delivery as an iterator of strings.

Token counts fall back to a ``chars/4`` heuristic when a provider does not
report usage; swap in ``tiktoken`` for exact counts (see requirements.txt).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class ChatMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass
class ChatResponse:
    content: str
    provider: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class ProviderError(Exception):
    """Raised when a provider call fails (auth, rate limit, network, bad response)."""


def estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4). Honest heuristic — use tiktoken for exactness."""
    return max(1, len(text) // 4)


def messages_tokens(messages: List[ChatMessage]) -> int:
    return sum(estimate_tokens(m.role) + estimate_tokens(m.content) + 4 for m in messages)


class LLMProvider(ABC):
    """Interface every LLM provider must implement."""

    name: str = "provider"
    model: str = ""

    @abstractmethod
    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Return one full completion."""

    @abstractmethod
    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        """Yield content chunks token-by-token (SSE-style)."""

    def _fill_usage(self, resp: ChatResponse, messages: List[ChatMessage]) -> ChatResponse:
        if resp.prompt_tokens == 0:
            resp.prompt_tokens = messages_tokens(messages)
        if resp.completion_tokens == 0:
            resp.completion_tokens = estimate_tokens(resp.content)
        return resp
