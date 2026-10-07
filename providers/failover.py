"""Multi-provider client: automatic failover, context-window management, usage/cost tracking.

Tries providers in order; on any :class:`ProviderError` it logs and moves to the
next one. Before each call, messages are trimmed to fit the provider's context
window. Every successful call records tokens and an estimated cost.

Cost figures are rough public-rate estimates — override ``cost_per_1k`` with
your contract rates for real accounting.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderError, estimate_tokens, messages_tokens

log = logging.getLogger(__name__)

#: Approximate context windows (tokens). Conservative on purpose.
CONTEXT_LIMITS: Dict[str, int] = {
    "openai": 128_000,
    "anthropic": 200_000,
    "gemini": 1_000_000,
    "mock": 32_000,
}

#: Rough USD cost per 1K tokens (input, output) — estimates, not quotes.
COST_PER_1K: Dict[str, tuple] = {
    "openai": (0.00015, 0.0006),
    "anthropic": (0.00025, 0.00125),
    "gemini": (0.0001, 0.0004),
    "mock": (0.0, 0.0),
}


@dataclass
class UsageTracker:
    """Accumulates tokens and estimated cost across calls and providers."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    by_provider: Dict[str, Dict[str, int]] = field(default_factory=dict)
    cost_per_1k: Dict[str, tuple] = field(default_factory=lambda: dict(COST_PER_1K))

    def record(self, resp: ChatResponse) -> None:
        self.prompt_tokens += resp.prompt_tokens
        self.completion_tokens += resp.completion_tokens
        self.calls += 1
        entry = self.by_provider.setdefault(resp.provider, {"prompt": 0, "completion": 0, "calls": 0})
        entry["prompt"] += resp.prompt_tokens
        entry["completion"] += resp.completion_tokens
        entry["calls"] += 1

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def estimated_cost_usd(self) -> float:
        total = 0.0
        for provider, entry in self.by_provider.items():
            in_rate, out_rate = self.cost_per_1k.get(provider, (0.0, 0.0))
            total += entry["prompt"] / 1000 * in_rate + entry["completion"] / 1000 * out_rate
        return total

    def summary(self) -> Dict[str, object]:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd(), 6),
            "by_provider": self.by_provider,
        }


class MultiProviderClient:
    """Failover wrapper around an ordered list of providers."""

    def __init__(self, providers: List[LLMProvider],
                 context_limits: Optional[Dict[str, int]] = None,
                 reserve_tokens: int = 2000) -> None:
        if not providers:
            raise ValueError("MultiProviderClient needs at least one provider")
        self.providers = providers
        self.context_limits = context_limits or dict(CONTEXT_LIMITS)
        self.reserve_tokens = reserve_tokens
        self.usage = UsageTracker()

    def _fit_context(self, provider: LLMProvider, messages: List[ChatMessage]) -> List[ChatMessage]:
        """Trim oldest non-system messages until the prompt fits the context window."""
        limit = self.context_limits.get(provider.name, 32_000) - self.reserve_tokens
        trimmed = list(messages)
        while len(trimmed) > 1 and messages_tokens(trimmed) > limit:
            # Never drop the system prompt (index 0) or the latest user message.
            drop_at = next((i for i, m in enumerate(trimmed[1:-1], start=1)
                            if m.role != "system"), None)
            if drop_at is None:
                break
            dropped = trimmed.pop(drop_at)
            log.debug("context trim: dropped %d tokens from a '%s' message",
                      estimate_tokens(dropped.content), dropped.role)
        return trimmed

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        """Try each provider in order; raise if all fail."""
        errors: List[str] = []
        for provider in self.providers:
            try:
                resp = provider.chat(self._fit_context(provider, messages), **kwargs)
                self.usage.record(resp)
                if provider is not self.providers[0]:
                    log.info("failover: '%s' served the request after %d failure(s)",
                             provider.name, len(errors))
                return resp
            except ProviderError as exc:
                log.warning("provider '%s' failed: %s — trying next", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise ProviderError(f"all {len(self.providers)} providers failed: {' | '.join(errors)}")

    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        """Stream from the first provider that accepts the request."""
        errors: List[str] = []
        for provider in self.providers:
            try:
                chunks = provider.stream(self._fit_context(provider, messages), **kwargs)
                first = True
                for chunk in chunks:
                    if first:
                        log.info("stream: serving from '%s'", provider.name)
                        first = False
                    yield chunk
                return
            except ProviderError as exc:
                log.warning("provider '%s' stream failed: %s — trying next", provider.name, exc)
                errors.append(f"{provider.name}: {exc}")
        raise ProviderError(f"all providers failed to stream: {' | '.join(errors)}")
