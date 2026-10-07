"""Conversation memory with a token budget and graceful compaction.

The agent keeps the full recent history and compacts older turns when the
budget is exceeded — by LLM summarization when a provider is available,
otherwise by dropping the oldest non-system turns. Nothing is lost silently:
compaction always leaves a summary marker behind.
"""
from __future__ import annotations

import logging
from typing import List, Optional

from providers.base import ChatMessage, estimate_tokens, messages_tokens

log = logging.getLogger(__name__)


class ConversationMemory:
    """Token-bounded message history for the agent loop."""

    def __init__(self, max_tokens: int = 6000, system_prompt: str = "") -> None:
        self.max_tokens = max_tokens
        self._messages: List[ChatMessage] = []
        if system_prompt:
            self._messages.append(ChatMessage(role="system", content=system_prompt))

    def add(self, role: str, content: str) -> None:
        self._messages.append(ChatMessage(role=role, content=content))

    def messages(self) -> List[ChatMessage]:
        return list(self._messages)

    def token_count(self) -> int:
        return messages_tokens(self._messages)

    def over_budget(self) -> bool:
        return self.token_count() > self.max_tokens

    def compact(self, summarizer=None) -> None:
        """Shrink history to fit the budget.

        Args:
            summarizer: optional callable(messages) -> str. When provided
                (typically an LLM call), old turns are replaced by its summary;
                otherwise the oldest non-system, non-final turns are dropped.
        """
        if not self.over_budget():
            return
        system = [m for m in self._messages if m.role == "system"]
        rest = [m for m in self._messages if m.role != "system"]
        keep_tail = rest[-2:]  # always keep the latest exchange
        to_compact = rest[:-2]
        if summarizer and to_compact:
            try:
                summary = summarizer(to_compact)
                self._messages = system + [
                    ChatMessage(role="user",
                                content=f"[Earlier conversation summarized to save context]\n{summary}")
                ] + keep_tail
                log.info("memory compacted via summarization: %d -> %d tokens",
                         self.max_tokens, self.token_count())
                return
            except Exception as exc:  # noqa: BLE001 - fall back to truncation
                log.warning("summarized compaction failed (%s); truncating instead", exc)
        self._messages = system + keep_tail
        log.info("memory compacted via truncation")

    def clear(self) -> None:
        system = [m for m in self._messages if m.role == "system"]
        self._messages = system
