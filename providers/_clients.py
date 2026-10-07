"""Concrete provider clients (OpenAI, Anthropic, Gemini) over plain ``requests``.

Each implements chat + SSE-style streaming against the provider's HTTP API.
Keys are read from constructor args or the standard environment variables.
Only ``requests`` is required — no provider SDKs.
"""
from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Iterator, List

import requests

from .base import ChatMessage, ChatResponse, LLMProvider, ProviderError

log = logging.getLogger(__name__)
_TIMEOUT = 60


def _sse_text_lines(resp: requests.Response):
    """Yield the payload of each ``data:`` line in an SSE stream."""
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if line.startswith("data:"):
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            yield payload


class OpenAIProvider(LLMProvider):
    """OpenAI-compatible chat completions (works with any OpenAI-style endpoint)."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini",
                 base_url: str = "https://api.openai.com/v1") -> None:
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self.model = model or os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        self.base_url = base_url.rstrip("/")

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _payload(self, messages: List[ChatMessage]) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        if not self.api_key:
            raise ProviderError("openai: missing API key")
        start = time.time()
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              headers=self._headers(),
                              json={"model": self.model, "messages": self._payload(messages),
                                    "stream": False, **kwargs},
                              timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"openai request failed: {exc}") from exc
        data = r.json()
        try:
            choice = data["choices"][0]
            content = choice["message"]["content"] or ""
            usage = data.get("usage", {})
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"openai: unexpected response shape: {exc}") from exc
        return self._fill_usage(ChatResponse(
            content=content, provider=self.name, model=self.model,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_s=time.time() - start), messages)

    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        if not self.api_key:
            raise ProviderError("openai: missing API key")
        try:
            r = requests.post(f"{self.base_url}/chat/completions",
                              headers=self._headers(),
                              json={"model": self.model, "messages": self._payload(messages),
                                    "stream": True, **kwargs},
                              timeout=_TIMEOUT, stream=True)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"openai stream failed: {exc}") from exc
        for payload in _sse_text_lines(r):
            try:
                delta = json.loads(payload)["choices"][0]["delta"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            if delta.get("content"):
                yield delta["content"]


class AnthropicProvider(LLMProvider):
    """Anthropic Messages API (system prompt extracted per API contract)."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-3-5-haiku-latest") -> None:
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", "claude-3-5-haiku-latest")

    def _split(self, messages: List[ChatMessage]):
        system = "\n".join(m.content for m in messages if m.role == "system")
        rest = [{"role": "user" if m.role in ("user", "tool") else "assistant", "content": m.content}
                for m in messages if m.role != "system"]
        return system, rest

    def _headers(self) -> Dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": "2023-06-01",
                "Content-Type": "application/json"}

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        if not self.api_key:
            raise ProviderError("anthropic: missing API key")
        system, rest = self._split(messages)
        start = time.time()
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", headers=self._headers(),
                              json={"model": self.model, "max_tokens": 1024, "system": system,
                                    "messages": rest, "stream": False, **kwargs},
                              timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"anthropic request failed: {exc}") from exc
        data = r.json()
        try:
            content = "".join(b.get("text", "") for b in data.get("content", []))
            usage = data.get("usage", {})
        except (AttributeError, TypeError) as exc:
            raise ProviderError(f"anthropic: unexpected response shape: {exc}") from exc
        return self._fill_usage(ChatResponse(
            content=content, provider=self.name, model=self.model,
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            latency_s=time.time() - start), messages)

    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        if not self.api_key:
            raise ProviderError("anthropic: missing API key")
        system, rest = self._split(messages)
        try:
            r = requests.post("https://api.anthropic.com/v1/messages", headers=self._headers(),
                              json={"model": self.model, "max_tokens": 1024, "system": system,
                                    "messages": rest, "stream": True, **kwargs},
                              timeout=_TIMEOUT, stream=True)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"anthropic stream failed: {exc}") from exc
        for raw in r.iter_lines(decode_unicode=True):
            if not raw or not raw.startswith("data:"):
                continue
            try:
                evt = json.loads(raw[5:].strip())
            except json.JSONDecodeError:
                continue
            if evt.get("type") == "content_block_delta":
                text = evt.get("delta", {}).get("text")
                if text:
                    yield text


class GeminiProvider(LLMProvider):
    """Google Gemini generateContent API (roles mapped to user/model)."""

    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str = "gemini-2.0-flash") -> None:
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY", "")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")

    def _contents(self, messages: List[ChatMessage]):
        contents = []
        for m in messages:
            role = "model" if m.role == "assistant" else "user"
            text = f"[system] {m.content}" if m.role == "system" else m.content
            contents.append({"role": role, "parts": [{"text": text}]})
        return contents

    def _url(self, stream: bool) -> str:
        if not self.api_key:
            raise ProviderError("gemini: missing API key")
        action = "streamGenerateContent?alt=sse" if stream else "generateContent"
        return (f"https://generativelanguage.googleapis.com/v1beta/models/"
                f"{self.model}:{action}?key={self.api_key}")

    def chat(self, messages: List[ChatMessage], **kwargs) -> ChatResponse:
        start = time.time()
        try:
            r = requests.post(self._url(False), json={"contents": self._contents(messages), **kwargs},
                              timeout=_TIMEOUT)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"gemini request failed: {exc}") from exc
        try:
            parts = r.json()["candidates"][0]["content"]["parts"]
            content = "".join(p.get("text", "") for p in parts)
        except (KeyError, IndexError) as exc:
            raise ProviderError(f"gemini: unexpected response shape: {exc}") from exc
        return self._fill_usage(ChatResponse(
            content=content, provider=self.name, model=self.model,
            latency_s=time.time() - start), messages)

    def stream(self, messages: List[ChatMessage], **kwargs) -> Iterator[str]:
        try:
            r = requests.post(self._url(True), json={"contents": self._contents(messages), **kwargs},
                              timeout=_TIMEOUT, stream=True)
            r.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(f"gemini stream failed: {exc}") from exc
        for payload in _sse_text_lines(r):
            try:
                parts = json.loads(payload)["candidates"][0]["content"]["parts"]
            except (KeyError, IndexError, json.JSONDecodeError):
                continue
            for p in parts:
                if p.get("text"):
                    yield p["text"]
