"""Built-in tools for the agent's MCP-style ecosystem.

- :class:`CalculatorTool` — safe arithmetic evaluation (AST-whitelisted, no eval of
  arbitrary code).
- :class:`FileReaderTool` — sandboxed file reads confined to an allowed directory.
- :class:`WebSearchStubTool` — clearly-marked stub; wire to a real search API
  (Tavily/Brave/Serper) for production use.
- :class:`RestApiCallerTool` — HTTP(S) GET/POST with timeouts and size caps.
"""
from __future__ import annotations

import ast
import json
import logging
import math
from pathlib import Path
from typing import Any

import requests

from .base import Tool, ToolError, ToolResult

log = logging.getLogger(__name__)


class CalculatorTool(Tool):
    """Evaluate a math expression safely via an AST whitelist."""

    name = "calculator"
    description = (
        "Evaluate an arithmetic expression and return the numeric result. "
        "Supports + - * / // % **, parentheses, and functions: sqrt, sin, cos, "
        "tan, log, exp, abs, round, pow, min, max, pi, e."
    )
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string", "description": "e.g. '15 * 240 / 100'"}},
        "required": ["expression"],
    }

    _SAFE_FUNCS = {
        "sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan,
        "log": math.log, "exp": math.exp, "abs": abs, "round": round,
        "pow": pow, "min": min, "max": max, "pi": math.pi, "e": math.e,
    }
    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load,
        ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod,
        ast.FloorDiv, ast.USub, ast.UAdd,
    )

    def _check(self, node: ast.AST) -> None:
        if not isinstance(node, self._ALLOWED_NODES):
            raise ToolError(f"expression uses disallowed syntax: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in self._SAFE_FUNCS:
                raise ToolError(f"disallowed function call in expression")
        if isinstance(node, ast.Name) and node.id not in self._SAFE_FUNCS:
            raise ToolError(f"disallowed name in expression: '{node.id}'")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ToolError("only numeric constants are allowed")
        for child in ast.iter_child_nodes(node):
            self._check(child)

    def execute(self, expression: str) -> ToolResult:
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError as exc:
            return ToolResult(ok=False, output=f"Invalid expression: {exc}", error=str(exc))
        try:
            self._check(tree)
            value = eval(  # noqa: S307 - AST-whitelisted above; no builtins exposed
                compile(tree, "<calculator>", "eval"), {"__builtins__": {}}, dict(self._SAFE_FUNCS)
            )
        except ToolError as exc:
            return ToolResult(ok=False, output=f"Unsafe expression: {exc}", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - e.g. ZeroDivisionError
            return ToolResult(ok=False, output=f"Evaluation failed: {exc}", error=str(exc))
        return ToolResult(ok=True, output=str(value), data=value)


class FileReaderTool(Tool):
    """Read a text file, confined to an allowed root directory (path-traversal safe)."""

    name = "file_reader"
    description = (
        "Read a UTF-8 text file and return its contents (truncated to max_chars). "
        "Paths are confined to the allowed root directory; '..' escapes are rejected."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "File path, relative to the allowed root"},
            "max_chars": {"type": "integer", "description": "Max characters to return (default 4000)"},
        },
        "required": ["path"],
    }

    def __init__(self, allowed_root: str = ".") -> None:
        self.allowed_root = Path(allowed_root).resolve()

    def execute(self, path: str, max_chars: int = 4000) -> ToolResult:
        try:
            target = (self.allowed_root / path).resolve()
        except Exception as exc:  # noqa: BLE001
            return ToolResult(ok=False, output=f"Bad path: {exc}", error=str(exc))
        if target != self.allowed_root and self.allowed_root not in target.parents:
            return ToolResult(ok=False, output="Access denied: path is outside the allowed directory.",
                              error="path_traversal")
        if not target.is_file():
            return ToolResult(ok=False, output=f"File not found: {path}", error="not_found")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult(ok=False, output=f"Cannot read file: {exc}", error=str(exc))
        truncated = len(text) > max_chars
        return ToolResult(
            ok=True,
            output=text[:max_chars] + ("\n…[truncated]" if truncated else ""),
            data={"path": str(target), "chars": len(text), "truncated": truncated},
        )


class WebSearchStubTool(Tool):
    """STUB web search — returns clearly-marked placeholder results.

    Replace with a real provider (Tavily, Brave Search, Serper) by overriding
    :meth:`execute` to call the API; the schema stays the same.
    """

    name = "web_search"
    description = (
        "Search the web for a query. STUB IMPLEMENTATION — returns clearly-marked "
        "placeholder results. Wire a real search API for production use."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
            "num_results": {"type": "integer", "description": "How many results (default 3)"},
        },
        "required": ["query"],
    }

    def execute(self, query: str, num_results: int = 3) -> ToolResult:
        results = [
            {
                "title": f"STUB result {i + 1} for '{query}'",
                "url": "https://example.com/stub",
                "snippet": "This is a stubbed search result. Connect a real search API for live data.",
            }
            for i in range(max(1, num_results))
        ]
        note = "STUB: not live web data. Replace WebSearchStubTool with a real API client."
        output = note + "\n" + json.dumps(results, indent=2)
        log.warning("web_search stub used for query: %s", query)
        return ToolResult(ok=True, output=output, data={"results": results, "stub": True})


class RestApiCallerTool(Tool):
    """Call an HTTP(S) JSON/REST endpoint with timeout and response size caps."""

    name = "rest_api"
    description = (
        "Perform an HTTP GET or POST against a URL and return the status code plus "
        "a truncated response body. Only http/https URLs; 10s default timeout."
    )
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "https:// URL to call"},
            "method": {"type": "string", "description": "GET or POST (default GET)"},
            "body": {"type": "object", "description": "JSON body for POST"},
            "timeout": {"type": "number", "description": "Seconds (default 10, max 30)"},
        },
        "required": ["url"],
    }

    def execute(self, url: str, method: str = "GET", body: Any = None, timeout: float = 10) -> ToolResult:
        method = method.upper()
        if method not in ("GET", "POST"):
            return ToolResult(ok=False, output="Only GET and POST are supported.", error="bad_method")
        if not url.startswith(("http://", "https://")):
            return ToolResult(ok=False, output="Only http/https URLs are allowed.", error="bad_scheme")
        timeout = min(max(float(timeout), 1), 30)
        try:
            resp = requests.request(method, url, json=body, timeout=timeout,
                                    headers={"User-Agent": "llm-agent-builder/1.0"})
        except requests.RequestException as exc:
            return ToolResult(ok=False, output=f"Request failed: {exc}", error=str(exc))
        text = resp.text[:4000]
        return ToolResult(
            ok=True,
            output=f"HTTP {resp.status_code}\n{text}",
            data={"status": resp.status_code, "truncated": len(resp.text) > 4000},
        )


def default_registry(allowed_root: str = "."):
    """Registry pre-loaded with the four built-in tools."""
    from .base import ToolRegistry

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(FileReaderTool(allowed_root=allowed_root))
    registry.register(WebSearchStubTool())
    registry.register(RestApiCallerTool())
    return registry
