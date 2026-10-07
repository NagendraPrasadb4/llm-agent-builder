"""MCP-style tool ecosystem: base classes, registry, and JSON-schema validation.

Every tool declares a JSON-schema ``parameters`` contract. The registry validates
arguments against the schema before dispatch and converts every outcome
(including failures) into a structured :class:`ToolResult`, so the agent loop
always gets a clean observation to reason over.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

#: JSON-schema type -> Python type(s) used for argument validation.
_TYPE_MAP: Dict[str, Any] = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "array": list,
    "object": dict,
}


class ToolError(Exception):
    """Raised when a tool cannot run: bad args, unsafe input, or execution failure."""


@dataclass
class ToolResult:
    """Structured outcome of a tool call — always returned, never raised."""

    ok: bool
    output: str
    data: Any = None
    error: Optional[str] = None


class Tool:
    """Base class for agent tools.

    Subclasses set ``name``, ``description`` and a JSON-schema ``parameters``
    dict, then implement :meth:`execute`.
    """

    name: str = "tool"
    description: str = ""
    parameters: Dict[str, Any] = {}

    def validate(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Validate ``args`` against the JSON-schema contract.

        Raises:
            ToolError: if a required argument is missing or mistyped.
        """
        if not isinstance(args, dict):
            raise ToolError(f"arguments must be an object, got {type(args).__name__}")
        schema = self.parameters or {}
        properties = schema.get("properties", {})
        for required in schema.get("required", []):
            if required not in args:
                raise ToolError(f"missing required argument: '{required}'")
        for key, value in args.items():
            spec = properties.get(key, {})
            expected = spec.get("type")
            py_type = _TYPE_MAP.get(expected)
            # bool is a subclass of int — require exactness for booleans/integers.
            if expected == "boolean" and not isinstance(value, bool):
                raise ToolError(f"argument '{key}' must be boolean, got {type(value).__name__}")
            elif expected == "integer" and not isinstance(value, int):
                raise ToolError(f"argument '{key}' must be integer, got {type(value).__name__}")
            elif py_type and expected not in ("boolean", "integer") and not isinstance(value, py_type):
                raise ToolError(f"argument '{key}' must be {expected}, got {type(value).__name__}")
        return args

    def schema(self) -> Dict[str, Any]:
        """Return the tool's declaration for provider function-calling / prompts."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def execute(self, **kwargs: Any) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError


class ToolRegistry:
    """Named collection of tools with validated dispatch."""

    def __init__(self) -> None:
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if not tool.name:
            raise ToolError("tool must have a name")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolError(
                f"unknown tool '{name}'. Available: {', '.join(sorted(self._tools)) or 'none'}"
            )

    def list_schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def dispatch(self, name: str, args: Dict[str, Any]) -> ToolResult:
        """Validate and run a tool, converting all failures to ToolResult."""
        try:
            tool = self.get(name)
            validated = tool.validate(args or {})
            return tool.execute(**validated)
        except ToolError as exc:
            return ToolResult(ok=False, output=f"Tool error: {exc}", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - tools must never crash the loop
            return ToolResult(ok=False, output=f"Tool '{name}' crashed: {exc}", error=str(exc))

    def describe(self) -> str:
        """Human/LLM-readable listing used in the agent system prompt."""
        lines = []
        for tool in self._tools.values():
            params = json.dumps(tool.parameters)
            lines.append(f"- {tool.name}: {tool.description}\n  parameters: {params}")
        return "\n".join(lines)
