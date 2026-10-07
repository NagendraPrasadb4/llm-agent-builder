"""MCP-style tool ecosystem for the agent."""
from .base import Tool, ToolError, ToolRegistry, ToolResult
from .builtin import (
    CalculatorTool,
    FileReaderTool,
    RestApiCallerTool,
    WebSearchStubTool,
    default_registry,
)

__all__ = [
    "Tool", "ToolError", "ToolRegistry", "ToolResult",
    "CalculatorTool", "FileReaderTool", "RestApiCallerTool",
    "WebSearchStubTool", "default_registry",
]
