# Tool ecosystem: MCP-style tools with JSON-schema contracts.
#
# Four built-in tools live in tools/builtin.py:
# - calculator: safe arithmetic via an AST whitelist (no arbitrary eval).
# - file_reader: sandboxed reads confined to an allowed root directory.
# - web_search: STUB — returns clearly-marked placeholder results until a real
#   search API (Tavily, Brave, Serper) is wired in.
# - rest_api: HTTP GET/POST with timeouts and response size caps.
#
# Every tool declares name, description, and a JSON-schema parameters object.
# ToolRegistry validates arguments before dispatch and converts all failures
# into structured ToolResult observations for the agent loop.
