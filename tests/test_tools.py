"""Tests for the tool ecosystem: schema validation, dispatch, and builtins."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import (
    CalculatorTool, FileReaderTool, RestApiCallerTool, ToolRegistry, WebSearchStubTool,
    default_registry,
)


def test_calculator_basic():
    r = CalculatorTool().execute(expression="15 * 240 / 100")
    assert r.ok and r.data == 36.0


def test_calculator_functions():
    r = CalculatorTool().execute(expression="sqrt(16) + pow(2, 3)")
    assert r.ok and r.data == 12.0


def test_calculator_rejects_code_injection():
    r = CalculatorTool().execute(expression="__import__('os').system('x')")
    assert not r.ok


def test_calculator_division_by_zero_is_result_not_crash():
    r = CalculatorTool().execute(expression="1/0")
    assert not r.ok and "failed" in r.output.lower()


def test_schema_validation_missing_required():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    r = reg.dispatch("calculator", {})
    assert not r.ok and "missing required" in r.output.lower()


def test_schema_validation_wrong_type():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    r = reg.dispatch("calculator", {"expression": 123})
    assert not r.ok and "must be string" in r.output.lower()


def test_dispatch_unknown_tool():
    r = default_registry().dispatch("nope", {})
    assert not r.ok and "unknown tool" in r.output.lower()


def test_file_reader_sandbox(tmp_path):
    (tmp_path / "ok.txt").write_text("hello")
    tool = FileReaderTool(allowed_root=str(tmp_path))
    assert tool.execute(path="ok.txt").ok
    r = tool.execute(path="../evil.txt")
    assert not r.ok and "outside the allowed directory" in r.output


def test_file_reader_missing():
    r = FileReaderTool(allowed_root=str(ROOT)).execute(path="no/such/file.txt")
    assert not r.ok


def test_web_search_stub_is_honest():
    r = WebSearchStubTool().execute(query="test")
    assert r.ok and r.data["stub"] is True
    assert "STUB" in r.output


def test_rest_api_rejects_non_http():
    r = RestApiCallerTool().execute(url="ftp://example.com/x")
    assert not r.ok


def test_registry_lists_all_builtin_tools():
    names = {s["name"] for s in default_registry().list_schemas()}
    assert {"calculator", "file_reader", "web_search", "rest_api"} <= names
