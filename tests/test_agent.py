"""Tests for the agent loop: tool calling, multi-step plans, loop detection,
failure recovery, and the step budget. All use MockProvider — no network."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent
from providers import MockProvider, MultiProviderClient
from tools import default_registry


def make_agent(script, **kw):
    client = MultiProviderClient([MockProvider(script)])
    return Agent(client, default_registry(allowed_root=str(ROOT)), **kw)


def test_single_tool_call_then_final():
    agent = make_agent([
        '{"tool": "calculator", "args": {"expression": "15 * 240 / 100"}}',
        '{"final": "36"}',
    ])
    res = agent.run("What is 15% of 240?")
    assert res.success and res.final_answer == "36"
    assert res.steps[0].tool == "calculator"
    assert res.steps[0].tool_ok
    assert "36" in res.steps[0].observation


def test_multi_step_plan():
    agent = make_agent([
        '{"tool": "calculator", "args": {"expression": "1200 * 0.10"}}',
        '{"tool": "calculator", "args": {"expression": "120 + 30"}}',
        '{"final": "150"}',
    ])
    res = agent.run("Compute 10% of 1200, then add 30.")
    assert res.success
    assert [s.tool for s in res.steps if s.tool] == ["calculator", "calculator"]
    assert res.final_answer == "150"


def test_loop_detection_stops_gracefully():
    agent = make_agent(
        ['{"tool": "calculator", "args": {"expression": "2+2"}}'] * 4,
        max_repeats=1,
    )
    res = agent.run("Compute 2+2.")
    assert not res.success
    assert res.error and "loop" in res.error.lower()


def test_bad_args_recovery():
    agent = make_agent([
        '{"tool": "calculator", "args": {}}',  # missing required arg -> observation error
        '{"tool": "calculator", "args": {"expression": "2+2"}}',
        '{"final": "4"}',
    ])
    res = agent.run("What is 2+2?")
    assert res.success and res.final_answer == "4"
    assert not res.steps[0].tool_ok  # first attempt failed validation
    assert res.steps[1].tool_ok


def test_unknown_tool_recovery():
    agent = make_agent([
        '{"tool": "no_such_tool", "args": {}}',
        '{"final": "No such tool exists; answering directly."}',
    ])
    res = agent.run("Do something.")
    assert res.success
    assert not res.steps[0].tool_ok
    assert "unknown tool" in res.steps[0].observation.lower()


def test_step_budget_exhaustion():
    agent = make_agent(
        ['{"tool": "calculator", "args": {"expression": "1+1"}}'] * 10,
        max_steps=3, max_repeats=10,  # repeats allowed, but budget is 3
    )
    res = agent.run("Keep calculating.")
    assert not res.success
    assert "step budget" in (res.error or "").lower()


def test_malformed_json_prompts_retry():
    agent = make_agent([
        'Here is my answer in prose, no JSON at all',
        '{"final": "recovered"}',
    ])
    res = agent.run("Say hi.")
    assert res.success and res.final_answer == "recovered"
    assert not res.steps[0].tool_ok  # parse failure recorded as a step


def test_usage_tracked():
    agent = make_agent(['{"final": "done"}'])
    res = agent.run("Hi.")
    assert res.total_tokens > 0
    assert agent.client.usage.calls >= 1
