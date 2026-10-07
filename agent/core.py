"""Core agent loop: plan -> act -> observe, with loop detection and recovery.

The agent reasons in a ReAct-style loop:

1. **Plan** — the LLM sees the task, tool schemas, and history, then emits one
   JSON action: either ``{"tool": name, "args": {...}}`` or ``{"final": answer}``.
2. **Act** — the tool registry validates args against the JSON schema and runs
   the tool; every outcome (including failures) becomes an observation.
3. **Observe** — the observation is appended to memory and the loop repeats.

Robustness (the hard part) comes from three mechanisms:

- **Loop detection** — repeating the exact same tool call is caught; the agent
  is told to try a different approach, and after repeated loops it stops
  gracefully instead of burning tokens forever.
- **Failure recovery** — tool errors are returned as observations (not
  exceptions), so the model can replan: fix arguments, pick another tool, or
  answer from what it has.
- **Step budget** — ``max_steps`` caps the loop; exhaustion yields a partial
  result with the full trace, never a hang.
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from providers.base import ChatMessage, estimate_tokens
from providers.failover import MultiProviderClient
from tools.base import ToolRegistry, ToolResult
from .memory import ConversationMemory

log = logging.getLogger(__name__)

SYSTEM_TEMPLATE = """You are a careful AI agent that solves tasks by calling tools.

TOOLS AVAILABLE:
{tools}

RULES:
- On each turn, output EXACTLY ONE JSON object, nothing else.
- To call a tool: {{"tool": "<tool_name>", "args": {{<arguments>}}}}
- When you have the answer: {{"final": "<your answer>"}}
- Use only the tools listed above. If a tool call fails, read the error and try a different approach.
- Never repeat the exact same tool call twice in a row. If you are stuck, answer with what you know.
- Keep reasoning concise. Cite tool outputs you relied on.
"""


@dataclass
class AgentStep:
    """One plan->act->observe iteration."""

    index: int
    raw_response: str
    tool: Optional[str] = None
    args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    tool_ok: bool = True


@dataclass
class AgentResult:
    """Outcome of Agent.run — always returned, even on graceful failure."""

    success: bool
    final_answer: Optional[str]
    steps: List[AgentStep] = field(default_factory=list)
    total_tokens: int = 0
    latency_s: float = 0.0
    error: Optional[str] = None

    def trace(self) -> str:
        lines = []
        for s in self.steps:
            lines.append(f"--- step {s.index} ---")
            if s.tool:
                lines.append(f"tool: {s.tool} args={json.dumps(s.args)} ok={s.tool_ok}")
                lines.append(f"observation: {s.observation[:500]}")
            else:
                lines.append(f"response: {s.raw_response[:500]}")
        lines.append(f"final: {self.final_answer}")
        return "\n".join(lines)


class _Final:
    def __init__(self, answer: str) -> None:
        self.answer = answer


class _ToolCall:
    def __init__(self, tool: str, args: Dict[str, Any]) -> None:
        self.tool = tool
        self.args = args


class Agent:
    """ReAct-style agent with loop detection and graceful failure recovery."""

    def __init__(self, client: MultiProviderClient, tools: ToolRegistry,
                 max_steps: int = 8, max_repeats: int = 2,
                 memory: Optional[ConversationMemory] = None) -> None:
        self.client = client
        self.tools = tools
        self.max_steps = max_steps
        self.max_repeats = max_repeats
        self.system_prompt = SYSTEM_TEMPLATE.format(tools=tools.describe())
        self.memory = memory or ConversationMemory(system_prompt=self.system_prompt)

    # -- planning ---------------------------------------------------------
    def _parse_action(self, text: str):
        """Extract one JSON action from the model response.

        Accepts a fenced ```json block or a bare {...} object. Raises ValueError
        with a helpful message so the loop can ask the model to retry.
        """
        match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if not match:
            match = re.search(r"(\{.*\})", text, re.DOTALL)
        if not match:
            raise ValueError("no JSON action found — reply with exactly one JSON object")
        try:
            data = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"malformed JSON action: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError("action must be a JSON object")
        if "final" in data:
            return _Final(str(data["final"]))
        tool = data.get("tool") or data.get("action")
        if not tool:
            raise ValueError("action needs 'tool' or 'final'")
        args = data.get("args") or data.get("arguments") or {}
        if not isinstance(args, dict):
            raise ValueError("'args' must be a JSON object")
        return _ToolCall(str(tool), args)

    # -- main loop --------------------------------------------------------
    def run(self, task: str) -> AgentResult:
        start = time.time()
        steps: List[AgentStep] = []
        self.memory.add("user", f"TASK: {task}\nBegin. Reply with exactly one JSON action.")
        recent_calls: List[str] = []  # for loop detection
        repeat_warnings = 0

        for i in range(1, self.max_steps + 1):
            self.memory.compact()
            try:
                resp = self.client.chat(self.memory.messages())
            except Exception as exc:  # noqa: BLE001 - provider failover exhausted
                return self._finish(False, None, steps, start,
                                    f"LLM unavailable after failover: {exc}")
            self.memory.add("assistant", resp.content)

            try:
                action = self._parse_action(resp.content)
            except ValueError as exc:
                obs = (f"Your last reply was not a valid action: {exc}. "
                       f"Reply with exactly one JSON object.")
                self.memory.add("user", obs)
                steps.append(AgentStep(index=i, raw_response=resp.content, observation=obs,
                                       tool_ok=False))
                continue

            if isinstance(action, _Final):
                return self._finish(True, action.answer, steps, start)

            # Loop detection: same tool+args as a recent call?
            fingerprint = f"{action.tool}:{json.dumps(action.args, sort_keys=True)}"
            if fingerprint in recent_calls:
                repeat_warnings += 1
                if repeat_warnings > self.max_repeats:
                    return self._finish(
                        False, None, steps, start,
                        f"stopped: repeated the same '{action.tool}' call "
                        f"{repeat_warnings}x with no progress (loop detected)")
                obs = (f"You already called '{action.tool}' with these exact arguments and "
                       f"made no progress. Try a DIFFERENT tool or different arguments, or "
                       f"answer with what you have using {{\"final\": ...}}.")
                self.memory.add("user", obs)
                steps.append(AgentStep(index=i, raw_response=resp.content, tool=action.tool,
                                       args=action.args, observation=obs, tool_ok=False))
                continue
            recent_calls.append(fingerprint)
            recent_calls = recent_calls[-6:]

            # Act — tool errors become observations (recovery), never exceptions.
            result: ToolResult = self.tools.dispatch(action.tool, action.args)
            obs = result.output if result.ok else (
                f"Tool '{action.tool}' failed: {result.output}\n"
                f"Recover by fixing the arguments, choosing a different tool, or "
                f"answering from what you know with {{\"final\": ...}}.")
            self.memory.add("user", f"OBSERVATION from {action.tool}:\n{obs}")
            steps.append(AgentStep(index=i, raw_response=resp.content, tool=action.tool,
                                   args=action.args, observation=obs, tool_ok=result.ok))

        return self._finish(False, None, steps, start,
                            f"step budget exhausted ({self.max_steps} steps) without a final answer")

    def _finish(self, success: bool, final: Optional[str], steps: List[AgentStep],
                start: float, error: Optional[str] = None) -> AgentResult:
        return AgentResult(
            success=success,
            final_answer=final,
            steps=steps,
            total_tokens=self.client.usage.total_tokens,
            latency_s=round(time.time() - start, 3),
            error=error,
        )
