# Agent loop: plan -> act -> observe with loop detection and graceful recovery.
#
# The agent runs a ReAct-style loop. Each turn the model emits exactly one JSON
# action: {"tool": name, "args": {...}} or {"final": answer}. Tool schemas from
# tools/ are injected into the system prompt so the model knows the contract.
#
# Robustness mechanisms (the core of this file):
# - Loop detection: every tool call is fingerprinted (tool + sorted args). A
#   repeated fingerprint triggers a warning observation; beyond max_repeats
#   the agent stops gracefully instead of burning tokens forever.
# - Failure recovery: tool errors (bad args, unknown tool, execution failure)
#   are returned as observations, never raised — the model can replan, fix
#   arguments, pick another tool, or answer from what it has.
# - Step budget: max_steps caps the loop; exhaustion returns a partial result
#   with the full trace, never a hang.
#
# See agent/core.py for the implementation.
