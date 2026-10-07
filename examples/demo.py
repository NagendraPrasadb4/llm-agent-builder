"""Offline demo of llm-agent-builder — no API keys needed.

Showcases:
1. The agent loop solving a multi-step task with the calculator tool.
2. The RAG pipeline answering a question with citations.
3. Provider failover (a failing provider is skipped automatically).

Run:  python examples/demo.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent
from providers import MockProvider, MultiProviderClient, ProviderError
from providers import LLMProvider
from rag import RAGPipeline
from tools import default_registry


class AlwaysFails(LLMProvider):
    name = "unreliable"

    def chat(self, messages, **kw):
        raise ProviderError("simulated outage")

    def stream(self, messages, **kw):
        raise ProviderError("simulated outage")
        yield


def demo_agent_loop():
    print("=" * 70)
    print("1) AGENT LOOP — multi-step tool use")
    print("=" * 70)
    provider = MockProvider([
        '{"tool": "calculator", "args": {"expression": "1200 * 0.10"}}',
        '{"tool": "calculator", "args": {"expression": "120 + 30"}}',
        '{"final": "10% of the 1200 budget is 120; plus 30 gives 150."}',
    ])
    agent = Agent(MultiProviderClient([provider]), default_registry(allowed_root=str(ROOT)))
    result = agent.run("Compute 10% of the 1200 project budget, then add 30.")
    for step in result.steps:
        if step.tool:
            print(f"  [tool] {step.tool} {step.args} -> {step.observation[:80]}")
    print(f"  final: {result.final_answer}")
    print(f"  success={result.success} steps={len(result.steps)} "
          f"tokens={result.total_tokens} latency={result.latency_s}s\n")


def demo_rag():
    print("=" * 70)
    print("2) RAG PIPELINE — grounded answer with citations")
    print("=" * 70)
    provider = MockProvider([
        "The agent fingerprints each tool call (tool name + sorted arguments) "
        "and keeps recent fingerprints in memory [1]. Repeating the same call "
        "beyond the limit stops the loop gracefully [1]."
    ])
    pipeline = RAGPipeline(MultiProviderClient([provider])).build(str(ROOT / "data" / "sample_docs"))
    answer = pipeline.query("How does the agent detect repeated tool calls?", top_k=3)
    print(f"  answer: {answer.answer[:200]}...")
    print(f"  citations: {answer.citations}")
    print(f"  top retrieved score: {answer.retrieved[0].score:.3f}\n")


def demo_failover():
    print("=" * 70)
    print("3) FAILOVER — first provider fails, second serves")
    print("=" * 70)
    client = MultiProviderClient([AlwaysFails(), MockProvider(['{"final": "served by fallback"}'])])
    agent = Agent(client, default_registry(allowed_root=str(ROOT)))
    result = agent.run("Say hello.")
    print(f"  final: {result.final_answer}")
    print(f"  usage: {client.usage.summary()}\n")


if __name__ == "__main__":
    demo_agent_loop()
    demo_rag()
    demo_failover()
    print("Demo complete — all offline, no API keys used.")
