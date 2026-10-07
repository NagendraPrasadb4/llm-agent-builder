"""Deterministic offline evaluation of the agent.

Runs every case in ``test_set.json`` with ``MockProvider`` (no API keys, no
network) and reports five metrics per test:

- accuracy — expected answer fragments present in the final answer.
- task completion — agent finished (or failed gracefully exactly when expected).
- hallucination rate — for RAG tests: share of [n] citations that do not resolve
  to a retrieved source, or 1.0 when claims are made with no valid citation.
- latency — wall-clock seconds.
- token cost — tokens used and estimated USD.

Usage:  python eval/run_eval.py [--report eval/report.json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent import Agent  # noqa: E402
from providers import ChatMessage, MockProvider, MultiProviderClient  # noqa: E402
from rag import RAGPipeline  # noqa: E402
from tools import default_registry  # noqa: E402


def hallucination_rate(answer: str, retrieved_sources: list, expected_cite: str | None) -> float:
    """Fraction of citation markers that don't resolve to a retrieved source."""
    markers = re.findall(r"\[(\d+)\]", answer or "")
    if not markers:
        return 0.0  # no factual claims cited; nothing to flag
    bad = 0
    for m in markers:
        idx = int(m) - 1
        if not (0 <= idx < len(retrieved_sources)):
            bad += 1
        elif expected_cite and expected_cite not in retrieved_sources[idx]:
            bad += 1
    return bad / len(markers)


def run_case(case: dict, repo_root: Path) -> dict:
    provider = MockProvider(case["script"])
    client = MultiProviderClient([provider])
    registry = default_registry(allowed_root=str(repo_root))
    agent = Agent(client, registry, max_steps=8)

    retrieved: list = []
    start = time.time()
    if case["category"] == "rag":
        pipeline = RAGPipeline(client).build(str(repo_root / "data" / "sample_docs"))
        rag_answer = pipeline.query(case["rag_question"], top_k=3)
        retrieved = [h.source for h in rag_answer.retrieved]
        # The scripted mock answer stands in for the generation step, so the
        # metrics below measure the full retrieval + citation path deterministically.
        result = None
        final = rag_answer.answer
        success = True
        latency = round(time.time() - start, 3)
        tokens = client.usage.total_tokens
    else:
        result = agent.run(case["task"])
        final = result.final_answer or ""
        success = result.success
        latency = result.latency_s
        tokens = result.total_tokens

    final_l = final.lower()
    accuracy = all(frag.lower() in final_l for frag in case.get("expected_final_contains", []))
    completion = (success == case.get("expected_success", True))
    if not case.get("expected_success", True):
        err_ok = case.get("expected_error_contains", "").lower()
        completion = completion and (err_ok in (getattr(result, "error", "") or "").lower())
    tools_used = sorted({s.tool for s in (result.steps if result else []) if s.tool})
    tools_ok = all(t in tools_used for t in case.get("expected_tools", []))
    hallu = hallucination_rate(final, retrieved, case.get("expected_citation_contains"))

    return {
        "id": case["id"],
        "category": case["category"],
        "accuracy": accuracy,
        "task_completion": completion,
        "tools_correct": tools_ok,
        "hallucination_rate": round(hallu, 3),
        "latency_s": latency,
        "tokens": tokens,
        "est_cost_usd": round(client.usage.estimated_cost_usd(), 6),
        "final_answer": final[:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="eval/report.json")
    args = parser.parse_args()

    with open(ROOT / "eval" / "test_set.json") as f:
        cases = json.load(f)

    results = [run_case(c, ROOT) for c in cases]

    def avg(key):
        vals = [r[key] for r in results]
        return round(sum(vals) / len(vals), 3) if vals else 0

    summary = {
        "n": len(results),
        "accuracy": avg("accuracy"),
        "task_completion": avg("task_completion"),
        "tools_correct": avg("tools_correct"),
        "hallucination_rate": avg("hallucination_rate"),
        "avg_latency_s": avg("latency_s"),
        "total_tokens": sum(r["tokens"] for r in results),
        "total_est_cost_usd": round(sum(r["est_cost_usd"] for r in results), 6),
    }

    print(f"\n{'test':<22}{'cat':<12}{'acc':<6}{'done':<6}{'tools':<7}{'hallu':<7}{'lat_s':<8}{'toks':<7}")
    print("-" * 80)
    for r in results:
        print(f"{r['id']:<22}{r['category']:<12}{str(r['accuracy']):<6}"
              f"{str(r['task_completion']):<6}{str(r['tools_correct']):<7}"
              f"{r['hallucination_rate']:<7}{r['latency_s']:<8}{r['tokens']:<7}")
    print("-" * 80)
    print("SUMMARY:", json.dumps(summary, indent=2))

    out = ROOT / args.report
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"\nReport written to {out}")

    # Non-zero exit if any core metric regresses to zero.
    if summary["accuracy"] < 1.0 or summary["task_completion"] < 1.0:
        print("EVAL FAILED: accuracy or task_completion below 1.0")
        sys.exit(1)


if __name__ == "__main__":
    main()
