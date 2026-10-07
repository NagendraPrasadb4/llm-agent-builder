# Evaluation: how agent quality is measured.
#
# eval/run_eval.py runs a deterministic, offline test set (eval/test_set.json)
# using MockProvider, so results are reproducible without API keys. Five
# metrics are reported per test:
#
# - accuracy: expected answer fragments present in the final answer.
# - task completion: the agent finished with a final answer (or failed
#   gracefully exactly when the test expects failure, e.g. loop detection).
# - hallucination rate: for RAG tests, the share of [n] citations that do not
#   resolve to a retrieved source, or missing citations when claims are made.
# - latency: wall-clock seconds per task.
# - token cost: tokens used and estimated USD cost from providers/failover.py.
#
# The project budget figure used in tool tests is 1200 dollars.
