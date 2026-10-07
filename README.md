# llm-agent-builder

A personal engineering project: a working, from-scratch AI agent system in Python —
a ReAct-style planning loop with tool calling, a pluggable RAG pipeline, and a
multi-provider LLM client with automatic failover. Built to explore how production
agent systems are actually put together: the loop, the tools, the retrieval, the
resilience.

No employer code or data is used here, and nothing here claims a production
deployment. The bundled evaluation runs fully offline with a deterministic mock
provider, so every number it reports is reproducible.

## What it does

- **Agent loop** (`agent/`) — plans with a reasoning loop, calls tools via JSON
  actions, keeps conversation memory under a token budget, executes multi-step
  tasks, detects repeated-action loops, and recovers gracefully from tool and
  provider failures instead of hanging or burning tokens.
- **Tool ecosystem** (`tools/`) — MCP-style tools with JSON-schema contracts and
  validated dispatch: safe calculator, sandboxed file reader, web-search stub,
  REST API caller.
- **RAG pipeline** (`rag/`) — ingest txt/md docs, hierarchical chunking with
  overlap, pluggable embeddings (offline trigram-hash baseline included),
  cosine-similarity vector search, and citation-grounded answers.
- **Multi-provider LLM client** (`providers/`) — OpenAI, Anthropic, and Gemini
  clients over plain `requests`, with ordered automatic failover, SSE-style
  streaming, per-call token/cost tracking, and context-window management.
- **Evaluation** (`eval/`) — deterministic offline harness measuring accuracy,
  task completion, hallucination rate, latency, and token cost on a 7-case test set.

## Architecture

```
                    ┌─────────────────────────────┐
                    │        Agent (core.py)       │
                    │  plan → act → observe loop  │
                    │  loop detection · recovery  │
                    │  step budget · memory       │
                    └──────┬──────────────┬───────┘
                           │              │
              ┌────────────▼─────┐  ┌─────▼──────────────────┐
              │  ToolRegistry    │  │ MultiProviderClient     │
              │  JSON-schema     │  │ openai → anthropic →    │
              │  validated tools │  │ gemini → (failover)     │
              │  calculator      │  │ streaming · token/cost  │
              │  file_reader     │  │ context-window mgmt     │
              │  web_search*     │  └─────────────────────────┘
              │  rest_api        │
              └──────────────────┘
                           │
              ┌────────────▼─────┐
              │   RAG pipeline   │
              │ ingest → chunk   │
              │ → embed → search │
              │ → cited answer   │
              └──────────────────┘
                    * stub — wire a real API for production
```

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the test suite (33 tests, offline)
pytest tests/ -q

# Offline demo: agent loop, RAG with citations, provider failover
python examples/demo.py

# Deterministic evaluation (7 cases, offline, writes eval/report.json)
python eval/run_eval.py

# Live LLM calls (optional) — copy .env.example to .env and add keys
cp .env.example .env
```

Docker:

```bash
docker build -t llm-agent-builder .
docker run --rm llm-agent-builder
```

## Configuration

All settings are optional; the repo runs fully offline without them.

| Variable | Purpose | Default |
|---|---|---|
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` | Live provider access | unset (mock mode) |
| `OPENAI_MODEL` / `ANTHROPIC_MODEL` / `GEMINI_MODEL` | Model selection | `gpt-4o-mini` / `claude-3-5-haiku-latest` / `gemini-2.0-flash` |
| `PROVIDER_ORDER` | Failover order | `openai,anthropic,gemini` |
| `AGENT_MAX_STEPS` | Cap on agent loop iterations | `8` |
| `RAG_TOP_K` | Retrieved chunks per query | `4` |

## Tools and frameworks used

- **Python 3.11**, standard library first; only hard dependencies are
  `requests` (HTTP/providers) and `pytest` (tests).
- **No provider SDKs** — OpenAI/Anthropic/Gemini are called over plain HTTP,
  including hand-rolled SSE stream parsing, so the failover and accounting
  logic stays visible and provider-agnostic.
- **Embeddings are pluggable**: `rag/embeddings.py` defines the `Embedder`
  interface; the default `HashEmbedder` is a deterministic offline baseline
  (commented example shows how to plug in `sentence-transformers`).
- Optional upgrades are marked in `requirements.txt` (`tiktoken` for exact
  token counts instead of the chars/4 heuristic).

## Hardest problem solved

Making the agent loop **robust without an LLM judge**. A naive ReAct loop fails
in three predictable ways: it repeats the same doomed tool call forever, it
crashes on the first tool error, and it blows past context limits on long tasks.
The fixes here are all mechanical and testable:

1. **Loop detection** — every tool call is fingerprinted (`tool` + sorted args);
   a repeat triggers a warning observation, and past `max_repeats` the agent
   stops gracefully with the full trace instead of burning tokens.
2. **Failure as observation** — tool errors (bad args, unknown tool, HTTP
   failures) are returned as observations, never raised, so the model can
   replan: fix arguments, switch tools, or answer from partial results.
3. **Budgets everywhere** — `max_steps` caps the loop, `ConversationMemory`
   compacts history under a token budget, and `MultiProviderClient` trims
   prompts to each provider's context window before sending.

The second-hardest problem was **provider-agnostic streaming with failover**:
each provider streams differently (OpenAI `data:` deltas, Anthropic
`content_block_delta` events, Gemini SSE), so each client normalizes to a plain
iterator of text chunks, and the failover wrapper only commits to a provider
once its stream actually yields.

## How it's measured

`eval/run_eval.py` runs 7 deterministic cases offline (mock provider replays
scripted replies; real tools and real retrieval execute):

| Metric | What it checks |
|---|---|
| Accuracy | Expected answer fragments present in the final answer |
| Task completion | Finished with a final answer — or failed gracefully exactly when the test expects it (loop detection) |
| Hallucination rate | Share of `[n]` citations that don't resolve to a retrieved source |
| Latency | Wall-clock seconds per task |
| Token cost | Tokens used + estimated USD from the provider cost table |

Current result: **7/7 pass, accuracy 1.0, completion 1.0, hallucination rate 0.0**
(see `eval/report.json` after a run). `pytest tests/` covers the loop, tools,
RAG ranking/citations, and failover/usage accounting — 33 tests.

## Honest limitations

- The web-search tool is a **stub** returning marked placeholder results; wire a
  real search API for live data.
- The default embedder is a trigram-hash **baseline**, not a semantic model —
  fine for the bundled demo corpus, not for production retrieval quality.
- Token counts use a chars/4 heuristic and costs are rough public-rate
  estimates unless you override them; use `tiktoken` and contract rates for
  real accounting.
- The eval measures the harness deterministically with a mock LLM — it proves
  the machinery (loop, tools, retrieval, citations, failover) works, not that a
  particular frontier model scores well.
