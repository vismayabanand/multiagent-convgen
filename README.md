# toolgen — Multi-Agent Tool-Use Conversation Generator

Generates synthetic multi-turn conversations with multi-step tool-use traces, grounded in ToolBench tool schemas. Suitable for training and evaluating tool-use agents.

---

## Prerequisites

- Python 3.10+
- An OpenAI or Anthropic API key

---

## Installation

```bash
git clone <repo-url>
cd <repo>
pip install -e ".[dev]"
```

Create a `.env` file in the repo root with your API key:

```bash
# .env
OPENAI_API_KEY=sk-...
# OR
ANTHROPIC_API_KEY=sk-ant-...
```

---

## Running the Pipeline

The pre-built artifacts (`registry.pkl`, `graph.pkl`) are already committed to the repo under `artifacts/`. You do **not** need to run `toolgen build` unless you want to rebuild from raw ToolBench data.

### Step 1 — Generate conversations

```bash
# Run B: cross-conversation steering enabled (primary run)
toolgen generate \
  --artifacts-dir ./artifacts \
  --output ./output/run_B.jsonl \
  --n 100 \
  --seed 42

# Run A: steering disabled (for diversity experiment comparison)
toolgen generate \
  --artifacts-dir ./artifacts \
  --output ./output/run_A.jsonl \
  --n 100 \
  --seed 42 \
  --no-cross-conversation-steering
```

Pre-generated datasets (`output/run_A.jsonl`, `output/run_B.jsonl`) are also already in the repo if you want to skip generation.

### Step 2 — Evaluate

```bash
toolgen evaluate --dataset ./output/run_B.jsonl --output ./output/report_B.json
toolgen evaluate --dataset ./output/run_A.jsonl --output ./output/report_A.json
```

### Step 3 — Run tests

```bash
# Unit + integration tests (no API key needed)
pytest tests/unit/ tests/integration/ -v

# End-to-end test against pre-generated dataset (no API key needed)
pytest tests/e2e/ --run-e2e --e2e-dataset ./output/run_B.jsonl -v
```

---

## Rebuilding from Raw ToolBench Data (optional)

If you want to rebuild the artifacts from scratch:

```bash
# Download ToolBench tool data
git clone https://github.com/OpenBMB/ToolBench.git
# Tool JSON files are under: ToolBench/data/toolenv/tools/

toolgen build \
  --data-dir ./ToolBench/data/toolenv/tools/ \
  --output-dir ./artifacts
```

Then proceed with Step 1 above.

---

## CLI Reference

### `toolgen build`

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | required | Path to ToolBench data directory or JSON file |
| `--output-dir` | `./artifacts` | Where to write artifacts |
| `--min-confidence` | `0.25` | Minimum edge confidence for Tool Graph |
| `--limit` | None | Limit number of tools loaded (for testing) |

### `toolgen generate`

| Flag | Default | Description |
|------|---------|-------------|
| `--artifacts-dir` | `./artifacts` | Directory with build artifacts |
| `--output` | `./output/dataset.jsonl` | Output JSONL path |
| `--n` | `100` | Number of conversations to generate |
| `--seed` | `42` | Random seed |
| `--model` | `gpt-4o-mini` | LLM model to use |
| `--provider` | `auto` | `auto` \| `openai` \| `anthropic` |
| `--no-cross-conversation-steering` | False | Disable diversity steering (Run A) |
| `--min-steps` | `2` | Minimum tool steps per chain |
| `--max-steps` | `5` | Maximum tool steps per chain |
| `--domain` | None | Restrict to a specific ToolBench category |

### `toolgen evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Path to JSONL file |
| `--output` | None | Write JSON report to this path |
| `--threshold` | `3.5` | Score threshold for pass/fail |

---

## Diversity Experiment Results

Two runs at N=200, seed=42, model=gpt-4o-mini:

| Metric | Run A (no steering) | Run B (with steering) |
|--------|--------------------|-----------------------|
| Tool-Pair TTR | 0.2706 | **0.3132** ▲ |
| Domain Entropy (normalized) | 0.9199 | **0.9486** ▲ |
| Mean Overall Score | 4.38 | 4.37 |
| Pass rate | 100% | 100% |

Steering improves both diversity metrics with negligible quality cost. See [DESIGN.md](DESIGN.md) §9 for full analysis.

---

## Output Format

Each JSONL line is a conversation record:

```json
{
  "conversation_id": "conv_42_0001",
  "messages": [
    {"role": "user", "content": "Find me a hotel in Paris"},
    {"role": "assistant", "content": "What's your budget?"},
    {"role": "user", "content": "Under 200 EUR per night"},
    {"role": "assistant", "content": null, "tool_calls": [
      {"endpoint": "Travel/hotels/search", "arguments": {"city": "Paris", "max_price": 200}}
    ]},
    {"role": "tool", "tool_id": "Travel/hotels/search", "content": {"hotel_id": "tra_a1b2c3", "name": "Hotel Lumière", "price": 185.5}},
    {"role": "assistant", "content": "I found Hotel Lumière at 185.50 EUR/night."}
  ],
  "judge_scores": {
    "tool_selection": 5,
    "naturalness": 4,
    "chaining": 5,
    "overall": 4.67,
    "reasoning": {
      "tool_selection": "Correct tool, sensible arguments",
      "naturalness": "Natural flow with appropriate disambiguation",
      "chaining": "Hotel ID correctly propagated"
    },
    "repair_hints": []
  },
  "metadata": {
    "seed": 42,
    "tools_used": ["Travel/hotels/search"],
    "tool_categories": ["Travel"],
    "num_turns": 6,
    "num_tool_calls": 1,
    "num_distinct_tools": 1,
    "conversation_type": "sequential",
    "disambiguation_turns": 1,
    "repair_attempts": 0,
    "steering_enabled": true,
    "generated_at": "2026-04-12T22:00:00Z",
    "model": "gpt-4o-mini"
  }
}
```

---

## Project Structure

```
toolgen/
  cli.py              CLI entry point (build, generate, evaluate)
  registry/
    models.py         Tool, Parameter, ResponseField data models
    loader.py         ToolBench JSON → normalized registry
  graph/
    builder.py        Tool Graph construction (edge detection)
    sampler.py        Constrained chain sampling
    coverage.py       Cross-conversation diversity tracker
  executor/
    mock_generator.py Schema-derived + LLM mock responses
    session.py        Session state for arg grounding
  agents/
    planner.py        Conversation planner (structured output)
    user.py           User simulator
    assistant.py      Assistant (tool calls + narrative)
    judge.py          LLM-as-judge scorer
    orchestrator.py   Dialogue loop + repair
  context/
    conversation.py   Shared conversation state
  output/
    schema.py         Output record schema
    writer.py         JSONL writer
tests/
  unit/               Per-module unit tests
  integration/        Retry/repair loop + dialogue controls tests
  e2e/                End-to-end pipeline test (100+ samples, judge threshold)
```

See [DESIGN.md](DESIGN.md) for full system design documentation.
