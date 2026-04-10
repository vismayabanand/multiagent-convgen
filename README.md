# toolgen — Multi-Agent Tool-Use Conversation Generator

Generates synthetic multi-turn conversations with multi-step tool-use traces, grounded in ToolBench tool schemas. Suitable for training and evaluating tool-use agents.

## Quick Start

### Prerequisites

- Python 3.10+
- An OpenAI API key (or compatible endpoint)

### Install

```bash
cd /path/to/repo
pip install -e ".[dev]"
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
```

### End-to-end pipeline

```bash
# 1. Build artifacts from ToolBench data
toolgen build --data-dir ./data/toolbench --output-dir ./artifacts

# 2. Generate conversations (Run B — steering enabled)
toolgen generate \
  --artifacts-dir ./artifacts \
  --output ./output/dataset_B.jsonl \
  --n 100 \
  --seed 42

# 3. Generate conversations (Run A — steering disabled, for diversity experiment)
toolgen generate \
  --artifacts-dir ./artifacts \
  --output ./output/dataset_A.jsonl \
  --n 100 \
  --seed 42 \
  --no-cross-conversation-steering

# 4. Evaluate either dataset
toolgen evaluate --dataset ./output/dataset_B.jsonl --output ./output/report_B.json
toolgen evaluate --dataset ./output/dataset_A.jsonl --output ./output/report_A.json
```

## CLI Reference

### `toolgen build`

Ingests ToolBench JSON and builds all derived artifacts.

| Flag | Default | Description |
|------|---------|-------------|
| `--data-dir` | required | Path to ToolBench data directory or JSON file |
| `--output-dir` | `./artifacts` | Where to write artifacts |
| `--min-confidence` | `0.25` | Minimum edge confidence for Tool Graph |
| `--limit` | None | Limit number of tools (for testing) |

Artifacts produced:
- `artifacts/registry.pkl` — normalized Tool registry
- `artifacts/graph.pkl` — Tool Graph (networkx DiGraph)
- `artifacts/stats.json` — graph statistics

### `toolgen generate`

Generates synthetic conversations from the built artifacts.

| Flag | Default | Description |
|------|---------|-------------|
| `--artifacts-dir` | `./artifacts` | Artifacts directory |
| `--output` | `./output/dataset.jsonl` | Output JSONL path |
| `--n` | `100` | Number of conversations to generate |
| `--seed` | `42` | Random seed |
| `--model` | `gpt-4o-mini` | LLM model to use |
| `--no-cross-conversation-steering` | False | Disable diversity steering (Run A) |
| `--min-steps` | `2` | Minimum tool steps per chain |
| `--max-steps` | `5` | Maximum tool steps per chain |
| `--domain` | None | Restrict to a specific ToolBench category |

### `toolgen evaluate`

Reads a JSONL dataset and computes evaluation metrics.

| Flag | Default | Description |
|------|---------|-------------|
| `--dataset` | required | Path to JSONL file |
| `--output` | None | Write JSON report to this path |
| `--threshold` | `3.5` | Score threshold for pass/fail |

## Running Tests

```bash
# All tests
pytest tests/ -v

# Unit tests only
pytest tests/unit/ -v

# Integration test (retry/repair loop — no API key needed, uses mocks)
pytest tests/integration/test_retry_repair.py -v
```

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
    "generated_at": "2026-04-09T12:00:00Z",
    "model": "gpt-4o-mini"
  }
}
```

## Architecture

See [DESIGN.md](DESIGN.md) for full system design documentation including:
- Component architecture and communication protocol
- Tool Graph construction and sampling strategy
- Context management design (within-conversation grounding + cross-conversation steering)
- Prompt design with iteration history
- Diversity & quality analysis

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
  integration/        Retry/repair loop integration test
```
