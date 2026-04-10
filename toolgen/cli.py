"""
toolgen CLI — three commands: build, generate, evaluate.

Usage:
    toolgen build   --data-dir ./toolbench_data --output-dir ./artifacts
    toolgen generate --artifacts-dir ./artifacts --output ./dataset.jsonl --n 100 --seed 42
    toolgen generate --artifacts-dir ./artifacts --output ./dataset_a.jsonl --n 100 --seed 42 --no-cross-conversation-steering
    toolgen evaluate --dataset ./dataset.jsonl
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import click
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("toolgen.cli")


def _get_llm_client(provider: str = "auto"):
    """
    Return an LLM client. Auto-detection priority:
      1. GROQ_API_KEY      → Groq (llama-3.3-70b, free tier, OpenAI-compatible)
      2. ANTHROPIC_API_KEY → Anthropic adapter
      3. OPENAI_API_KEY    → OpenAI

    All returned clients expose the same OpenAI-compatible interface.
    """
    import openai
    from toolgen.llm_client import AnthropicAdapter

    groq_key      = os.environ.get("GROQ_API_KEY")
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    openai_key    = os.environ.get("OPENAI_API_KEY")

    if provider == "groq" or (provider == "auto" and groq_key):
        if not groq_key:
            raise click.ClickException("GROQ_API_KEY not set in .env")
        click.echo("Backend: Groq (llama-3.3-70b-versatile)")
        # Groq is OpenAI-compatible. We patch the default model name so
        # every agent that passes "gpt-4o-mini" gets routed to the right model.
        client = openai.OpenAI(
            api_key=groq_key,
            base_url="https://api.groq.com/openai/v1",
        )
        client._toolgen_model_override = "llama-3.1-8b-instant"
        return client

    if provider == "anthropic" or (provider == "auto" and anthropic_key):
        if not anthropic_key:
            raise click.ClickException("ANTHROPIC_API_KEY not set in .env")
        click.echo("Backend: Anthropic (claude-haiku-4-5-20251001)")
        return AnthropicAdapter(api_key=anthropic_key)

    if provider == "openai" or (provider == "auto" and openai_key):
        if not openai_key:
            raise click.ClickException("OPENAI_API_KEY not set in .env")
        click.echo("Backend: OpenAI (gpt-4o-mini)")
        return openai.OpenAI(api_key=openai_key)

    raise click.ClickException(
        "No API key found. Set GROQ_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY in .env"
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool):
    """Multi-agent tool-use conversation generator."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)


# =============================================================================
# build command
# =============================================================================

@cli.command()
@click.option("--data-dir", required=True, type=click.Path(exists=True),
              help="Path to ToolBench data directory or JSON file")
@click.option("--output-dir", default="./artifacts", show_default=True,
              type=click.Path(), help="Directory to write artifacts (graph, registry)")
@click.option("--min-confidence", default=0.25, show_default=True, type=float,
              help="Minimum edge confidence for Tool Graph")
@click.option("--limit", default=None, type=int,
              help="Limit number of tools loaded (useful for testing)")
def build(data_dir: str, output_dir: str, min_confidence: float, limit: int):
    """
    Ingest ToolBench data and build the Tool Graph and registry artifacts.

    Artifacts written:
      {output_dir}/registry.pkl    — list of normalized Tool objects
      {output_dir}/graph.pkl       — ToolGraph (networkx DiGraph)
      {output_dir}/stats.json      — graph statistics
    """
    from toolgen.graph.builder import ToolGraphBuilder
    from toolgen.registry.loader import ToolBenchLoader

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Load registry
    click.echo(f"Loading ToolBench data from {data_dir}...")
    loader = ToolBenchLoader()
    tools = loader.load(data_dir)
    if limit:
        tools = tools[:limit]
    click.echo(f"Loaded {len(tools)} tools")

    # Save registry
    registry_path = out / "registry.pkl"
    with open(registry_path, "wb") as f:
        pickle.dump(tools, f)
    click.echo(f"Registry saved to {registry_path}")

    # Build graph
    click.echo("Building Tool Graph...")
    builder = ToolGraphBuilder(min_confidence=min_confidence)
    graph = builder.build(tools)
    stats = graph.stats()

    graph_path = out / "graph.pkl"
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)

    stats_path = out / "stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    click.echo(f"Graph saved to {graph_path}")
    click.echo(f"Stats: {json.dumps(stats, indent=2)}")


# =============================================================================
# generate command
# =============================================================================

@cli.command()
@click.option("--artifacts-dir", default="./artifacts", show_default=True,
              type=click.Path(exists=True), help="Directory with build artifacts")
@click.option("--output", default="./output/dataset.jsonl", show_default=True,
              type=click.Path(), help="Output JSONL path")
@click.option("--n", default=100, show_default=True, type=int,
              help="Number of conversations to generate")
@click.option("--seed", default=42, show_default=True, type=int,
              help="Random seed for reproducibility")
@click.option("--model", default="gpt-4o-mini", show_default=True,
              help="LLM model to use")
@click.option("--no-cross-conversation-steering", "no_steering",
              is_flag=True, default=False,
              help="Disable cross-conversation diversity steering (Run A)")
@click.option("--min-steps", default=2, show_default=True, type=int)
@click.option("--max-steps", default=5, show_default=True, type=int)
@click.option("--domain", default=None, type=str,
              help="Restrict to a specific ToolBench domain/category")
@click.option("--provider", default="auto", show_default=True,
              type=click.Choice(["auto", "groq", "anthropic", "openai"]),
              help="LLM provider to use")
def generate(
    artifacts_dir: str,
    output: str,
    n: int,
    seed: int,
    model: str,
    no_steering: bool,
    min_steps: int,
    max_steps: int,
    domain: str,
    provider: str,
):
    """
    Generate synthetic tool-use conversations.

    Run A (no steering):  toolgen generate --no-cross-conversation-steering --seed 42
    Run B (with steering): toolgen generate --seed 42
    """
    import pickle
    from tqdm import tqdm

    from toolgen.agents.assistant import AssistantAgent
    from toolgen.agents.judge import JudgeAgent
    from toolgen.agents.orchestrator import Orchestrator
    from toolgen.agents.planner import PlannerAgent
    from toolgen.agents.user import UserSimulatorAgent
    from toolgen.executor.mock_generator import MockGenerator
    from toolgen.graph.coverage import CoverageTracker
    from toolgen.graph.sampler import ChainSampler, SamplingConstraint
    from toolgen.output.writer import ConversationWriter

    steering_enabled = not no_steering
    art = Path(artifacts_dir)

    click.echo(f"Loading artifacts from {art}...")
    with open(art / "registry.pkl", "rb") as f:
        tools_list = pickle.load(f)
    with open(art / "graph.pkl", "rb") as f:
        graph = pickle.load(f)

    tools_map = {t.id: t for t in tools_list}
    click.echo(f"Loaded {len(tools_list)} tools, graph has {len(graph)} nodes")

    # Initialize LLM client
    llm = _get_llm_client(provider)
    # If provider overrides the model (e.g. Groq), use it
    if hasattr(llm, "_toolgen_model_override"):
        model = llm._toolgen_model_override

    # Initialize components
    mock_gen = MockGenerator(llm_client=llm, llm_model=model, seed=seed)
    # Separate coverage files for Run A vs B so they don't share state
    run_label = "B" if steering_enabled else "A"
    tracker = CoverageTracker(
        persist_path=art / f"coverage_{run_label}.json",
        use_mem0=False,  # set True to enable semantic tracking
    )
    sampler = ChainSampler(
        graph=graph,
        tracker=tracker,
        seed=seed,
        steering_enabled=steering_enabled,
    )

    planner = PlannerAgent(llm, model=model)
    user_agent = UserSimulatorAgent(llm, model=model)
    assistant = AssistantAgent(llm, model=model)
    judge = JudgeAgent(llm, model=model)

    orchestrator = Orchestrator(
        planner=planner,
        user_agent=user_agent,
        assistant_agent=assistant,
        judge_agent=judge,
        mock_generator=mock_gen,
        seed=seed,
        model_name=model,
    )

    constraint = SamplingConstraint(
        min_steps=min_steps,
        max_steps=max_steps,
        required_domains=[domain] if domain else [],
    )

    # Run generation
    mode = "Run B (steering ON)" if steering_enabled else "Run A (steering OFF)"
    click.echo(f"Generating {n} conversations — {mode} — seed={seed}")

    generated = 0
    discarded = 0

    with ConversationWriter(output, tracker=tracker, steering_enabled=steering_enabled) as writer:
        with tqdm(total=n, desc="Generating") as pbar:
            while generated < n:
                chain = sampler.sample(constraint)
                result = orchestrator.generate(
                    chain=chain,
                    tools=tools_map,
                    steering_enabled=steering_enabled,
                    conversation_id=f"conv_{seed}_{generated:04d}",
                )
                if result is not None:
                    writer.write(result)
                    generated += 1
                    pbar.update(1)
                    pbar.set_postfix({"discarded": discarded, "repairs": result.repair_attempts})
                else:
                    discarded += 1

    click.echo(f"\nDone. Generated: {generated}, Discarded: {discarded}")
    click.echo(f"Output: {output}")

    # Print diversity metrics
    metrics = tracker.diversity_metrics()
    click.echo(f"\nDiversity metrics:")
    click.echo(f"  Tool-pair TTR:        {metrics['tool_pair_ttr']:.4f}")
    click.echo(f"  Domain entropy (norm): {metrics['domain_entropy_normalized']:.4f}")
    click.echo(f"  Unique tool pairs:     {metrics['unique_tool_pairs']}")
    click.echo(f"  Domains seen:          {metrics['domains_seen']}")


# =============================================================================
# evaluate command
# =============================================================================

@cli.command()
@click.option("--dataset", required=True, type=click.Path(exists=True),
              help="Path to JSONL dataset file")
@click.option("--output", default=None, type=click.Path(),
              help="Write evaluation report to this JSON file")
@click.option("--threshold", default=3.5, show_default=True, type=float,
              help="Score threshold; conversations below this are flagged")
def evaluate(dataset: str, output: str, threshold: float):
    """
    Validate a generated dataset and compute evaluation metrics.

    Reads a JSONL file and reports:
      - Mean/std judge scores per dimension
      - Pass rate (conversations above threshold)
      - Diversity metrics
      - Dataset composition statistics
    """
    import statistics
    from collections import Counter, defaultdict

    scores = defaultdict(list)
    tool_pairs = Counter()
    domain_counts = Counter()
    num_turns_list = []
    num_tool_calls_list = []
    pass_count = 0
    total = 0
    repair_counts = Counter()

    with open(dataset) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                click.echo(f"Warning: skipping malformed line: {e}", err=True)
                continue

            total += 1
            js = record.get("judge_scores", {})
            meta = record.get("metadata", {})

            scores["tool_selection"].append(js.get("tool_selection", 0))
            scores["naturalness"].append(js.get("naturalness", 0))
            scores["chaining"].append(js.get("chaining", 0))
            scores["overall"].append(js.get("overall", 0))

            if js.get("overall", 0) >= threshold:
                pass_count += 1

            tools = meta.get("tools_used", [])
            for a, b in zip(tools, tools[1:]):
                tool_pairs[(a, b)] += 1

            for cat in meta.get("tool_categories", []):
                domain_counts[cat] += 1

            num_turns_list.append(meta.get("num_turns", 0))
            num_tool_calls_list.append(meta.get("num_tool_calls", 0))
            repair_counts[meta.get("repair_attempts", 0)] += 1

    if total == 0:
        click.echo("No records found in dataset.")
        return

    # Compute diversity metrics
    import math
    total_pairs = sum(tool_pairs.values())
    unique_pairs = len(tool_pairs)
    ttr = unique_pairs / max(total_pairs, 1)

    total_domain = sum(domain_counts.values())
    num_domains = len(domain_counts)
    if num_domains > 1 and total_domain > 0:
        probs = [c / total_domain for c in domain_counts.values()]
        raw_entropy = -sum(p * math.log(p) for p in probs if p > 0)
        domain_entropy = raw_entropy / math.log(num_domains)
    else:
        domain_entropy = 0.0

    report = {
        "total_conversations": total,
        "pass_rate": round(pass_count / total, 4),
        "pass_count": pass_count,
        "threshold": threshold,
        "scores": {
            dim: {
                "mean": round(statistics.mean(vals), 4),
                "std": round(statistics.stdev(vals) if len(vals) > 1 else 0, 4),
                "min": min(vals),
                "max": max(vals),
            }
            for dim, vals in scores.items()
        },
        "diversity": {
            "tool_pair_ttr": round(ttr, 4),
            "domain_entropy_normalized": round(domain_entropy, 4),
            "unique_tool_pairs": unique_pairs,
            "domains_seen": num_domains,
            "top_domains": dict(domain_counts.most_common(10)),
        },
        "composition": {
            "mean_turns": round(statistics.mean(num_turns_list), 2),
            "mean_tool_calls": round(statistics.mean(num_tool_calls_list), 2),
            "repair_distribution": dict(sorted(repair_counts.items())),
        },
    }

    # Print report
    click.echo(f"\n{'='*60}")
    click.echo(f"Dataset Evaluation Report")
    click.echo(f"{'='*60}")
    click.echo(f"Total conversations: {total}")
    click.echo(f"Pass rate (≥{threshold}): {report['pass_rate']:.1%} ({pass_count}/{total})")
    click.echo(f"\nScores (mean ± std):")
    for dim, stats in report["scores"].items():
        click.echo(f"  {dim:20s}: {stats['mean']:.2f} ± {stats['std']:.2f}")
    click.echo(f"\nDiversity:")
    click.echo(f"  Tool-pair TTR:         {ttr:.4f}")
    click.echo(f"  Domain entropy (norm): {domain_entropy:.4f}")
    click.echo(f"  Unique tool pairs:     {unique_pairs}")
    click.echo(f"  Domains seen:          {num_domains}")
    click.echo(f"\nComposition:")
    click.echo(f"  Mean turns/conv:    {report['composition']['mean_turns']}")
    click.echo(f"  Mean tool calls:    {report['composition']['mean_tool_calls']}")

    if output:
        out_path = Path(output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        click.echo(f"\nReport written to {output}")


if __name__ == "__main__":
    cli()
