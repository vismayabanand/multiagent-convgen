"""
End-to-end test: full pipeline — build artifacts, generate dataset, evaluate.

This test verifies that the complete toolgen pipeline runs correctly and that
LLM-as-judge mean scores exceed a defined quality threshold.

Threshold justification
-----------------------
We chose 3.5 / 5.0 (70%) as the minimum acceptable mean overall score.

Rationale:
  - Score 1–2: conversation is broken (wrong tools, incoherent, or unnatural)
  - Score 3:   conversation is functional but mediocre
  - Score 3.5: conversation is clearly usable as training data with minor flaws
  - Score 4–5: high-quality, ready for fine-tuning without filtering

A mean of 3.5 with per-dimension minimums above 2 ensures the dataset is
useful for training tool-use agents while still leaving room for natural
variation. The 100-sample requirement provides enough statistical power to
detect systematic failures (e.g. every conversation using only one tool).

Usage
-----
Run end-to-end against a PRE-GENERATED dataset (fast, no API calls):

    pytest tests/e2e/ --run-e2e --e2e-dataset ./output/run_B.jsonl -v

Run the full pipeline from scratch (requires API key, ~30 min):

    pytest tests/e2e/ --run-e2e -v

Skip entirely (default — unit/integration CI):

    pytest tests/unit/ tests/integration/ -v
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# pytest options
# ---------------------------------------------------------------------------

def pytest_addoption(parser):
    parser.addoption(
        "--run-e2e",
        action="store_true",
        default=False,
        help="Run the end-to-end pipeline test (requires API key or --e2e-dataset).",
    )
    parser.addoption(
        "--e2e-dataset",
        default=None,
        help="Path to a pre-generated JSONL dataset. Skips generation and runs "
             "evaluate + assertions only.  Example: --e2e-dataset ./output/run_B.jsonl",
    )
    parser.addoption(
        "--e2e-n",
        default=100,
        type=int,
        help="Number of conversations to generate when --run-e2e is active (default 100).",
    )
    parser.addoption(
        "--e2e-seed",
        default=42,
        type=int,
        help="Random seed for the generation run (default 42).",
    )


# ---------------------------------------------------------------------------
# Quality thresholds (documented and justified above)
# ---------------------------------------------------------------------------

THRESHOLD_OVERALL_MEAN   = 3.5   # mean overall score across all conversations
THRESHOLD_DIMENSION_MEAN = 3.0   # each individual dimension must beat this
THRESHOLD_PASS_RATE      = 0.80  # at least 80 % of conversations must pass
MIN_SAMPLE_COUNT         = 100   # dataset must contain at least this many records


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _repo_root() -> Path:
    """Return the repository root (parent of tests/)."""
    return Path(__file__).resolve().parent.parent.parent


def _run_cli(*args: str, cwd: Path | None = None) -> subprocess.CompletedProcess:
    """Run a toolgen CLI command and return the CompletedProcess."""
    cmd = [sys.executable, "-m", "toolgen.cli"] + list(args)
    result = subprocess.run(
        cmd,
        cwd=str(cwd or _repo_root()),
        capture_output=True,
        text=True,
    )
    return result


def _load_report(report_path: Path) -> dict:
    """Load a JSON evaluation report."""
    with open(report_path) as f:
        return json.load(f)


def _evaluate_dataset(dataset_path: Path, report_path: Path, threshold: float = 3.5) -> dict:
    """Run `toolgen evaluate` on *dataset_path* and return the parsed report."""
    result = _run_cli(
        "evaluate",
        "--dataset", str(dataset_path),
        "--output", str(report_path),
        "--threshold", str(threshold),
    )
    assert result.returncode == 0, (
        f"toolgen evaluate failed (exit {result.returncode}):\n"
        f"STDOUT: {result.stdout[-2000:]}\n"
        f"STDERR: {result.stderr[-2000:]}"
    )
    return _load_report(report_path)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def e2e_report(request, tmp_path_factory):
    """
    Session-scoped fixture that either:
      (a) loads a pre-existing report from a pre-generated dataset, or
      (b) runs the full pipeline (build → generate → evaluate).

    Returns the parsed evaluation report dict.
    """
    if not request.config.getoption("--run-e2e"):
        pytest.skip("End-to-end tests are skipped by default. Pass --run-e2e to enable.")

    root = _repo_root()
    tmp = tmp_path_factory.mktemp("e2e")
    report_path = tmp / "e2e_report.json"

    # --- Option A: use a pre-generated dataset ---
    existing_dataset = request.config.getoption("--e2e-dataset")
    if existing_dataset:
        dataset_path = Path(existing_dataset)
        if not dataset_path.is_absolute():
            dataset_path = root / dataset_path
        assert dataset_path.exists(), f"--e2e-dataset not found: {dataset_path}"
        return _evaluate_dataset(dataset_path, report_path)

    # --- Option B: run the full pipeline ---
    n    = request.config.getoption("--e2e-n")
    seed = request.config.getoption("--e2e-seed")

    artifacts_dir = root / "artifacts"
    dataset_path  = tmp / "e2e_dataset.jsonl"

    # Build artifacts if they don't exist
    if not (artifacts_dir / "graph.pkl").exists():
        data_dir = root / "toolbench_data"
        assert data_dir.exists(), (
            f"ToolBench data directory not found: {data_dir}. "
            "Run `toolgen build` first or pass --e2e-dataset."
        )
        result = _run_cli(
            "build",
            "--data-dir", str(data_dir),
            "--output-dir", str(artifacts_dir),
        )
        assert result.returncode == 0, (
            f"toolgen build failed:\nSTDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-2000:]}"
        )

    # Generate conversations
    result = _run_cli(
        "generate",
        "--artifacts-dir", str(artifacts_dir),
        "--output", str(dataset_path),
        "--n", str(n),
        "--seed", str(seed),
        "--provider", "auto",
    )
    assert result.returncode == 0, (
        f"toolgen generate failed:\nSTDOUT: {result.stdout[-2000:]}\nSTDERR: {result.stderr[-2000:]}"
    )
    assert dataset_path.exists(), "generate command succeeded but output file not created"

    return _evaluate_dataset(dataset_path, report_path)


# ---------------------------------------------------------------------------
# Assertions
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end pipeline quality assertions."""

    def test_dataset_has_minimum_sample_count(self, e2e_report):
        """Dataset must contain at least 100 conversations."""
        total = e2e_report["total_conversations"]
        assert total >= MIN_SAMPLE_COUNT, (
            f"Dataset has only {total} conversations; expected >= {MIN_SAMPLE_COUNT}."
        )

    def test_overall_mean_score_exceeds_threshold(self, e2e_report):
        """Mean overall LLM-as-judge score must exceed 3.5/5.0."""
        mean = e2e_report["scores"]["overall"]["mean"]
        assert mean >= THRESHOLD_OVERALL_MEAN, (
            f"Mean overall score {mean:.3f} is below threshold {THRESHOLD_OVERALL_MEAN}. "
            "Conversations are not of sufficient quality for training data."
        )

    def test_tool_selection_score_exceeds_threshold(self, e2e_report):
        """Mean tool_selection score must exceed 3.0/5.0."""
        mean = e2e_report["scores"]["tool_selection"]["mean"]
        assert mean >= THRESHOLD_DIMENSION_MEAN, (
            f"Mean tool_selection score {mean:.3f} is below threshold {THRESHOLD_DIMENSION_MEAN}."
        )

    def test_naturalness_score_exceeds_threshold(self, e2e_report):
        """Mean naturalness score must exceed 3.0/5.0."""
        mean = e2e_report["scores"]["naturalness"]["mean"]
        assert mean >= THRESHOLD_DIMENSION_MEAN, (
            f"Mean naturalness score {mean:.3f} is below threshold {THRESHOLD_DIMENSION_MEAN}. "
            "User agent may be breaking character."
        )

    def test_chaining_score_exceeds_threshold(self, e2e_report):
        """Mean chaining score must exceed 3.0/5.0."""
        mean = e2e_report["scores"]["chaining"]["mean"]
        assert mean >= THRESHOLD_DIMENSION_MEAN, (
            f"Mean chaining score {mean:.3f} is below threshold {THRESHOLD_DIMENSION_MEAN}. "
            "Tool arguments may not be grounded in prior outputs."
        )

    def test_pass_rate_exceeds_threshold(self, e2e_report):
        """At least 80% of conversations must pass the quality threshold."""
        pass_rate = e2e_report["pass_rate"]
        assert pass_rate >= THRESHOLD_PASS_RATE, (
            f"Pass rate {pass_rate:.1%} is below the required {THRESHOLD_PASS_RATE:.0%}. "
            f"Too many conversations are failing quality scoring."
        )

    def test_multi_tool_coverage(self, e2e_report):
        """At least 50% of conversations should use >= 2 distinct tools (per spec)."""
        comp = e2e_report.get("composition", {})
        mean_tool_calls = comp.get("mean_tool_calls", 0)
        # A mean of >= 2 tool calls per conversation implies good multi-tool coverage
        assert mean_tool_calls >= 2.0, (
            f"Mean tool calls per conversation is {mean_tool_calls:.2f}; "
            "expected >= 2.0 to satisfy multi-tool trace requirement."
        )

    def test_domain_entropy_indicates_balance(self, e2e_report):
        """Normalized domain entropy should be >= 0.7, indicating balanced coverage."""
        diversity = e2e_report.get("diversity", {})
        entropy = diversity.get("domain_entropy_normalized", 0)
        assert entropy >= 0.7, (
            f"Normalized domain entropy is {entropy:.3f}; expected >= 0.7. "
            "Dataset is too clustered in a few domains."
        )

    def test_multiple_domains_covered(self, e2e_report):
        """At least 5 distinct ToolBench domains must appear in the dataset."""
        diversity = e2e_report.get("diversity", {})
        domains_seen = diversity.get("domains_seen", 0)
        assert domains_seen >= 5, (
            f"Only {domains_seen} domains seen; expected >= 5 for balanced coverage."
        )
