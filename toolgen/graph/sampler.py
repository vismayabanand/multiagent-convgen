"""
Tool-chain sampler.

Performs constrained weighted random walks on the ToolGraph to produce
ToolChain objects that serve as the skeleton for conversation generation.

See DESIGN.md §3.3 for the constraint interface design.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Literal, Optional

from .builder import ToolGraph
from .coverage import CoverageTracker

logger = logging.getLogger(__name__)


@dataclass
class ToolChain:
    """
    An ordered sequence of tool IDs representing a planned conversation path.

    Parallel steps are represented as lists-within-lists:
      ["tool_a", ["tool_b", "tool_c"], "tool_d"]
    means tool_b and tool_c run in parallel, then tool_d.
    """

    steps: list[str | list[str]]    # tool IDs; nested list = parallel
    pattern: Literal["sequential", "parallel", "mixed"]
    domain: str                      # primary ToolBench category
    metadata: dict = field(default_factory=dict)

    @property
    def flat_tool_ids(self) -> list[str]:
        """Flat list of all tool IDs in order."""
        ids = []
        for step in self.steps:
            if isinstance(step, list):
                ids.extend(step)
            else:
                ids.append(step)
        return ids

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def num_distinct_tools(self) -> int:
        return len(set(self.flat_tool_ids))

    def tool_pairs(self) -> list[tuple[str, str]]:
        """All (tool_a, tool_b) consecutive pairs in the flat sequence."""
        ids = self.flat_tool_ids
        return [(ids[i], ids[i + 1]) for i in range(len(ids) - 1)]


@dataclass
class SamplingConstraint:
    """
    Constraint interface for tool-chain sampling.

    All fields are optional; omitted fields impose no constraint.
    """

    min_steps: int = 2
    max_steps: int = 6
    required_domains: list[str] = field(default_factory=list)
    required_tool_ids: list[str] = field(default_factory=list)
    forbidden_tool_ids: set[str] = field(default_factory=set)
    pattern: Literal["sequential", "parallel", "mixed"] = "sequential"

    # Probability distribution over chain lengths within [min_steps, max_steps]
    # Keys are lengths; values are relative weights (will be normalized).
    length_weights: dict[int, float] = field(
        default_factory=lambda: {2: 0.20, 3: 0.30, 4: 0.30, 5: 0.15, 6: 0.05}
    )


class ChainSampler:
    """
    Samples ToolChain objects from a ToolGraph.

    Usage:
        sampler = ChainSampler(graph, tracker, seed=42)
        chain = sampler.sample(SamplingConstraint(min_steps=3, required_domains=["Travel"]))
    """

    MAX_ATTEMPTS = 15           # per-walk retries before relaxing constraints
    MAX_BACKTRACK = 5           # steps to backtrack when stuck

    def __init__(
        self,
        graph: ToolGraph,
        tracker: Optional[CoverageTracker] = None,
        seed: Optional[int] = None,
        steering_enabled: bool = True,
    ):
        self.graph = graph
        self.tracker = tracker
        self.steering_enabled = steering_enabled
        self._rng = random.Random(seed)

    def sample(self, constraint: Optional[SamplingConstraint] = None) -> ToolChain:
        """
        Sample a ToolChain satisfying the given constraint.

        Falls back to relaxed constraints if no valid chain is found within
        MAX_ATTEMPTS.
        """
        if constraint is None:
            constraint = SamplingConstraint()

        for attempt in range(self.MAX_ATTEMPTS):
            chain = self._try_sample(constraint)
            if chain is not None:
                return chain

            if attempt == self.MAX_ATTEMPTS // 2:
                # Relax required_domains first
                logger.debug("Relaxing required_domains constraint")
                constraint = SamplingConstraint(
                    min_steps=constraint.min_steps,
                    max_steps=constraint.max_steps,
                    required_tool_ids=constraint.required_tool_ids,
                    forbidden_tool_ids=constraint.forbidden_tool_ids,
                    pattern=constraint.pattern,
                    length_weights=constraint.length_weights,
                )

        # Last resort: return whatever we can build
        logger.warning("Could not satisfy all constraints; returning best-effort chain")
        return self._fallback_chain(constraint)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _try_sample(self, c: SamplingConstraint) -> Optional[ToolChain]:
        target_length = self._sample_length(c)

        # Pick start node
        start = self._pick_start(c)
        if start is None:
            return None

        if c.pattern == "parallel":
            return self._sample_parallel(start, c, target_length)
        return self._sample_sequential(start, c, target_length)

    def _sample_sequential(
        self, start: str, c: SamplingConstraint, target_length: int
    ) -> Optional[ToolChain]:
        """Random walk along edges."""
        path = [start]
        forbidden = set(c.forbidden_tool_ids)

        while len(path) < target_length:
            successors = self.graph.successors(path[-1])
            candidates = [
                (nid, edge)
                for nid, edge in successors
                if nid not in forbidden and nid not in path
            ]
            if not candidates:
                if len(path) >= c.min_steps:
                    break  # short chain is acceptable
                return None  # dead end, retry

            weights = [
                edge.confidence * self._node_weight(nid)
                for nid, edge in candidates
            ]
            chosen_id = self._weighted_choice([nid for nid, _ in candidates], weights)
            path.append(chosen_id)

        if not self._satisfies_required_domains(path, c):
            return None
        if not self._satisfies_required_tools(path, c):
            return None

        primary_domain = self.graph.get_tool(path[0]).category
        return ToolChain(
            steps=path,
            pattern="sequential",
            domain=primary_domain,
            metadata={"target_length": target_length},
        )

    def _sample_parallel(
        self, start: str, c: SamplingConstraint, target_length: int
    ) -> Optional[ToolChain]:
        """Fan-out / fan-in pattern."""
        successors = self.graph.successors(start)
        parallel_candidates = [
            nid for nid, edge in successors
            if edge.pattern == "parallel" and nid not in c.forbidden_tool_ids
        ]
        if len(parallel_candidates) < 2:
            # Fall back to sequential
            return self._sample_sequential(start, c, target_length)

        parallel_group = self._rng.sample(
            parallel_candidates, min(3, len(parallel_candidates))
        )
        steps: list[str | list[str]] = [start, parallel_group]

        # Try to find a fan-in node
        for nid in parallel_group:
            for successor_id, _ in self.graph.successors(nid):
                if successor_id not in c.forbidden_tool_ids and successor_id != start:
                    steps.append(successor_id)
                    break
            else:
                continue
            break

        primary_domain = self.graph.get_tool(start).category
        return ToolChain(
            steps=steps,
            pattern="parallel",
            domain=primary_domain,
            metadata={"target_length": target_length},
        )

    def _pick_start(self, c: SamplingConstraint) -> Optional[str]:
        """Select a starting tool node respecting constraints and steering weights."""
        if c.required_tool_ids:
            # If a specific tool is required, it may be forced as start
            candidates = c.required_tool_ids
        elif c.required_domains:
            candidates = [
                t.id for domain in c.required_domains
                for t in self.graph.tools_in_category(domain)
            ]
        else:
            candidates = self.graph.all_tool_ids

        candidates = [c_id for c_id in candidates if c_id not in c.forbidden_tool_ids]
        if not candidates:
            return None

        weights = [self._node_weight(nid) for nid in candidates]
        return self._weighted_choice(candidates, weights)

    def _node_weight(self, tool_id: str) -> float:
        """
        Inverse-frequency weight for diversity steering.
        Returns 1.0 when steering is disabled or tracker has no data.
        """
        if not self.steering_enabled or self.tracker is None:
            return 1.0
        count = self.tracker.tool_use_counts.get(tool_id, 0)
        return 1.0 / (1.0 + math.log1p(count))

    def _sample_length(self, c: SamplingConstraint) -> int:
        valid_lengths = {
            length: weight
            for length, weight in c.length_weights.items()
            if c.min_steps <= length <= c.max_steps
        }
        if not valid_lengths:
            return c.min_steps
        lengths = list(valid_lengths.keys())
        weights = list(valid_lengths.values())
        return self._weighted_choice(lengths, weights)

    def _weighted_choice(self, items: list, weights: list) -> any:
        total = sum(weights)
        if total == 0:
            return self._rng.choice(items)
        r = self._rng.uniform(0, total)
        cumulative = 0.0
        for item, w in zip(items, weights):
            cumulative += w
            if r <= cumulative:
                return item
        return items[-1]

    def _satisfies_required_domains(self, path: list[str], c: SamplingConstraint) -> bool:
        if not c.required_domains:
            return True
        path_domains = {self.graph.get_tool(tid).category for tid in path}
        return all(d in path_domains for d in c.required_domains)

    def _satisfies_required_tools(self, path: list[str], c: SamplingConstraint) -> bool:
        return all(tid in path for tid in c.required_tool_ids)

    def _fallback_chain(self, c: SamplingConstraint) -> ToolChain:
        """Return a minimal chain of any two connected tools."""
        for tool_id in self._rng.sample(self.graph.all_tool_ids,
                                         min(20, len(self.graph.all_tool_ids))):
            succs = self.graph.successors(tool_id)
            if succs:
                succ_id = succs[0][0]
                tool = self.graph.get_tool(tool_id)
                return ToolChain(
                    steps=[tool_id, succ_id],
                    pattern="sequential",
                    domain=tool.category if tool else "unknown",
                    metadata={"fallback": True},
                )
        # Absolute fallback: two random tools, no edge
        ids = self._rng.sample(self.graph.all_tool_ids, 2)
        return ToolChain(steps=ids, pattern="sequential", domain="unknown",
                         metadata={"fallback": True, "no_edge": True})
