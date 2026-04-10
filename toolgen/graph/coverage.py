"""
Coverage Tracker — cross-conversation diversity steering.

Maintains lightweight counters of what has been generated so far,
exposing them to the ChainSampler as inverse-frequency weights.

Optionally integrates with mem0 for semantic-similarity tracking
across pipeline restarts (see DESIGN.md §7.2).
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CoverageTracker:
    """
    Tracks tool usage across generated conversations to enable
    cross-conversation diversity steering.

    Usage:
        tracker = CoverageTracker()
        tracker.record(chain)               # after each generation
        weight = tracker.node_weight(tool_id)  # used by ChainSampler
    """

    def __init__(
        self,
        persist_path: Optional[str | Path] = None,
        use_mem0: bool = False,
        mem0_config: Optional[dict] = None,
    ):
        self.tool_use_counts: Counter[str] = Counter()
        self.tool_pair_counts: Counter[tuple[str, str]] = Counter()
        self.domain_counts: Counter[str] = Counter()
        self.pattern_counts: Counter[str] = Counter()

        self.persist_path = Path(persist_path) if persist_path else None
        self._mem0_client = None

        if use_mem0:
            self._init_mem0(mem0_config or {})

        if self.persist_path and self.persist_path.exists():
            self._load(self.persist_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, tool_ids: list[str], domain: str, pattern: str) -> None:
        """Record a generated conversation's tool usage."""
        for tid in tool_ids:
            self.tool_use_counts[tid] += 1
        for pair in zip(tool_ids, tool_ids[1:]):
            self.tool_pair_counts[pair] += 1
        self.domain_counts[domain] += 1
        self.pattern_counts[pattern] += 1

        if self.persist_path:
            self._save(self.persist_path)

    def node_weight(self, tool_id: str) -> float:
        """
        Inverse-frequency weight: 1.0 for unseen tools, lower for frequently used.
        Formula: 1 / (1 + log(1 + count))
        """
        count = self.tool_use_counts.get(tool_id, 0)
        return 1.0 / (1.0 + math.log1p(count))

    def domain_weight(self, domain: str) -> float:
        count = self.domain_counts.get(domain, 0)
        return 1.0 / (1.0 + math.log1p(count))

    def diversity_metrics(self) -> dict:
        """
        Compute the two diversity metrics used for the Run A vs B experiment.

        Metric 1: Tool-Pair Type-Token Ratio (TTR)
            unique_pairs / total_pair_occurrences

        Metric 2: Domain Entropy (normalized)
            -Σ p(domain) * log(p(domain)) / log(|domains|)
        """
        # TTR
        total_pairs = sum(self.tool_pair_counts.values())
        unique_pairs = len(self.tool_pair_counts)
        ttr = unique_pairs / max(total_pairs, 1)

        # Domain entropy
        total_domain = sum(self.domain_counts.values())
        num_domains = len(self.domain_counts)
        if num_domains <= 1 or total_domain == 0:
            entropy = 0.0
        else:
            probs = [c / total_domain for c in self.domain_counts.values()]
            raw_entropy = -sum(p * math.log(p) for p in probs if p > 0)
            entropy = raw_entropy / math.log(num_domains)  # normalize to [0, 1]

        return {
            "tool_pair_ttr": round(ttr, 4),
            "domain_entropy_normalized": round(entropy, 4),
            "unique_tool_pairs": unique_pairs,
            "total_tool_pair_occurrences": total_pairs,
            "domains_seen": num_domains,
            "tool_use_counts": dict(self.tool_use_counts.most_common(20)),
        }

    def reset(self) -> None:
        self.tool_use_counts.clear()
        self.tool_pair_counts.clear()
        self.domain_counts.clear()
        self.pattern_counts.clear()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self, path: Path) -> None:
        data = {
            "tool_use_counts": dict(self.tool_use_counts),
            "tool_pair_counts": {str(k): v for k, v in self.tool_pair_counts.items()},
            "domain_counts": dict(self.domain_counts),
            "pattern_counts": dict(self.pattern_counts),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def _load(self, path: Path) -> None:
        with open(path) as f:
            data = json.load(f)
        self.tool_use_counts = Counter(data.get("tool_use_counts", {}))
        # Restore tuple keys for tool_pair_counts
        self.tool_pair_counts = Counter({
            tuple(k.strip("()").replace("'", "").split(", ")): v
            for k, v in data.get("tool_pair_counts", {}).items()
        })
        self.domain_counts = Counter(data.get("domain_counts", {}))
        self.pattern_counts = Counter(data.get("pattern_counts", {}))
        logger.info(f"Loaded coverage tracker from {path}")

    # ------------------------------------------------------------------
    # mem0 integration (optional)
    # ------------------------------------------------------------------

    def _init_mem0(self, config: dict) -> None:
        try:
            from mem0 import Memory
            self._mem0_client = Memory.from_config(config) if config else Memory()
            logger.info("mem0 client initialized for semantic coverage tracking")
        except ImportError:
            logger.warning("mem0ai not installed; semantic tracking disabled")
        except Exception as e:
            logger.warning(f"mem0 init failed: {e}; semantic tracking disabled")

    def record_semantic(self, conversation_summary: str, user_id: str = "corpus") -> None:
        """
        Store a semantic fingerprint of a conversation in mem0.
        Used for similarity-based steering (soft signal, non-deterministic).
        See DESIGN.md §7.2 for the determinism tradeoff.
        """
        if self._mem0_client is None:
            return
        try:
            self._mem0_client.add(
                messages=[{"role": "user", "content": conversation_summary}],
                user_id=user_id,
            )
        except Exception as e:
            logger.warning(f"mem0 record failed: {e}")

    def semantic_similarity_exists(
        self, query: str, threshold: float = 0.85, user_id: str = "corpus"
    ) -> bool:
        """
        Check if a semantically similar conversation has already been generated.
        Returns False (no similar exists) if mem0 is unavailable.

        Note: non-deterministic due to ANN search. Used only as a soft signal.
        """
        if self._mem0_client is None:
            return False
        try:
            results = self._mem0_client.search(query, user_id=user_id, limit=1)
            if results and results[0].get("score", 0) >= threshold:
                return True
        except Exception as e:
            logger.warning(f"mem0 search failed: {e}")
        return False
