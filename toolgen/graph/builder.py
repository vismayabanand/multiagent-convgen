"""
Tool Graph construction.

Edges represent "can chain into" relationships detected via three methods
(ranked by confidence):
  1. Exact field name match between response outputs and input parameters (0.9)
  2. Semantic name match within the same category (0.6)
  3. Same-category weak edges for parallel use (0.3)

See DESIGN.md §3.2 for the full rationale on why LLM-based edge detection
was not used at this stage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import networkx as nx

from toolgen.registry.models import Tool

logger = logging.getLogger(__name__)

# Minimum confidence to include an edge in the graph
_MIN_CONFIDENCE: float = 0.25


@dataclass
class EdgeData:
    """Metadata stored on each graph edge."""
    matched_field: str           # field name that connects source → target
    confidence: float            # 0.0–1.0
    pattern: str = "sequential"  # "sequential" | "parallel"
    source_field: str = ""       # field name in source response
    target_field: str = ""       # field name in target parameter


class ToolGraph:
    """
    Directed graph where nodes are tools and edges are chaining relationships.

    Wraps networkx.DiGraph and exposes domain-specific accessors.
    """

    def __init__(self, graph: nx.DiGraph, tools: dict[str, Tool]):
        self._g = graph
        self._tools = tools  # tool_id → Tool

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._g

    def get_tool(self, tool_id: str) -> Optional[Tool]:
        return self._tools.get(tool_id)

    def tools_in_category(self, category: str) -> list[Tool]:
        return [t for t in self._tools.values() if t.category == category]

    def successors(self, tool_id: str) -> list[tuple[str, EdgeData]]:
        """Return (successor_tool_id, edge_data) pairs."""
        return [
            (nbr, EdgeData(**self._g.edges[tool_id, nbr]))
            for nbr in self._g.successors(tool_id)
        ]

    def predecessors(self, tool_id: str) -> list[tuple[str, EdgeData]]:
        return [
            (nbr, EdgeData(**self._g.edges[nbr, tool_id]))
            for nbr in self._g.predecessors(tool_id)
        ]

    def edge_confidence(self, src: str, dst: str) -> float:
        if self._g.has_edge(src, dst):
            return self._g.edges[src, dst].get("confidence", 0.0)
        return 0.0

    @property
    def categories(self) -> list[str]:
        return list({t.category for t in self._tools.values()})

    @property
    def all_tool_ids(self) -> list[str]:
        return list(self._tools.keys())

    def __len__(self) -> int:
        return len(self._tools)

    def stats(self) -> dict:
        return {
            "num_tools": len(self._tools),
            "num_edges": self._g.number_of_edges(),
            "num_categories": len(self.categories),
            "avg_out_degree": (
                sum(d for _, d in self._g.out_degree()) / max(len(self._tools), 1)
            ),
        }


class ToolGraphBuilder:
    """
    Builds a ToolGraph from a list of Tool objects.

    Usage:
        builder = ToolGraphBuilder(min_confidence=0.3)
        graph = builder.build(tools)
    """

    def __init__(self, min_confidence: float = _MIN_CONFIDENCE):
        self.min_confidence = min_confidence

    def build(self, tools: list[Tool]) -> ToolGraph:
        """Construct the Tool Graph from a list of normalized tools."""
        g = nx.DiGraph()
        tool_map: dict[str, Tool] = {}

        # Add all nodes first
        for tool in tools:
            g.add_node(tool.id, category=tool.category, use_count=0)
            tool_map[tool.id] = tool

        logger.info(f"Building graph edges for {len(tools)} tools...")

        edge_count = 0
        for i, src in enumerate(tools):
            for dst in tools:
                if src.id == dst.id:
                    continue

                edge = self._detect_edge(src, dst)
                if edge and edge.confidence >= self.min_confidence:
                    g.add_edge(
                        src.id, dst.id,
                        matched_field=edge.matched_field,
                        confidence=edge.confidence,
                        pattern=edge.pattern,
                        source_field=edge.source_field,
                        target_field=edge.target_field,
                    )
                    edge_count += 1

        logger.info(
            f"Graph built: {len(tools)} nodes, {edge_count} edges "
            f"(min_confidence={self.min_confidence})"
        )
        return ToolGraph(g, tool_map)

    # ------------------------------------------------------------------
    # Edge detection methods (priority order)
    # ------------------------------------------------------------------

    def _detect_edge(self, src: Tool, dst: Tool) -> Optional[EdgeData]:
        """Try all detection methods and return the highest-confidence edge."""
        # Method 1: exact name match
        edge = self._exact_name_match(src, dst)
        if edge:
            return edge

        # Method 2: semantic name match (same category)
        edge = self._semantic_name_match(src, dst)
        if edge:
            return edge

        # Method 3: same-category weak edge (parallel candidates)
        if src.category == dst.category:
            return EdgeData(
                matched_field="(same_category)",
                confidence=0.3,
                pattern="parallel",
                source_field="",
                target_field="",
            )

        return None

    def _exact_name_match(self, src: Tool, dst: Tool) -> Optional[EdgeData]:
        """
        High-confidence: src response field name exactly matches dst parameter name.
        """
        if not src.response_fields:
            return None

        src_output_names = {f.name for f in src.response_fields}
        dst_required_names = set(dst.required_params)

        overlap = src_output_names & dst_required_names
        if not overlap:
            return None

        # Pick the most "ID-like" overlapping field first, else any
        id_overlaps = [n for n in overlap if any(
            n.endswith(s) for s in ("_id", "_key", "_token", "_ref")
        )]
        matched = id_overlaps[0] if id_overlaps else next(iter(overlap))

        return EdgeData(
            matched_field=matched,
            confidence=0.9,
            pattern="sequential",
            source_field=matched,
            target_field=matched,
        )

    def _semantic_name_match(self, src: Tool, dst: Tool) -> Optional[EdgeData]:
        """
        Medium-confidence: src has an 'id' or generic field, dst has a
        domain-specific ID param that shares the src's category keyword.
        """
        if not src.response_fields or src.category != dst.category:
            return None

        generic_id_fields = [f for f in src.response_fields
                             if f.name in ("id", "result_id", "item_id", "data_id")]
        if not generic_id_fields:
            return None

        # Check if dst has a param named {category_keyword}_id
        category_keyword = src.category.lower().split()[0]
        domain_id_params = [p for p in dst.parameters
                            if p.name.startswith(category_keyword) and p.is_id_field()]
        if not domain_id_params:
            return None

        src_f = generic_id_fields[0]
        dst_p = domain_id_params[0]

        return EdgeData(
            matched_field=f"{src_f.name}→{dst_p.name}",
            confidence=0.6,
            pattern="sequential",
            source_field=src_f.name,
            target_field=dst_p.name,
        )
