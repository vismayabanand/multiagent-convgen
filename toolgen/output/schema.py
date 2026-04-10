"""
Output schema for the generated dataset.

Each ConversationRecord maps to one JSONL line.
See DESIGN.md §10 for schema design decisions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class JudgeScores:
    tool_selection: int
    naturalness: int
    chaining: int
    overall: float
    reasoning: dict[str, str]
    repair_hints: list[str]


@dataclass
class ConversationMetadata:
    seed: Optional[int]
    tools_used: list[str]
    tool_categories: list[str]
    num_turns: int
    num_tool_calls: int
    num_distinct_tools: int
    conversation_type: str          # "sequential" | "parallel" | "mixed"
    disambiguation_turns: int
    repair_attempts: int
    steering_enabled: bool
    generated_at: str
    model: str


@dataclass
class ConversationRecord:
    """
    Single output record written to the JSONL dataset.

    Format mirrors the example in the problem spec but with additional
    metadata fields for reproducibility and analysis.
    """
    conversation_id: str
    messages: list[dict[str, Any]]
    judge_scores: JudgeScores
    metadata: ConversationMetadata

    def to_dict(self) -> dict:
        return {
            "conversation_id": self.conversation_id,
            "messages": self.messages,
            "judge_scores": {
                "tool_selection": self.judge_scores.tool_selection,
                "naturalness": self.judge_scores.naturalness,
                "chaining": self.judge_scores.chaining,
                "overall": self.judge_scores.overall,
                "reasoning": self.judge_scores.reasoning,
                "repair_hints": self.judge_scores.repair_hints,
            },
            "metadata": {
                "seed": self.metadata.seed,
                "tools_used": self.metadata.tools_used,
                "tool_categories": self.metadata.tool_categories,
                "num_turns": self.metadata.num_turns,
                "num_tool_calls": self.metadata.num_tool_calls,
                "num_distinct_tools": self.metadata.num_distinct_tools,
                "conversation_type": self.metadata.conversation_type,
                "disambiguation_turns": self.metadata.disambiguation_turns,
                "repair_attempts": self.metadata.repair_attempts,
                "steering_enabled": self.metadata.steering_enabled,
                "generated_at": self.metadata.generated_at,
                "model": self.metadata.model,
            },
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
