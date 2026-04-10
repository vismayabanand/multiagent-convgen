"""
JSONL writer for generated conversations.

Converts GeneratedConversation objects (from the Orchestrator) into
ConversationRecord objects and writes them to a JSONL file.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from toolgen.agents.judge import JudgeResult
from toolgen.agents.orchestrator import GeneratedConversation
from toolgen.agents.planner import ConversationPlan
from toolgen.graph.coverage import CoverageTracker
from toolgen.output.schema import ConversationMetadata, ConversationRecord, JudgeScores

logger = logging.getLogger(__name__)


class ConversationWriter:
    """
    Converts GeneratedConversation → ConversationRecord → JSONL.

    Usage:
        writer = ConversationWriter("output/dataset.jsonl")
        writer.write(generated_conv, steering_enabled=True)
        writer.close()
    """

    def __init__(
        self,
        output_path: str | Path,
        tracker: Optional[CoverageTracker] = None,
        steering_enabled: bool = True,
    ):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._tracker = tracker
        self._steering_enabled = steering_enabled
        self._file = open(self.output_path, "a", encoding="utf-8")
        self._count = 0

    def write(self, generated: GeneratedConversation) -> ConversationRecord:
        """Convert and write a generated conversation to JSONL."""
        record = self._to_record(generated)
        self._file.write(record.to_json() + "\n")
        self._file.flush()
        self._count += 1

        # Update coverage tracker
        if self._tracker is not None:
            self._tracker.record(
                tool_ids=generated.ctx.tools_used,
                domain=generated.plan.tool_chain.domain,
                pattern=generated.plan.tool_chain.pattern,
            )

        return record

    def close(self) -> None:
        self._file.close()
        logger.info(f"Wrote {self._count} conversations to {self.output_path}")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _to_record(self, generated: GeneratedConversation) -> ConversationRecord:
        ctx = generated.ctx
        plan = generated.plan
        judge = generated.judge_result

        # Count disambiguation turns (user turns that follow an assistant question)
        disambiguation_turns = 0
        for i, msg in enumerate(ctx.messages):
            if msg.role == "assistant" and msg.content and "?" in msg.content:
                if i + 1 < len(ctx.messages) and ctx.messages[i + 1].role == "user":
                    disambiguation_turns += 1

        categories = list({
            cat for cat in [
                self._category_from_tool_id(tid)
                for tid in ctx.tools_used
            ] if cat
        })

        return ConversationRecord(
            conversation_id=generated.conversation_id,
            messages=ctx.to_messages_list(),
            judge_scores=JudgeScores(
                tool_selection=judge.tool_selection_score,
                naturalness=judge.naturalness_score,
                chaining=judge.chaining_score,
                overall=judge.overall_score,
                reasoning={
                    "tool_selection": judge.tool_selection_reasoning,
                    "naturalness": judge.naturalness_reasoning,
                    "chaining": judge.chaining_reasoning,
                },
                repair_hints=judge.repair_hints,
            ),
            metadata=ConversationMetadata(
                seed=ctx.metadata.get("seed"),
                tools_used=ctx.tools_used,
                tool_categories=categories,
                num_turns=ctx.num_turns,
                num_tool_calls=ctx.num_tool_calls,
                num_distinct_tools=len(set(ctx.tools_used)),
                conversation_type=plan.conversation_type,
                disambiguation_turns=disambiguation_turns,
                repair_attempts=generated.repair_attempts,
                steering_enabled=self._steering_enabled,
                generated_at=ctx.metadata.get("generated_at", ""),
                model=ctx.metadata.get("model", ""),
            ),
        )

    @staticmethod
    def _category_from_tool_id(tool_id: str) -> Optional[str]:
        """Extract category from canonical tool ID (category/api/endpoint)."""
        parts = tool_id.split("/")
        return parts[0] if parts else None
