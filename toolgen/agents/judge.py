"""
Judge Agent — scores conversations and produces repair hints.

Uses structured JSON output for reliable score parsing.
See DESIGN.md §6 for dimension justification.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Optional

from toolgen.context.conversation import ConversationContext

logger = logging.getLogger(__name__)

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI assistant conversations that involve tool use.
You score conversations on three dimensions relevant to training tool-use agents.
Output ONLY valid JSON — no prose, no markdown.\
"""

JUDGE_USER_TEMPLATE = """\
Evaluate this tool-use conversation:

{conversation_text}

Score on three dimensions (1-5 integer each):

1. tool_selection_score: Were the right tools called for the stated goal?
   Were arguments sensible and correctly derived? Were unnecessary calls avoided?
   1 = completely wrong tools, 5 = perfect tool selection throughout.

2. naturalness_score: Did the conversation flow naturally?
   Was disambiguation appropriate (asked when truly needed, not excessive)?
   Did the assistant's narrative match the tool outputs?
   1 = robotic/incoherent, 5 = indistinguishable from a real conversation.

3. chaining_score: Did later tool calls use actual values from earlier outputs?
   (e.g., did the booking call use the ID returned by the search call?)
   Did the final response correctly reference confirmed outputs?
   1 = all IDs hallucinated, 5 = perfect chaining throughout.

Return JSON with exactly these fields:
{{
  "tool_selection_score": <1-5>,
  "tool_selection_reasoning": "<one sentence>",
  "naturalness_score": <1-5>,
  "naturalness_reasoning": "<one sentence>",
  "chaining_score": <1-5>,
  "chaining_reasoning": "<one sentence>",
  "overall_score": <float, mean of the three>,
  "repair_hints": ["<specific actionable fix>", ...],
  "is_repairable": <true if targeted fixes can bring score above 3.5, else false>
}}\
"""


@dataclass
class JudgeResult:
    tool_selection_score: int
    tool_selection_reasoning: str
    naturalness_score: int
    naturalness_reasoning: str
    chaining_score: int
    chaining_reasoning: str
    overall_score: float
    repair_hints: list[str]
    is_repairable: bool

    @property
    def passes(self) -> bool:
        """Whether this conversation passes the quality threshold."""
        return (
            self.overall_score >= 3.5
            and self.tool_selection_score >= 2
            and self.naturalness_score >= 2
            and self.chaining_score >= 2
        )

    def to_dict(self) -> dict:
        return {
            "tool_selection": self.tool_selection_score,
            "naturalness": self.naturalness_score,
            "chaining": self.chaining_score,
            "overall": self.overall_score,
            "reasoning": {
                "tool_selection": self.tool_selection_reasoning,
                "naturalness": self.naturalness_reasoning,
                "chaining": self.chaining_reasoning,
            },
            "repair_hints": self.repair_hints,
            "is_repairable": self.is_repairable,
        }


class JudgeAgent:
    """
    Evaluates generated conversations using an LLM-as-judge.

    Usage:
        judge = JudgeAgent(llm_client)
        result = judge.score(ctx)
        if not result.passes:
            # use result.repair_hints for targeted repair
    """

    PASSING_THRESHOLD = 3.5

    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,   # low temp for consistent scoring
    ):
        self._llm = llm_client
        self._model = model
        self._temperature = temperature

    def score(self, ctx: ConversationContext) -> JudgeResult:
        """Score a completed conversation."""
        # Hard-fail: a conversation with no tool calls cannot score well on
        # tool_selection or chaining regardless of what the LLM judge says.
        if ctx.num_tool_calls == 0:
            return JudgeResult(
                tool_selection_score=1,
                tool_selection_reasoning="No tool calls were made.",
                naturalness_score=3,
                naturalness_reasoning="Conversation had no tool use.",
                chaining_score=1,
                chaining_reasoning="No tool calls to chain.",
                overall_score=round((1 + 3 + 1) / 3, 2),
                repair_hints=["The assistant must call at least one tool to complete the task."],
                is_repairable=True,
            )

        conversation_text = self._format_conversation(ctx)
        user_content = JUDGE_USER_TEMPLATE.format(conversation_text=conversation_text)

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        return self._parse_result(raw)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _format_conversation(self, ctx: ConversationContext) -> str:
        """Format conversation history for the judge prompt."""
        lines = []
        for i, msg in enumerate(ctx.messages):
            if msg.role == "user":
                lines.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                if msg.content:
                    lines.append(f"Assistant: {msg.content}")
                for tc in (msg.tool_calls or []):
                    lines.append(
                        f"  [Tool call: {tc.endpoint} with args {json.dumps(tc.arguments)}]"
                    )
            elif msg.role == "tool":
                output_preview = json.dumps(msg.tool_output or {})[:200]
                lines.append(f"  [Tool output ({msg.tool_id}): {output_preview}]")
        return "\n".join(lines)

    def _call_llm(self, messages: list[dict]) -> dict:
        import re
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=self._temperature,
            )
        except Exception:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature,
            )
        content = response.choices[0].message.content.strip()
        # Extract JSON if wrapped in prose or code fences
        if not content.startswith("{"):
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
            if m:
                content = m.group(1)
            else:
                m = re.search(r"\{.*\}", content, re.DOTALL)
                if m:
                    content = m.group(0)
        return json.loads(content)

    def _parse_result(self, raw: dict) -> JudgeResult:
        """Parse judge output with fallbacks for malformed responses."""
        def clamp(v, lo=1, hi=5) -> int:
            try:
                return max(lo, min(hi, int(v)))
            except (TypeError, ValueError):
                return 3

        tool_sel = clamp(raw.get("tool_selection_score", 3))
        nat = clamp(raw.get("naturalness_score", 3))
        chain = clamp(raw.get("chaining_score", 3))

        # Recompute overall rather than trusting LLM's arithmetic
        overall = round((tool_sel + nat + chain) / 3.0, 2)

        hints = raw.get("repair_hints", [])
        if not isinstance(hints, list):
            hints = []

        return JudgeResult(
            tool_selection_score=tool_sel,
            tool_selection_reasoning=str(raw.get("tool_selection_reasoning", "")),
            naturalness_score=nat,
            naturalness_reasoning=str(raw.get("naturalness_reasoning", "")),
            chaining_score=chain,
            chaining_reasoning=str(raw.get("chaining_reasoning", "")),
            overall_score=overall,
            repair_hints=[str(h) for h in hints],
            is_repairable=bool(raw.get("is_repairable", True)),
        )
