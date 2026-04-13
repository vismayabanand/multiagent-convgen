"""
Planner Agent — produces a structured ConversationPlan from a tool chain.

Uses JSON-mode structured output (satisfying the "at least one agent must use
structured output" requirement). See DESIGN.md §5.1 and §8.1.
"""

from __future__ import annotations

import datetime
import json
import logging
from dataclasses import dataclass, field
from typing import Literal, Optional

from toolgen.graph.sampler import ToolChain
from toolgen.registry.models import Tool

logger = logging.getLogger(__name__)

PLANNER_SYSTEM_PROMPT = """\
You are a conversation planner for a tool-use training dataset.

Given a sequence of API tools, you design realistic user scenarios where:
- A user has a concrete, achievable goal
- Some information is naturally withheld by the user (requiring the assistant to ask)
- The conversation uses exactly the tools provided — no more, no less

Output ONLY valid JSON. No prose, no markdown fences.\
"""

PLANNER_USER_TEMPLATE = """\
Tool chain to plan a conversation around:
{tool_chain_json}

Tool descriptions:
{tool_descriptions}

Produce a JSON plan with exactly these fields:
{{
  "user_goal": "<a natural language goal achievable with exactly these tools>",
  "persona": "<one of: terse_professional | casual_friendly | technical_expert | confused_novice>",
  "disambiguation_points": [
    {{
      "before_tool_index": <int, 0-based index of tool in chain>,
      "missing_field": "<exact parameter name that is missing>",
      "assistant_question": "<natural question to ask the user>"
    }}
  ],
  "estimated_turns": <integer 3-10>,
  "conversation_type": "<sequential | parallel | mixed>"
}}

Today's date: {today}

Rules:
- user_goal must be something a real person would ask (not API jargon)
- Any dates in the scenario (travel dates, booking dates, deadlines) must be AFTER today's date
- disambiguation_points should only cover parameters a real user would naturally know (budget, destination, dates, preferences)
- NEVER create a disambiguation point for IDs (hotel_id, flight_id, booking_ref, user_id, etc.) — these come from tool outputs, not from users
- Keep disambiguation_points to 0-2 items (don't interrogate the user)
- estimated_turns = 2 * (tool_steps + len(disambiguation_points)) approximately\
"""


@dataclass
class DisambiguationPoint:
    before_tool_index: int
    missing_field: str
    assistant_question: str


@dataclass
class ConversationPlan:
    user_goal: str
    persona: Literal[
        "terse_professional", "casual_friendly",
        "technical_expert", "confused_novice"
    ]
    disambiguation_points: list[DisambiguationPoint]
    estimated_turns: int
    conversation_type: Literal["sequential", "parallel", "mixed"]
    tool_chain: ToolChain

    # Populated during generation
    failure_context: Optional[str] = None  # set during repair attempts


class PlannerAgent:
    """
    Generates a ConversationPlan from a tool chain using structured LLM output.

    Usage:
        planner = PlannerAgent(llm_client)
        plan = planner.plan(chain, tools)
    """

    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
    ):
        self._llm = llm_client
        self._model = model
        self._temperature = temperature

    def plan(
        self,
        chain: ToolChain,
        tools: dict[str, Tool],
        failure_context: Optional[str] = None,
    ) -> ConversationPlan:
        """Generate a conversation plan for the given tool chain."""
        tool_chain_json = json.dumps(
            {"steps": chain.steps, "pattern": chain.pattern, "domain": chain.domain},
            indent=2,
        )

        tool_descriptions = "\n".join(
            f"- {tid}: {tools[tid].description}" if tid in tools else f"- {tid}: (unknown)"
            for tid in chain.flat_tool_ids
        )

        user_content = PLANNER_USER_TEMPLATE.format(
            tool_chain_json=tool_chain_json,
            tool_descriptions=tool_descriptions,
            today=datetime.date.today().isoformat(),
        )

        # Inject repair context if this is a retry
        if failure_context:
            user_content += (
                f"\n\nPrior attempt failed. Avoid these issues:\n{failure_context}"
            )

        messages = [
            {"role": "system", "content": PLANNER_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        raw = self._call_llm(messages)
        return self._parse_plan(raw, chain, failure_context)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> dict:
        # Try with json_object mode first; fall back to plain text + extraction
        # (Groq supports json_object for llama-3.3-70b but not all models)
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
        content = response.choices[0].message.content
        return json.loads(self._extract_json(content))

    @staticmethod
    def _extract_json(text: str) -> str:
        import re
        text = text.strip()
        if text.startswith("{"):
            return text
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            return m.group(1)
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            return m.group(0)
        return text

    def _parse_plan(
        self, raw: dict, chain: ToolChain, failure_context: Optional[str]
    ) -> ConversationPlan:
        """Parse raw LLM JSON into a ConversationPlan, with fallbacks."""
        disambiguation_points = []
        for dp in raw.get("disambiguation_points", []):
            if isinstance(dp, dict):
                disambiguation_points.append(DisambiguationPoint(
                    before_tool_index=int(dp.get("before_tool_index", 0)),
                    missing_field=str(dp.get("missing_field", "")),
                    assistant_question=str(dp.get("assistant_question", "")),
                ))

        valid_personas = {
            "terse_professional", "casual_friendly",
            "technical_expert", "confused_novice",
        }
        persona = raw.get("persona", "casual_friendly")
        if persona not in valid_personas:
            persona = "casual_friendly"

        valid_types = {"sequential", "parallel", "mixed"}
        conv_type = raw.get("conversation_type", "sequential")
        if conv_type not in valid_types:
            conv_type = "sequential"

        return ConversationPlan(
            user_goal=str(raw.get("user_goal", "Help me with a task")),
            persona=persona,
            disambiguation_points=disambiguation_points,
            estimated_turns=max(3, int(raw.get("estimated_turns", 5))),
            conversation_type=conv_type,
            tool_chain=chain,
            failure_context=failure_context,
        )
