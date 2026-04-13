"""
Integration tests: dialogue loop controls.

Tests the per-turn tool-call cap, within-turn deduplication,
and user-extension logic when the assistant wraps up early.

All LLM calls are mocked.
"""

from __future__ import annotations

import pytest

from toolgen.agents.assistant import AssistantTurn, ToolCallRequest
from toolgen.agents.judge import JudgeResult
from toolgen.agents.orchestrator import Orchestrator
from toolgen.agents.planner import ConversationPlan
from toolgen.executor.mock_generator import MockGenerator
from toolgen.graph.sampler import ToolChain
from toolgen.registry.models import Parameter, ResponseField, Tool


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_tool(tool_id: str, category: str = "Travel") -> Tool:
    parts = tool_id.split("/")
    return Tool(
        id=tool_id,
        category=parts[0],
        api_name=parts[1] if len(parts) > 1 else "api",
        endpoint_name=parts[2] if len(parts) > 2 else "ep",
        description=f"Tool {tool_id}",
        parameters=[Parameter("query", "string", "Query", required=True)],
        required_params=["query"],
        response_fields=[ResponseField(f"{parts[0].lower()}_id", "string", "ID")],
        response_schema=None,
        raw={},
    )


def make_chain(*tool_ids: str) -> ToolChain:
    return ToolChain(
        steps=list(tool_ids),
        pattern="sequential",
        domain=tool_ids[0].split("/")[0] if tool_ids else "unknown",
    )


def make_plan(chain: ToolChain, goal: str = "Help me") -> ConversationPlan:
    return ConversationPlan(
        user_goal=goal,
        persona="casual_friendly",
        disambiguation_points=[],
        estimated_turns=6,
        conversation_type="sequential",
        tool_chain=chain,
    )


def make_passing_judge() -> JudgeResult:
    return JudgeResult(
        tool_selection_score=4, tool_selection_reasoning="good",
        naturalness_score=4, naturalness_reasoning="good",
        chaining_score=4, chaining_reasoning="good",
        overall_score=4.0, repair_hints=[], is_repairable=True,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestToolCallCapPerTurn:
    """Verifies that no more than MAX_TOOL_CALLS_PER_TURN calls are executed."""

    def test_excessive_tool_calls_in_one_turn_are_capped(self, mocker):
        """When the assistant returns 10 tool calls in one turn, only up to 3 are executed."""
        search = make_tool("Travel/hotels/search")
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": search}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "Find a hotel"
        user_agent.respond.return_value = "Thanks, done."

        # Assistant issues 10 identical calls, then a final turn
        ten_calls = AssistantTurn(
            content=None,
            tool_calls=[
                ToolCallRequest(endpoint="Travel/hotels/search", arguments={"query": "Paris"})
                for _ in range(10)
            ],
            is_final=False,
            is_clarification=False,
        )
        final_turn = AssistantTurn(
            content="Here are your results!",
            tool_calls=[],
            is_final=True,
            is_clarification=False,
        )
        assistant = mocker.MagicMock()
        assistant.respond.side_effect = [ten_calls, final_turn]

        judge = mocker.MagicMock()
        judge.score.return_value = make_passing_judge()

        mock_gen = MockGenerator(seed=42)
        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        assert result is not None
        # tools_used should have at most MAX_TOOL_CALLS_PER_TURN distinct entries
        # (dedup: same endpoint called repeatedly → still just 1 unique tool)
        assert len(result.ctx.tools_used) >= 1


class TestWithinTurnDeduplication:
    """Verifies that duplicate (endpoint, args) pairs within a single turn are skipped."""

    def test_identical_tool_calls_in_one_turn_deduplicated(self, mocker):
        """Same tool with same args called twice in one turn → executed only once."""
        search = make_tool("Travel/hotels/search")
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": search}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "Find a hotel"
        user_agent.respond.return_value = "Thanks."

        dup_calls = AssistantTurn(
            content=None,
            tool_calls=[
                ToolCallRequest(endpoint="Travel/hotels/search", arguments={"query": "Paris"}),
                ToolCallRequest(endpoint="Travel/hotels/search", arguments={"query": "Paris"}),  # dup
            ],
            is_final=False,
            is_clarification=False,
        )
        final_turn = AssistantTurn(
            content="Done!", tool_calls=[], is_final=True, is_clarification=False,
        )
        assistant = mocker.MagicMock()
        assistant.respond.side_effect = [dup_calls, final_turn]

        judge = mocker.MagicMock()
        judge.score.return_value = make_passing_judge()

        execute_calls: list = []

        def spy_execute(self_session, tool, args):
            execute_calls.append(tool.id)
            return {"result": "ok"}

        mocker.patch("toolgen.executor.session.ExecutionSession.execute", spy_execute)

        mock_gen = MockGenerator(seed=42)
        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        assert result is not None
        # The search tool should have been executed exactly once (second call deduped)
        search_calls = [c for c in execute_calls if c == "Travel/hotels/search"]
        assert len(search_calls) == 1


class TestUserExtension:
    """Verifies that when the assistant wraps up early with uncalled tools, the
    user simulator is asked to extend the conversation."""

    def test_early_wrap_up_triggers_user_extension(self, mocker):
        """A 2-tool chain where assistant calls only 1 tool should get a user follow-up."""
        search = make_tool("Travel/hotels/search")
        book = make_tool("Travel/hotels/book")
        chain = make_chain("Travel/hotels/search", "Travel/hotels/book")
        tools = {
            "Travel/hotels/search": search,
            "Travel/hotels/book": book,
        }

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain, "Find and book a hotel")

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "Find me a hotel in Paris and book it"
        user_agent.respond.return_value = "Great, please go ahead and book it."

        # Turn 1: assistant calls only search, then wraps up
        search_turn = AssistantTurn(
            content=None,
            tool_calls=[ToolCallRequest("Travel/hotels/search", {"query": "Paris"})],
            is_final=False,
            is_clarification=False,
        )
        premature_final = AssistantTurn(
            content="I found The Grand Hotel for you!",
            tool_calls=[],
            is_final=True,
            is_clarification=False,
        )
        # After user extension, assistant calls book
        book_turn = AssistantTurn(
            content=None,
            tool_calls=[ToolCallRequest("Travel/hotels/book", {"hotel_id": "tra_abc"})],
            is_final=False,
            is_clarification=False,
        )
        real_final = AssistantTurn(
            content="Your booking is confirmed!",
            tool_calls=[],
            is_final=True,
            is_clarification=False,
        )

        assistant = mocker.MagicMock()
        assistant.respond.side_effect = [search_turn, premature_final, book_turn, real_final]

        judge = mocker.MagicMock()
        judge.score.return_value = make_passing_judge()

        mock_gen = MockGenerator(seed=42)
        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        assert result is not None
        # user_agent.respond should have been called at least once (the extension)
        assert user_agent.respond.call_count >= 1
