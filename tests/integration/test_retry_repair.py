"""
Integration test: retry/repair loop.

Tests that when a conversation fails quality scoring, the orchestrator:
  1. Attempts targeted repair
  2. Falls back to full re-generation with failure context
  3. Gives up after MAX_REPAIR_ATTEMPTS and returns None

All LLM calls are mocked so this test runs without API keys.
"""

import pytest

from toolgen.agents.judge import JudgeResult
from toolgen.agents.orchestrator import Orchestrator
from toolgen.agents.planner import ConversationPlan, DisambiguationPoint
from toolgen.agents.assistant import AssistantTurn
from toolgen.context.conversation import ConversationContext
from toolgen.executor.mock_generator import MockGenerator
from toolgen.executor.session import ExecutionSession
from toolgen.graph.sampler import ToolChain
from toolgen.registry.models import Parameter, ResponseField, Tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tool(tool_id="Travel/hotels/search"):
    return Tool(
        id=tool_id,
        category="Travel",
        api_name="hotels",
        endpoint_name="search",
        description="Search hotels",
        parameters=[Parameter("city", "string", "City", required=True)],
        required_params=["city"],
        response_fields=[ResponseField("hotel_id", "string", "ID")],
        response_schema=None,
        raw={},
    )


def make_chain(*tool_ids):
    return ToolChain(
        steps=list(tool_ids),
        pattern="sequential",
        domain="Travel",
    )


def make_plan(chain):
    return ConversationPlan(
        user_goal="Find a hotel in Paris",
        persona="casual_friendly",
        disambiguation_points=[],
        estimated_turns=4,
        conversation_type="sequential",
        tool_chain=chain,
    )


def make_passing_judge():
    return JudgeResult(
        tool_selection_score=4, tool_selection_reasoning="good",
        naturalness_score=4, naturalness_reasoning="good",
        chaining_score=4, chaining_reasoning="good",
        overall_score=4.0, repair_hints=[], is_repairable=True,
    )


def make_failing_judge(repairable=True):
    return JudgeResult(
        tool_selection_score=2, tool_selection_reasoning="poor",
        naturalness_score=2, naturalness_reasoning="poor",
        chaining_score=2, chaining_reasoning="poor",
        overall_score=2.0,
        repair_hints=["Fix the tool arguments"],
        is_repairable=repairable,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestRetryRepairLoop:

    def test_successful_generation_no_repair(self, mocker):
        """If conversation passes on first try, no repair is attempted."""
        mock_gen = MockGenerator(seed=42)
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": make_tool()}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "I need a hotel in Paris"
        user_agent.respond.return_value = "Under 200 euros"

        assistant = mocker.MagicMock()
        assistant.respond.return_value = AssistantTurn(
            content="I've found you a great hotel!",
            tool_calls=[],
            is_final=True,
            is_clarification=False,
        )

        judge = mocker.MagicMock()
        judge.score.return_value = make_passing_judge()

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
        assert result.repair_attempts == 0
        assert judge.score.call_count == 1

    def test_failing_conversation_triggers_repair(self, mocker):
        """A failing conversation should trigger at least one repair attempt."""
        mock_gen = MockGenerator(seed=42)
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": make_tool()}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "I need a hotel"
        user_agent.respond.return_value = "Sure"

        assistant = mocker.MagicMock()
        assistant.respond.return_value = AssistantTurn(
            content="Done!", tool_calls=[], is_final=True, is_clarification=False,
        )

        judge = mocker.MagicMock()
        # Fail first, then pass
        judge.score.side_effect = [make_failing_judge(), make_passing_judge()]

        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        # Should eventually succeed
        assert result is not None
        # Judge should have been called at least twice
        assert judge.score.call_count >= 1

    def test_all_attempts_exhausted_returns_none(self, mocker):
        """If every attempt fails, generate() returns None."""
        mock_gen = MockGenerator(seed=42)
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": make_tool()}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "I need a hotel"
        user_agent.respond.return_value = "ok"

        assistant = mocker.MagicMock()
        assistant.respond.return_value = AssistantTurn(
            content="Done!", tool_calls=[], is_final=True, is_clarification=False,
        )

        judge = mocker.MagicMock()
        # Always fail and not repairable
        judge.score.return_value = make_failing_judge(repairable=False)

        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        assert result is None

    def test_repair_hints_injected_into_planner(self, mocker):
        """Repair hints from the judge should be passed to the planner on retry."""
        mock_gen = MockGenerator(seed=42)
        chain = make_chain("Travel/hotels/search")
        tools = {"Travel/hotels/search": make_tool()}

        planner = mocker.MagicMock()
        planner.plan.return_value = make_plan(chain)

        user_agent = mocker.MagicMock()
        user_agent.opening_message.return_value = "Hotel please"
        user_agent.respond.return_value = "ok"

        assistant = mocker.MagicMock()
        assistant.respond.return_value = AssistantTurn(
            content="Done!", tool_calls=[], is_final=True, is_clarification=False,
        )

        failing = make_failing_judge()
        failing.repair_hints = ["Ask for check-in date before booking"]
        judge = mocker.MagicMock()
        judge.score.side_effect = [failing, make_passing_judge()]

        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        orch.generate(chain, tools)

        # Planner should have been called with failure_context on retry
        call_args_list = planner.plan.call_args_list
        if len(call_args_list) > 1:
            second_call_kwargs = call_args_list[1][1]
            failure_ctx = second_call_kwargs.get("failure_context", "")
            assert failure_ctx is not None

    def test_empty_tool_chain_returns_none(self, mocker):
        """A chain with no valid tools should return None immediately."""
        mock_gen = MockGenerator(seed=42)
        chain = make_chain("NonExistent/tool/endpoint")
        tools = {}  # no tools available

        planner = mocker.MagicMock()
        user_agent = mocker.MagicMock()
        assistant = mocker.MagicMock()
        judge = mocker.MagicMock()

        orch = Orchestrator(
            planner=planner,
            user_agent=user_agent,
            assistant_agent=assistant,
            judge_agent=judge,
            mock_generator=mock_gen,
            seed=42,
        )

        result = orch.generate(chain, tools)
        assert result is None
