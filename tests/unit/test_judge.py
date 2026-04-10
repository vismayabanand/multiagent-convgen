"""Unit tests for the Judge agent — score parsing and threshold logic."""

import pytest

from toolgen.agents.judge import JudgeAgent, JudgeResult


# ---------------------------------------------------------------------------
# JudgeResult tests (no LLM needed)
# ---------------------------------------------------------------------------

class TestJudgeResult:

    def make_result(self, tool_sel=4, nat=4, chain=4, hints=None, repairable=True):
        return JudgeResult(
            tool_selection_score=tool_sel,
            tool_selection_reasoning="ok",
            naturalness_score=nat,
            naturalness_reasoning="ok",
            chaining_score=chain,
            chaining_reasoning="ok",
            overall_score=round((tool_sel + nat + chain) / 3, 2),
            repair_hints=hints or [],
            is_repairable=repairable,
        )

    def test_passes_above_threshold(self):
        result = self.make_result(4, 4, 4)
        assert result.passes is True

    def test_fails_below_mean_threshold(self):
        result = self.make_result(2, 2, 2)
        assert result.passes is False

    def test_fails_when_any_dimension_below_2(self):
        result = self.make_result(5, 5, 1)  # chaining is 1
        assert result.passes is False

    def test_borderline_passes(self):
        # mean exactly 3.5 should pass
        result = self.make_result(4, 3, 3)  # mean = 3.33 — actually fails
        assert result.passes is False

        result2 = self.make_result(4, 4, 3)  # mean = 3.67 — passes
        assert result2.passes is True

    def test_to_dict_structure(self):
        result = self.make_result()
        d = result.to_dict()
        assert "tool_selection" in d
        assert "naturalness" in d
        assert "chaining" in d
        assert "overall" in d
        assert "repair_hints" in d
        assert "is_repairable" in d
        assert "reasoning" in d


# ---------------------------------------------------------------------------
# JudgeAgent._parse_result tests (mocking the LLM call)
# ---------------------------------------------------------------------------

class TestJudgeAgentParsing:

    @pytest.fixture
    def judge(self, mocker):
        mock_llm = mocker.MagicMock()
        return JudgeAgent(mock_llm, model="gpt-4o-mini")

    def test_parses_valid_response(self, judge):
        raw = {
            "tool_selection_score": 4,
            "tool_selection_reasoning": "Good tool selection",
            "naturalness_score": 3,
            "naturalness_reasoning": "Somewhat natural",
            "chaining_score": 5,
            "chaining_reasoning": "Perfect chaining",
            "overall_score": 4.0,
            "repair_hints": [],
            "is_repairable": True,
        }
        result = judge._parse_result(raw)
        assert result.tool_selection_score == 4
        assert result.naturalness_score == 3
        assert result.chaining_score == 5
        assert result.overall_score == 4.0  # recomputed
        assert result.repair_hints == []

    def test_clamps_scores_to_valid_range(self, judge):
        raw = {
            "tool_selection_score": 10,  # out of range
            "naturalness_score": 0,       # out of range
            "chaining_score": 3,
        }
        result = judge._parse_result(raw)
        assert result.tool_selection_score == 5   # clamped to 5
        assert result.naturalness_score == 1      # clamped to 1
        assert result.chaining_score == 3

    def test_handles_missing_fields_with_defaults(self, judge):
        result = judge._parse_result({})
        assert result.tool_selection_score == 3  # default
        assert result.naturalness_score == 3
        assert result.chaining_score == 3
        assert result.repair_hints == []

    def test_overall_score_recomputed_from_dimensions(self, judge):
        raw = {
            "tool_selection_score": 5,
            "naturalness_score": 3,
            "chaining_score": 4,
            "overall_score": 1.0,  # wrong value — should be recomputed
        }
        result = judge._parse_result(raw)
        assert result.overall_score == pytest.approx((5 + 3 + 4) / 3, rel=0.01)

    def test_repair_hints_non_list_handled(self, judge):
        raw = {
            "tool_selection_score": 3,
            "naturalness_score": 3,
            "chaining_score": 3,
            "repair_hints": "fix the args",  # should be a list
        }
        result = judge._parse_result(raw)
        assert isinstance(result.repair_hints, list)
        assert result.repair_hints == []
