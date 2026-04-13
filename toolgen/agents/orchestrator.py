"""
Orchestrator — runs the full conversation generation pipeline.

Manages the dialogue loop, routes tool calls to the ExecutionSession,
and runs the repair loop when conversations fail quality checks.

See DESIGN.md §5.3 for the full loop design.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from toolgen.agents.assistant import AssistantAgent, AssistantTurn
from toolgen.context.conversation import ToolCallRef
from toolgen.agents.judge import JudgeAgent, JudgeResult
from toolgen.agents.planner import ConversationPlan, PlannerAgent
from toolgen.agents.user import UserSimulatorAgent
from toolgen.context.conversation import ConversationContext
from toolgen.executor.mock_generator import MockGenerator
from toolgen.executor.session import ExecutionSession
from toolgen.graph.sampler import ToolChain
from toolgen.registry.models import Tool

logger = logging.getLogger(__name__)

MAX_REPAIR_ATTEMPTS = 3
MAX_TURNS_PER_CONVERSATION = 20
MAX_TOOL_CALLS_PER_TURN = 3   # prevent model from issuing 10 identical calls in one turn
MAX_SAME_TOOL_CALLS = 2       # how many times the same tool may be called before we steer


@dataclass
class GeneratedConversation:
    """Final output of the orchestrator for a single conversation."""
    conversation_id: str
    ctx: ConversationContext
    judge_result: JudgeResult
    plan: ConversationPlan
    repair_attempts: int


class Orchestrator:
    """
    Orchestrates the full multi-agent conversation generation pipeline.

    Usage:
        orch = Orchestrator(planner, user, assistant, judge, mock_gen)
        result = orch.generate(chain, tools, seed=42)
    """

    def __init__(
        self,
        planner: PlannerAgent,
        user_agent: UserSimulatorAgent,
        assistant_agent: AssistantAgent,
        judge_agent: JudgeAgent,
        mock_generator: MockGenerator,
        seed: Optional[int] = None,
        model_name: str = "gpt-4o-mini",
    ):
        self.planner = planner
        self.user_agent = user_agent
        self.assistant_agent = assistant_agent
        self.judge_agent = judge_agent
        self.mock_generator = mock_generator
        self.seed = seed
        self.model_name = model_name
        self._rng = random.Random(seed)

    def generate(
        self,
        chain: ToolChain,
        tools: dict[str, Tool],
        steering_enabled: bool = True,
        conversation_id: Optional[str] = None,
    ) -> Optional[GeneratedConversation]:
        """
        Generate a single conversation from a tool chain.

        Returns None only if all repair attempts are exhausted and the
        conversation is deemed unrepairable.
        """
        conv_id = conversation_id or f"conv_{uuid.uuid4().hex[:8]}"
        failure_context: Optional[str] = None
        repair_attempts = 0

        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            if attempt > 0:
                repair_attempts = attempt
                logger.info(f"[{conv_id}] Repair attempt {attempt}/{MAX_REPAIR_ATTEMPTS}")

            # Get available tools for this chain
            chain_tools = [tools[tid] for tid in chain.flat_tool_ids if tid in tools]
            if not chain_tools:
                logger.warning(f"[{conv_id}] No valid tools for chain {chain.steps}")
                return None

            # Plan the conversation
            plan = self.planner.plan(chain, tools, failure_context=failure_context)

            # Run the dialogue
            ctx = self._create_context(conv_id)
            success = self._run_dialogue(ctx, plan, chain_tools)

            if not success:
                logger.warning(f"[{conv_id}] Dialogue loop failed on attempt {attempt}")
                failure_context = "Dialogue loop terminated abnormally. Ensure the conversation reaches a natural conclusion."
                continue

            # Judge the result
            judge_result = self.judge_agent.score(ctx)
            logger.info(
                f"[{conv_id}] Scores: tool_sel={judge_result.tool_selection_score} "
                f"nat={judge_result.naturalness_score} "
                f"chain={judge_result.chaining_score} "
                f"overall={judge_result.overall_score}"
            )

            if judge_result.passes:
                return GeneratedConversation(
                    conversation_id=conv_id,
                    ctx=ctx,
                    judge_result=judge_result,
                    plan=plan,
                    repair_attempts=repair_attempts,
                )

            # Determine repair strategy
            if not judge_result.is_repairable:
                logger.info(f"[{conv_id}] Marked unrepairable by judge, full re-plan")
                failure_context = (
                    "Prior attempt was fundamentally flawed. "
                    + "\n".join(judge_result.repair_hints)
                )
                continue

            # Attempt 1: targeted repair (only on first failure)
            if attempt == 0 and judge_result.repair_hints:
                repaired = self._targeted_repair(ctx, plan, chain_tools, judge_result)
                if repaired is not None:
                    repair_judge = self.judge_agent.score(repaired)
                    if repair_judge.passes:
                        return GeneratedConversation(
                            conversation_id=conv_id,
                            ctx=repaired,
                            judge_result=repair_judge,
                            plan=plan,
                            repair_attempts=repair_attempts + 1,
                        )

            failure_context = (
                "Prior conversation failed with these issues:\n"
                + "\n".join(f"- {h}" for h in judge_result.repair_hints)
                + f"\n\nScores were: tool_selection={judge_result.tool_selection_score}, "
                f"naturalness={judge_result.naturalness_score}, "
                f"chaining={judge_result.chaining_score}"
            )

        logger.warning(f"[{conv_id}] All repair attempts exhausted, discarding")
        return None

    # ------------------------------------------------------------------
    # Dialogue loop
    # ------------------------------------------------------------------

    def _run_dialogue(
        self,
        ctx: ConversationContext,
        plan: ConversationPlan,
        chain_tools: list[Tool],
    ) -> bool:
        """
        Run the main dialogue loop.

        Returns True if the conversation reached a natural conclusion.
        """
        # User opens the conversation
        opening = self.user_agent.opening_message(plan)
        ctx.add_user_message(opening)
        logger.debug(f"[{ctx.conversation_id}] User: {opening[:80]}...")

        tool_index = 0  # tracks which tool in the chain we've reached
        consecutive_assistant_turns = 0
        tool_call_counts: dict[str, int] = {}  # endpoint → total call count this conversation
        pending_steer_hint: str | None = None  # injected before next assistant turn
        user_extensions = 0  # how many times we've kept conversation alive via user follow-up
        max_user_extensions = max(0, len(chain_tools) - 1)

        for turn in range(MAX_TURNS_PER_CONVERSATION):
            # Inject steering hint if the assistant has been looping on one tool.
            # Only inject when the last message is NOT already from the user —
            # otherwise we'd create two consecutive user messages.
            if pending_steer_hint:
                if not ctx.messages or ctx.messages[-1].role != "user":
                    ctx.add_user_message(pending_steer_hint)
                pending_steer_hint = None

            # Check if we should trigger disambiguation before this tool
            self._maybe_inject_disambiguation(ctx, plan, chain_tools, tool_index, turn)

            # Assistant responds
            assistant_turn = self.assistant_agent.respond(ctx, chain_tools)
            consecutive_assistant_turns += 1

            # Handle tool calls
            if assistant_turn.tool_calls:
                consecutive_assistant_turns = 0
                steer_after = False

                # Cap per-turn calls and deduplicate (same endpoint+args)
                seen_in_turn: set[str] = set()
                calls_to_run: list[tuple] = []   # (tc, tool) pairs to execute
                for tc in assistant_turn.tool_calls[:MAX_TOOL_CALLS_PER_TURN]:
                    dedup_key = f"{tc.endpoint}|{sorted(tc.arguments.items())}"
                    if dedup_key in seen_in_turn:
                        continue
                    seen_in_turn.add(dedup_key)

                    tool = self._find_tool(tc.endpoint, chain_tools)
                    if tool is None:
                        logger.warning(f"Assistant called unknown tool: {tc.endpoint}")
                        continue

                    # Track repetition and steer when the model is stuck
                    tool_call_counts[tc.endpoint] = tool_call_counts.get(tc.endpoint, 0) + 1
                    if tool_call_counts[tc.endpoint] > MAX_SAME_TOOL_CALLS:
                        steer_after = True
                        continue  # skip executing the redundant call

                    calls_to_run.append((tc, tool))

                # If ALL calls were filtered out (all redundant), don't add an empty
                # assistant message — fall through to handle as a non-tool turn instead.
                # This prevents the cascade where skipped calls keep triggering steer hints.
                if not calls_to_run and not assistant_turn.content:
                    if steer_after:
                        # Force a natural wrap-up instead of looping
                        pending_steer_hint = self._rng.choice([
                            "Great, can you give me a quick summary of everything?",
                            "Perfect, that's all I needed — thanks!",
                            "Awesome, can you wrap it up for me?",
                        ])
                        steer_after = False
                    consecutive_assistant_turns += 1
                    if consecutive_assistant_turns >= 3:
                        return True
                    continue

                # *** Correct ordering: assistant message (with tool_calls) BEFORE tool outputs ***
                tool_call_refs = [ToolCallRef(tc.endpoint, tc.arguments) for tc, _ in calls_to_run]
                ctx.add_assistant_message(
                    content=assistant_turn.content,
                    tool_calls=tool_call_refs,
                )

                # Now execute each tool and append its output
                for tc, tool in calls_to_run:
                    response = ctx.execution_session.execute(tool, tc.arguments)
                    ctx.add_tool_output(tool.id, response)
                    tool_index = min(tool_index + 1, len(chain_tools) - 1)

                if steer_after:
                    uncalled = [
                        t for t in chain_tools
                        if tool_call_counts.get(t.id, 0) == 0
                    ]
                    # Steer hints must sound like something a real user would say —
                    # they appear verbatim in the output JSONL as user messages.
                    if uncalled:
                        pending_steer_hint = self._rng.choice([
                            "Okay, what's the next step from here?",
                            "Got it, what do we do next?",
                            "Alright, where do we go from here?",
                        ])
                    else:
                        pending_steer_hint = self._rng.choice([
                            "Great, can you give me a quick summary of everything?",
                            "Perfect, that's all I needed — thanks!",
                            "Awesome, can you wrap it up for me?",
                        ])

                continue  # assistant will respond to tool outputs next turn

            # No tool call — if the assistant returned nothing at all, treat as stuck
            if not assistant_turn.content and not assistant_turn.tool_calls:
                logger.debug(f"[{ctx.conversation_id}] Empty assistant turn, forcing completion")
                return True

            ctx.add_assistant_message(content=assistant_turn.content)

            if assistant_turn.is_final:
                # If significant chain tools are uncalled, let the user naturally extend
                called_ids = set(ctx.tools_used)
                uncalled = [t for t in chain_tools if t.id not in called_ids]
                enough_room = turn < MAX_TURNS_PER_CONVERSATION - 3
                if uncalled and user_extensions < max_user_extensions and enough_room:
                    next_topic = uncalled[0].description[:60] if uncalled else None
                    hint = (
                        f"The task isn't fully done yet. Naturally bring up: {next_topic}"
                        if next_topic else None
                    )
                    user_extension = self.user_agent.respond(plan, ctx, hint=hint)
                    # Stop extending if the user is clearly wrapping up
                    if self._user_is_done(user_extension):
                        ctx.add_user_message(user_extension)
                        logger.debug(f"[{ctx.conversation_id}] User signalled done, ending")
                        return True
                    if user_extension.strip():
                        ctx.add_user_message(user_extension)
                        consecutive_assistant_turns = 0
                        user_extensions += 1
                        continue
                logger.debug(f"[{ctx.conversation_id}] Conversation complete at turn {turn}")
                return True

            if assistant_turn.is_clarification:
                # User responds to the question
                user_response = self.user_agent.respond(plan, ctx)
                # If user says bye in response to a clarification, end cleanly
                if self._user_is_done(user_response):
                    ctx.add_user_message(user_response)
                    return True
                ctx.add_user_message(user_response)
                consecutive_assistant_turns = 0
                continue

            # Safety: if assistant keeps responding without progress, end it
            if consecutive_assistant_turns >= 3:
                logger.debug(f"[{ctx.conversation_id}] Forcing completion after stuck loop")
                return True

        logger.warning(f"[{ctx.conversation_id}] Hit max turns limit")
        return True  # Return True — we have *something*, let the judge decide

    _DONE_PHRASES = (
        "bye", "goodbye", "thanks, bye", "that's all", "all done",
        "that's everything", "no more", "all good", "i'm good", "im good",
    )

    def _user_is_done(self, msg: str) -> bool:
        """Return True if the user message signals the conversation is over."""
        lower = msg.lower().strip()
        return any(phrase in lower for phrase in self._DONE_PHRASES)

    def _maybe_inject_disambiguation(
        self,
        ctx: ConversationContext,
        plan: ConversationPlan,
        chain_tools: list[Tool],
        tool_index: int,
        turn: int,
    ) -> None:
        """
        Check if a disambiguation point should fire before the current tool step.
        The Planner pre-computed when to ask; we inject the question here.
        """
        # This is a coordination signal only — the actual question comes from
        # the assistant agent naturally. We log it for debugging.
        for dp in plan.disambiguation_points:
            if dp.before_tool_index == tool_index and turn > 0:
                logger.debug(
                    f"[{ctx.conversation_id}] Disambiguation point reached: {dp.missing_field}"
                )

    def _targeted_repair(
        self,
        ctx: ConversationContext,
        plan: ConversationPlan,
        chain_tools: list[Tool],
        judge_result: JudgeResult,
    ) -> Optional[ConversationContext]:
        """
        Attempt targeted repair: keep the conversation up to the last successful
        tool call, then regenerate from that point with repair hints injected.

        Returns a repaired ConversationContext or None if repair is not possible.
        """
        # Find the last successful tool output as the cutoff point
        last_good_idx = 0
        for i, msg in enumerate(ctx.messages):
            if msg.role == "tool":
                last_good_idx = i

        if last_good_idx == 0:
            return None  # No tool calls at all, can't do targeted repair

        # Create new context with messages up to last good tool output
        new_session = ExecutionSession(self.mock_generator)
        # Replay state from original session
        new_session.state = dict(ctx.execution_session.state)

        repaired_ctx = ConversationContext(
            conversation_id=ctx.conversation_id + "_repaired",
            execution_session=new_session,
            messages=ctx.messages[:last_good_idx + 1],
            metadata=ctx.metadata.copy(),
        )

        # Inject repair hint into a system-level user message
        hint_text = "Please complete this task correctly. Issues to fix:\n" + \
                    "\n".join(f"- {h}" for h in judge_result.repair_hints)
        repaired_ctx.add_user_message(hint_text)

        # Let assistant complete from here
        for _ in range(5):
            turn = self.assistant_agent.respond(repaired_ctx, chain_tools)
            if turn.tool_calls:
                for tc in turn.tool_calls:
                    tool = self._find_tool(tc.endpoint, chain_tools)
                    if tool:
                        response = repaired_ctx.execution_session.execute(tool, tc.arguments)
                        repaired_ctx.add_tool_output(tool.id, response)
                repaired_ctx.add_assistant_message(turn.content, [
                    ToolCallRef(tc.endpoint, tc.arguments) for tc in turn.tool_calls
                ])
            else:
                repaired_ctx.add_assistant_message(turn.content)
                if turn.is_final:
                    break

        return repaired_ctx

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _create_context(self, conv_id: str) -> ConversationContext:
        session = ExecutionSession(self.mock_generator)
        return ConversationContext(
            conversation_id=conv_id,
            execution_session=session,
            metadata={
                "seed": self.seed,
                "model": self.model_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _find_tool(self, endpoint: str, tools: list[Tool]) -> Optional[Tool]:
        for tool in tools:
            if tool.id == endpoint or tool.endpoint_name == endpoint:
                return tool
        return None
