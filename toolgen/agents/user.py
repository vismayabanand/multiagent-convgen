"""
User Simulator Agent — generates user turns in the conversation.

Key design: the user agent does NOT see the tool chain or disambiguation plan.
It only knows its goal and persona. This prevents robotic, plan-following
responses and forces natural information withholding.
"""

from __future__ import annotations

import logging
from typing import Optional

from toolgen.agents.planner import ConversationPlan
from toolgen.context.conversation import ConversationContext

logger = logging.getLogger(__name__)

_PERSONA_INSTRUCTIONS = {
    "terse_professional": (
        "You are terse and direct. Provide only the information asked. "
        "Use short sentences. Do not volunteer extra context."
    ),
    "casual_friendly": (
        "You are conversational and friendly. You may digress slightly. "
        "You naturally forget to provide some details upfront."
    ),
    "technical_expert": (
        "You are technically precise. You use domain-appropriate terminology. "
        "You provide structured, complete information when asked."
    ),
    "confused_novice": (
        "You are unfamiliar with the process. You may ask what terms mean. "
        "You provide vague answers and sometimes misunderstand questions."
    ),
}

USER_SYSTEM_TEMPLATE = """\
You are a user interacting with an AI assistant.

Your goal: {user_goal}

Personality: {persona_instruction}

Important:
- You do not know what tools the assistant has available
- You naturally omit some details in your first message (the assistant may need to ask)
- When the assistant asks for specific information, provide it naturally
- When the task is done, thank the assistant briefly and end the conversation
- Keep your messages concise (1-4 sentences)\
"""


class UserSimulatorAgent:
    """
    Simulates the user side of the conversation.

    The user agent is intentionally kept simple — it follows the goal and
    persona but has no knowledge of the underlying tool chain. This produces
    more realistic information withholding.
    """

    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.9,  # higher temp for varied user utterances
    ):
        self._llm = llm_client
        self._model = model
        self._temperature = temperature

    def opening_message(self, plan: ConversationPlan) -> str:
        """Generate the first user message that starts the conversation."""
        system = USER_SYSTEM_TEMPLATE.format(
            user_goal=plan.user_goal,
            persona_instruction=_PERSONA_INSTRUCTIONS.get(plan.persona, ""),
        )
        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    "Write your opening message to the assistant. "
                    "State your goal naturally but do not reveal all details upfront."
                ),
            },
        ]
        return self._call_llm(messages)

    def respond(self, plan: ConversationPlan, ctx: ConversationContext) -> str:
        """Generate a user response given the current conversation context."""
        system = USER_SYSTEM_TEMPLATE.format(
            user_goal=plan.user_goal,
            persona_instruction=_PERSONA_INSTRUCTIONS.get(plan.persona, ""),
        )

        # Build conversation history for the user agent
        # (only user and assistant messages — not tool calls/outputs)
        history = []
        for msg in ctx.messages:
            if msg.role == "user":
                history.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant" and msg.content:
                history.append({"role": "assistant", "content": msg.content})

        messages = [{"role": "system", "content": system}] + history
        return self._call_llm(messages)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict]) -> str:
        response = self._llm.chat.completions.create(
            model=self._model,
            messages=messages,
            temperature=self._temperature,
            max_tokens=200,
        )
        return response.choices[0].message.content.strip()
