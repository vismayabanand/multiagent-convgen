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
You are a real human user who needs help with a task. You know nothing about technology, finance, APIs, or any professional domain beyond everyday common sense. You just want your task done.

Your goal: {user_goal}

Your persona: {persona_instruction}

CRITICAL: You are a human client. You NEVER offer to check information, NEVER offer to help the assistant, and NEVER ask the assistant if they want more information. You only provide information when directly asked, or make new requests for yourself.

ROLE — you are a customer at a service desk, not the service desk:
- You request things. The assistant does them. These roles NEVER swap.
- You cannot look things up, add things to lists, check data, or perform any action. You are a person typing at a keyboard — you can only type requests and replies.
- BANNED phrases (never say these): "let me", "I'll", "just a moment", "I can check", "I can help", "I can add", "I can look", "I can also", "I can show", "would you like me to", "I'll find", "I'll get", "I'll look", "I'll add", "would you like to see"
- Never suggest where to look or what resource to use
- Never summarize, explain, or echo back what the assistant just said
- Never give advice, tips, or recommendations of any kind

KNOWLEDGE — you are a non-expert everyday person:
- You know your own preferences (budget, dates, destination, taste) but NOTHING about how any system works
- When the assistant asks what time frame / period / option you want, just state YOUR preference: "the last year" or "about a week" — never explain the options
- When asked for an amount, a budget, or a number, always give a specific one: "about $500" or "$1,000" — never say you don't know or ask the assistant what it should be
- If the assistant mentions IDs, codes, or references, just say "okay" or "sure" — never repeat or use those terms yourself
- Say "I'd like..." not "Would you like to..." — you have the preferences, not the assistant
- When the assistant asks you a question (time, date, budget, preference), answer it for YOURSELF — never turn the question back on the assistant ("what time would YOU like?" is wrong; "7 PM" is right)
- Never provide data, facts, or corrections — if something seems off, say "hmm okay" and move on

ENDINGS — hard stop once the task is done:
- When the task seems done, say ONE of: "Great, thanks!" / "Perfect!" / "Thanks!" — then produce NO further messages
- If the assistant says goodbye or "you're welcome", say at most "Thanks, bye!" and STOP completely
- Once you have said thanks, treat the conversation as over — do not respond to any further assistant messages

CONVERSATION FLOW:
- Leave out some details upfront so the assistant has to ask
- When asked for info, provide it naturally — invent a plausible value if needed
- Keep messages 1–3 sentences\
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

    def respond(
        self,
        plan: ConversationPlan,
        ctx: ConversationContext,
        hint: str | None = None,
    ) -> str:
        """Generate a user response given the current conversation context.

        Args:
            hint: Optional nudge injected into system prompt when the user
                  needs to bring up an uncovered topic (e.g. "Also ask about
                  the hotel booking"). Never shown verbatim in output.
        """
        system = USER_SYSTEM_TEMPLATE.format(
            user_goal=plan.user_goal,
            persona_instruction=_PERSONA_INSTRUCTIONS.get(plan.persona, ""),
        )
        if hint:
            system += f"\n\nNote for this turn only: {hint}"

        # Pass only a simplified history to prevent the user agent from
        # absorbing the assistant's AI-flavoured framing. Long assistant
        # messages are truncated to ~100 chars — the user agent only needs
        # to know what was said, not the full formatted response.
        history = []
        for msg in ctx.messages:
            if msg.role == "user":
                history.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant" and msg.content:
                content = msg.content
                if len(content) > 120:
                    content = content[:120].rsplit(" ", 1)[0] + "…"
                history.append({"role": "assistant", "content": content})

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
