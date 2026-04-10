"""
Assistant Agent — generates assistant turns, including tool calls.

Uses native OpenAI-compatible function calling (supported by Groq, OpenAI,
and Anthropic's OpenAI-compatible endpoint). This is more reliable than the
text-based TOOL_CALL: prefix approach, especially with smaller models.

See DESIGN.md §5.1 and §8.2.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from toolgen.context.conversation import ConversationContext
from toolgen.registry.models import Tool

logger = logging.getLogger(__name__)

ASSISTANT_SYSTEM = """\
You are a helpful AI assistant. Use the provided tools to help the user accomplish their goal.

RULES:
- Call tools when you need information or to take action — do not make up results
- If a required parameter is missing, ask the user before calling the tool
- After receiving tool results, use the actual values (IDs, names, etc.) in follow-up calls
- When the task is fully complete, give a concise final summary with no further tool calls\
"""


@dataclass
class ToolCallRequest:
    endpoint: str
    arguments: dict[str, Any]


@dataclass
class AssistantTurn:
    content: Optional[str]
    tool_calls: list[ToolCallRequest] = field(default_factory=list)
    is_final: bool = False
    is_clarification: bool = False


class AssistantAgent:
    """
    Generates assistant turns using native function calling.

    Uses the OpenAI tools API (supported by Groq and OpenAI), which is more
    reliable than a text-based tool call format for smaller models.
    """

    def __init__(
        self,
        llm_client,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
    ):
        self._llm = llm_client
        self._model = model
        self._temperature = temperature

    def respond(
        self,
        ctx: ConversationContext,
        available_tools: list[Tool],
        tool_result: Optional[dict] = None,
    ) -> AssistantTurn:
        """Generate an assistant turn using native function calling."""
        # Build OpenAI-format tool schemas
        tools_spec = [self._to_openai_tool(t) for t in available_tools]

        # Build message history
        messages = [{"role": "system", "content": ASSISTANT_SYSTEM}]

        # Add session state hint if there's anything in it
        state_summary = ctx.execution_session.get_state_summary()
        if "No prior" not in state_summary:
            messages[0]["content"] += f"\n\nSession context:\n{state_summary}"

        for msg in ctx.messages:
            if msg.role == "user":
                messages.append({"role": "user", "content": msg.content or ""})
            elif msg.role == "assistant":
                # Reconstruct assistant message with tool_calls if any
                asst_msg: dict[str, Any] = {"role": "assistant"}
                if msg.content:
                    asst_msg["content"] = msg.content
                if msg.tool_calls:
                    asst_msg["tool_calls"] = [
                        {
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {
                                "name": tc.endpoint.replace("/", "__"),
                                "arguments": json.dumps(tc.arguments),
                            },
                        }
                        for i, tc in enumerate(msg.tool_calls)
                    ]
                    if "content" not in asst_msg:
                        asst_msg["content"] = None
                messages.append(asst_msg)
            elif msg.role == "tool" and msg.tool_output is not None:
                # Tool result message
                messages.append({
                    "role": "tool",
                    "tool_call_id": "call_0",
                    "content": json.dumps(msg.tool_output),
                })

        return self._call_with_tools(messages, tools_spec, available_tools)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _call_with_tools(
        self,
        messages: list[dict],
        tools_spec: list[dict],
        available_tools: list[Tool],
    ) -> AssistantTurn:
        try:
            response = self._llm.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=tools_spec,
                tool_choice="auto",
                temperature=self._temperature,
                max_tokens=600,
            )
            time.sleep(0.3)
        except Exception as e:
            # Fallback: call without tools parameter (AnthropicAdapter or unsupported)
            logger.debug(f"Native tool calling failed ({e}), falling back to text mode")
            return self._call_text_fallback(messages, available_tools)

        msg = response.choices[0].message
        narrative = msg.content if msg.content else None

        tool_calls = []
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function
                # Convert double-underscore back to slash for our tool IDs
                endpoint = fn.name.replace("__", "/")
                try:
                    args = json.loads(fn.arguments)
                except json.JSONDecodeError:
                    args = {}
                tool_calls.append(ToolCallRequest(endpoint=endpoint, arguments=args))

        is_clarification = (
            not tool_calls
            and narrative is not None
            and "?" in narrative
            and len(narrative) < 400
        )
        is_final = not tool_calls and not is_clarification

        return AssistantTurn(
            content=narrative,
            tool_calls=tool_calls,
            is_final=is_final,
            is_clarification=is_clarification,
        )

    def _call_text_fallback(
        self, messages: list[dict], available_tools: list[Tool]
    ) -> AssistantTurn:
        """
        Text-based fallback for providers that don't support the tools parameter.
        Injects tool schemas into the system prompt and parses TOOL_CALL: lines.
        """
        tool_desc = "\n".join(
            f"- {t.id}: {t.description} (required: {t.required_params})"
            for t in available_tools
        )
        fallback_system = (
            ASSISTANT_SYSTEM
            + f"\n\nAvailable tools:\n{tool_desc}\n\n"
            "To call a tool, output a line in EXACTLY this format:\n"
            'TOOL_CALL: {"endpoint": "<tool_id>", "arguments": {<args>}}'
        )
        msgs = [{"role": "system", "content": fallback_system}] + messages[1:]

        response = self._llm.chat.completions.create(
            model=self._model,
            messages=msgs,
            temperature=self._temperature,
            max_tokens=600,
        )
        time.sleep(0.3)
        raw = response.choices[0].message.content or ""
        return self._parse_text_response(raw)

    def _parse_text_response(self, raw: str) -> AssistantTurn:
        """Parse TOOL_CALL: lines from free-text response."""
        tool_calls = []
        narrative_lines = []
        for line in raw.split("\n"):
            stripped = line.strip()
            if stripped.upper().startswith("TOOL_CALL:"):
                json_str = stripped[len("TOOL_CALL:"):].strip()
                try:
                    data = json.loads(json_str)
                    tool_calls.append(ToolCallRequest(
                        endpoint=str(data.get("endpoint", "")),
                        arguments=data.get("arguments", {}),
                    ))
                except json.JSONDecodeError:
                    pass
            else:
                narrative_lines.append(line)

        narrative = "\n".join(narrative_lines).strip() or None
        is_clarification = (
            not tool_calls and narrative and "?" in narrative and len(narrative) < 400
        )
        return AssistantTurn(
            content=narrative,
            tool_calls=tool_calls,
            is_final=not tool_calls and not is_clarification,
            is_clarification=bool(is_clarification),
        )

    def _to_openai_tool(self, tool: Tool) -> dict:
        """Convert our Tool model to OpenAI function calling format."""
        schema = tool.to_schema_dict()
        # OpenAI function names must be alphanumeric + underscores
        fn_name = tool.id.replace("/", "__")
        return {
            "type": "function",
            "function": {
                "name": fn_name,
                "description": schema["description"],
                "parameters": schema["parameters"],
            },
        }
