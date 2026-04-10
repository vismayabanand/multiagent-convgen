"""
ConversationContext — shared mutable state within a single conversation.

Passed between agents as the single source of truth for conversation history,
execution session state, and metadata. See DESIGN.md §5.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from toolgen.executor.session import ExecutionSession


@dataclass
class ToolCallRef:
    """Compact representation of a tool call within a message."""
    endpoint: str
    arguments: dict[str, Any]


@dataclass
class Message:
    """A single message in the conversation."""
    role: str           # "user" | "assistant" | "tool"
    content: Optional[str] = None
    tool_calls: Optional[list[ToolCallRef]] = None   # assistant messages with tool calls
    tool_output: Optional[dict] = None               # tool role messages
    tool_id: Optional[str] = None                    # which tool produced this output


@dataclass
class ConversationContext:
    """
    All state for a single in-progress conversation.

    Agents read from and write to this object. The orchestrator owns it.
    """
    conversation_id: str
    execution_session: ExecutionSession
    messages: list[Message] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Populated during orchestration
    tools_used: list[str] = field(default_factory=list)
    repair_attempts: int = 0

    def add_user_message(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content))

    def add_assistant_message(
        self,
        content: Optional[str],
        tool_calls: Optional[list[ToolCallRef]] = None,
    ) -> None:
        self.messages.append(Message(
            role="assistant",
            content=content,
            tool_calls=tool_calls,
        ))

    def add_tool_output(self, tool_id: str, output: dict) -> None:
        self.messages.append(Message(
            role="tool",
            tool_id=tool_id,
            tool_output=output,
        ))
        if tool_id not in self.tools_used:
            self.tools_used.append(tool_id)

    @property
    def num_turns(self) -> int:
        return len(self.messages)

    @property
    def num_tool_calls(self) -> int:
        count = 0
        for msg in self.messages:
            if msg.tool_calls:
                count += len(msg.tool_calls)
        return count

    @property
    def last_assistant_message(self) -> Optional[Message]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg
        return None

    def to_messages_list(self) -> list[dict]:
        """Serialize messages to the output JSONL format."""
        out = []
        for msg in self.messages:
            record: dict[str, Any] = {"role": msg.role}
            if msg.content is not None:
                record["content"] = msg.content
            if msg.tool_calls:
                record["tool_calls"] = [
                    {"endpoint": tc.endpoint, "arguments": tc.arguments}
                    for tc in msg.tool_calls
                ]
            if msg.tool_output is not None:
                record["content"] = msg.tool_output
                record["tool_id"] = msg.tool_id
            out.append(record)
        return out
