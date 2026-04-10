"""
Execution Session — maintains state across tool calls in a single conversation.

The two-layer grounding mechanism (see DESIGN.md §7.1):
  1. _resolve_args: substitutes arguments with values from session state
     (deterministic override that prevents hallucinated IDs)
  2. _extract_refs: pulls ID-like fields from responses into state
     for use in subsequent calls
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from toolgen.registry.models import Tool
from .mock_generator import MockGenerator

logger = logging.getLogger(__name__)

_REF_SUFFIXES = ("_id", "_key", "_token", "_ref", "_code", "_number", "_uuid")


@dataclass
class ToolCallRecord:
    """A single executed tool call and its response."""
    tool_id: str
    resolved_args: dict[str, Any]
    response: dict[str, Any]
    raw_args: dict[str, Any]   # args as provided by the LLM (before resolution)


class ExecutionSession:
    """
    Maintains state across tool calls within a single conversation.

    The session state maps field names to their most recently returned values.
    When a tool call's arguments contain a name present in state, the state
    value is used instead — preventing hallucinated IDs.

    Usage:
        session = ExecutionSession(mock_generator)
        response = session.execute(tool, llm_args)
        # response contains real mock values, and state is updated
    """

    def __init__(self, mock_generator: MockGenerator):
        self.mock_generator = mock_generator
        self.state: dict[str, Any] = {}    # field_name → most recent value
        self.history: list[ToolCallRecord] = []

    def execute(self, tool: Tool, llm_args: dict[str, Any]) -> dict[str, Any]:
        """
        Execute a tool call:
          1. Resolve args against session state
          2. Generate mock response
          3. Extract new refs into state
          4. Return response
        """
        resolved = self._resolve_args(tool, llm_args)
        response = self.mock_generator.generate(tool, resolved)
        self._extract_refs(response)

        record = ToolCallRecord(
            tool_id=tool.id,
            resolved_args=resolved,
            response=response,
            raw_args=llm_args,
        )
        self.history.append(record)

        logger.debug(
            f"Executed {tool.id}: "
            f"state_keys={list(self.state.keys())}, "
            f"response_keys={list(response.keys())}"
        )
        return response

    def get_state_summary(self) -> str:
        """Human-readable summary of current session state, injected into prompts."""
        if not self.state:
            return "No prior tool outputs yet."
        lines = ["Prior tool outputs available for reference:"]
        for k, v in self.state.items():
            lines.append(f"  {k} = {v!r}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _resolve_args(self, tool: Tool, args: dict[str, Any]) -> dict[str, Any]:
        """
        Replace any argument whose name is in session state with the actual value.
        This is the deterministic safety net against hallucinated IDs.
        """
        resolved = {}
        for param in tool.parameters:
            name = param.name
            if name not in args:
                # Use default if available
                if param.default is not None:
                    resolved[name] = param.default
                continue

            llm_value = args[name]

            # Override with state value if the field is an ID type and we have it
            if param.is_id_field() and name in self.state:
                state_value = self.state[name]
                if state_value != llm_value:
                    logger.debug(
                        f"Grounding: {name}={llm_value!r} → {state_value!r} (from session state)"
                    )
                resolved[name] = state_value
            else:
                resolved[name] = llm_value

        # Pass through any extra args the LLM provided (not in schema)
        for k, v in args.items():
            if k not in resolved:
                resolved[k] = v

        return resolved

    def _extract_refs(self, response: dict[str, Any]) -> None:
        """
        Walk the response and extract ID-like fields into session state.
        Handles nested dicts and first-element of arrays.
        """
        self._walk_and_extract(response)

    def _walk_and_extract(self, obj: Any, depth: int = 0) -> None:
        """Recursively walk a response object to extract refs."""
        if depth > 3:  # don't recurse too deep
            return

        if isinstance(obj, dict):
            for k, v in obj.items():
                if any(k.endswith(s) for s in _REF_SUFFIXES):
                    if isinstance(v, (str, int, float)):
                        self.state[k] = v
                if isinstance(v, (dict, list)):
                    self._walk_and_extract(v, depth + 1)

        elif isinstance(obj, list) and obj:
            # Extract from first element only (representative)
            self._walk_and_extract(obj[0], depth + 1)
