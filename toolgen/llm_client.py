"""
LLM client factory — returns an OpenAI-compatible client.

Supports two backends:
  - openai:   uses the OpenAI Python SDK directly
  - anthropic: wraps the Anthropic SDK in an OpenAI-compatible adapter

The adapter exposes the same interface our agents use:
    client.chat.completions.create(model=..., messages=..., ...)

This lets all agent code remain provider-agnostic.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Literal, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Anthropic → OpenAI-compatible adapter
# ---------------------------------------------------------------------------

class _FakeChoice:
    def __init__(self, text: str):
        self.message = _FakeMessage(text)


class _FakeMessage:
    def __init__(self, text: str):
        self.content = text


class _FakeCompletion:
    def __init__(self, text: str):
        self.choices = [_FakeChoice(text)]


class _FakeCompletions:
    def __init__(self, anthropic_client, default_model: str):
        self._client = anthropic_client
        self._default_model = default_model

    def create(
        self,
        model: str,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        response_format: Optional[dict] = None,
        **kwargs,
    ) -> _FakeCompletion:
        """
        OpenAI-compatible create() that delegates to Anthropic messages.create().

        JSON mode: Anthropic doesn't have response_format, so we inject a
        strong instruction and extract the JSON block from the response.
        """
        wants_json = (
            response_format is not None
            and response_format.get("type") == "json_object"
        )

        # Split system message from conversation messages
        system_content = ""
        conv_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content += msg["content"] + "\n"
            else:
                conv_messages.append({"role": msg["role"], "content": msg["content"]})

        # Anthropic requires at least one human message
        if not conv_messages:
            conv_messages = [{"role": "user", "content": "Begin."}]

        if wants_json:
            system_content += (
                "\nIMPORTANT: Your response must be valid JSON only. "
                "No prose, no markdown fences, no explanation outside the JSON object."
            )

        # Map model names: if caller passes an OpenAI model name, use our default
        anthropic_model = self._resolve_model(model)

        response = self._client.messages.create(
            model=anthropic_model,
            max_tokens=max(max_tokens, 1024),
            system=system_content.strip() or "You are a helpful assistant.",
            messages=conv_messages,
            temperature=min(temperature, 1.0),  # Anthropic max is 1.0
        )

        raw_text = response.content[0].text.strip()

        if wants_json:
            raw_text = self._extract_json(raw_text)

        return _FakeCompletion(raw_text)

    def _resolve_model(self, model: str) -> str:
        """Map OpenAI model names to Anthropic equivalents."""
        mapping = {
            "gpt-4o-mini": "claude-haiku-4-5-20251001",
            "gpt-4o": "claude-sonnet-4-6",
            "gpt-4": "claude-sonnet-4-6",
            "gpt-3.5-turbo": "claude-haiku-4-5-20251001",
        }
        # If it's already an Anthropic model name, use as-is
        if model.startswith("claude-"):
            return model
        resolved = mapping.get(model, "claude-haiku-4-5-20251001")
        if resolved != model:
            logger.debug(f"Model mapping: {model} → {resolved}")
        return resolved

    def _extract_json(self, text: str) -> str:
        """Extract JSON from a response that may have surrounding prose."""
        # Already clean JSON
        stripped = text.strip()
        if stripped.startswith("{") or stripped.startswith("["):
            return stripped

        # Try to extract from markdown code fence
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1).strip()

        # Try to find the first { ... } block
        brace_match = re.search(r"\{.*\}", text, re.DOTALL)
        if brace_match:
            return brace_match.group(0).strip()

        # Return as-is and let the caller fail with a clear parse error
        logger.warning(f"Could not extract JSON from response: {text[:200]!r}")
        return text


class _FakeChat:
    def __init__(self, anthropic_client, default_model: str):
        self.completions = _FakeCompletions(anthropic_client, default_model)


class AnthropicAdapter:
    """
    Wraps the Anthropic client with an OpenAI-compatible interface.

    Usage (same as openai.OpenAI()):
        client = AnthropicAdapter()
        response = client.chat.completions.create(
            model="gpt-4o-mini",   # mapped to claude-haiku automatically
            messages=[...],
        )
        text = response.choices[0].message.content
    """

    def __init__(self, api_key: Optional[str] = None, default_model: str = "claude-haiku-4-5-20251001"):
        import anthropic
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )
        self.chat = _FakeChat(self._client, default_model)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_llm_client(
    provider: Literal["openai", "anthropic"] = "anthropic",
    model: Optional[str] = None,
) -> Any:
    """
    Return an LLM client. Defaults to Anthropic if ANTHROPIC_API_KEY is set,
    OpenAI otherwise.

    The returned client has an OpenAI-compatible interface regardless of provider.
    """
    if provider == "anthropic" or (
        provider is None and os.environ.get("ANTHROPIC_API_KEY")
    ):
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        logger.info("Using Anthropic backend (claude-haiku-4-5-20251001)")
        return AnthropicAdapter(api_key=api_key)

    # OpenAI
    import openai
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set")
    logger.info("Using OpenAI backend")
    return openai.OpenAI(api_key=api_key)
