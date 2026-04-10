"""
Mock response generator for offline tool execution.

Two strategies (see DESIGN.md §4.3):
  1. Schema-derived: uses faker to generate type-conformant values (fast, deterministic)
  2. LLM-generated fallback: used when no response schema exists

The generator is deterministic when a seed is set — important for Run A/B reproducibility.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from typing import Any, Optional

from toolgen.registry.models import ResponseField, Tool

logger = logging.getLogger(__name__)

# Field name patterns that should produce realistic-looking ID values
_ID_SUFFIXES = ("_id", "_key", "_token", "_ref", "_code", "_number", "_uuid")
# Field name patterns → faker method hints
_FIELD_HINTS: dict[str, str] = {
    "name": "name",
    "city": "city",
    "country": "country",
    "address": "address",
    "email": "email",
    "phone": "phone_number",
    "date": "date",
    "url": "url",
    "description": "sentence",
    "status": "word",
    "message": "sentence",
    "title": "sentence",
    "currency": "currency_code",
    "author": "name",
    "venue": "company",
    "airline": "company",
    "source": "company",
}

# Fields with capped realistic numeric ranges
_NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "price": (10.0, 500.0),
    "price_per_night": (50.0, 400.0),
    "total_price": (50.0, 2000.0),
    "rating": (1.0, 5.0),
    "change_percent": (-10.0, 10.0),
    "change": (-50.0, 50.0),
    "humidity": (10.0, 99.0),
    "temperature": (-20.0, 45.0),
    "feels_like": (-25.0, 50.0),
    "wind_speed": (0.0, 80.0),
    "distance_km": (0.1, 50.0),
    "distance_meters": (50.0, 5000.0),
    "duration_minutes": (5.0, 480.0),
    "reading_time": (1.0, 30.0),
    "word_count": (100.0, 2000.0),
    "attendance": (1000.0, 80000.0),
    "volume": (100000.0, 50000000.0),
    "market_cap": (1e9, 3e12),
    "exchange_rate": (0.5, 5.0),
    "precipitation_chance": (0.0, 100.0),
    "available_rooms": (1.0, 50.0),
    "available_spots": (1.0, 200.0),
    "party_size": (1.0, 12.0),
    "table_number": (1.0, 50.0),
    "num_guests": (1.0, 10.0),
    "stops": (0.0, 3.0),
    "shipping_days": (1.0, 14.0),
    "review_count": (1.0, 5000.0),
    "cart_total": (10.0, 1000.0),
    "average_price": (5.0, 500.0),
}


class MockGenerator:
    """
    Generates mock tool responses consistent with endpoint schemas.

    Preference order:
      1. Schema-derived generation (if response_fields is non-empty)
      2. LLM-generated (if llm_client is provided and schema is absent)
      3. Generic fallback {"status": "success", "result": null}

    Usage:
        gen = MockGenerator(seed=42)
        response = gen.generate(tool, resolved_args)
    """

    def __init__(
        self,
        llm_client=None,          # optional: openai.OpenAI instance
        llm_model: str = "gpt-4o-mini",
        seed: Optional[int] = None,
    ):
        self._llm = llm_client
        self._llm_model = llm_model
        self._rng = random.Random(seed)

        try:
            from faker import Faker
            self._faker = Faker()
            self._faker.seed_instance(seed or 0)
        except ImportError:
            self._faker = None
            logger.warning("faker not installed; falling back to basic mock generation")

    def generate(self, tool: Tool, resolved_args: dict[str, Any]) -> dict[str, Any]:
        """Generate a mock response for the given tool call."""
        if tool.response_fields:
            return self._schema_derived(tool, resolved_args)
        if self._llm is not None:
            result = self._llm_generated(tool, resolved_args)
            if result is not None:
                return result
        return self._generic_fallback(tool)

    # ------------------------------------------------------------------
    # Schema-derived generation
    # ------------------------------------------------------------------

    def _schema_derived(self, tool: Tool, args: dict) -> dict:
        """Walk response_fields and generate faker values."""
        result = {}
        for field in tool.response_fields:
            result[field.name] = self._generate_field_value(field, args, tool)
        return result

    def _generate_field_value(
        self, field: ResponseField, args: dict, tool: Tool
    ) -> Any:
        if field.is_array:
            count = self._rng.randint(1, 4)
            return [self._scalar_value(field, args, tool) for _ in range(count)]
        return self._scalar_value(field, args, tool)

    def _scalar_value(self, field: ResponseField, args: dict, tool: Tool) -> Any:
        name = field.name
        name_lower = name.lower()

        # If it's an ID field, generate a stable domain-prefixed ID
        if field.is_id_field():
            prefix = tool.category.lower()[:3]
            return f"{prefix}_{self._short_hash(name + tool.id + str(self._rng.random()))}"

        # Check capped numeric ranges first (prevents absurd values like price=8e11)
        if field.type == "number":
            for range_key, (lo, hi) in _NUMERIC_RANGES.items():
                if range_key in name_lower:
                    return round(self._rng.uniform(lo, hi), 2)

        # Check field name hints
        for hint_key, faker_method in _FIELD_HINTS.items():
            if hint_key in name_lower:
                return self._call_faker(faker_method)

        # Domain-specific overrides
        if "symbol" in name_lower:
            # Stock ticker — 2-4 uppercase letters
            return self._rng.choice(["AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"])
        if "currency" in name_lower or "code" in name_lower:
            return self._rng.choice(["USD", "EUR", "GBP", "JPY", "CAD"])
        if "status" in name_lower:
            return self._rng.choice(["success", "confirmed", "pending", "active"])
        if "country" in name_lower:
            return self._rng.choice(["US", "UK", "FR", "DE", "JP", "CA"])

        # Fall back on type
        return self._type_value(field.type)

    def _type_value(self, type_: str) -> Any:
        if self._faker:
            if type_ == "string":
                return self._faker.word()
            if type_ == "number":
                return round(self._rng.uniform(1, 1000), 2)
            if type_ == "boolean":
                return self._rng.choice([True, False])
            if type_ == "array":
                return [self._faker.word() for _ in range(self._rng.randint(1, 3))]
            if type_ == "object":
                return {"key": self._faker.word(), "value": self._faker.word()}
        # Minimal fallback without faker
        defaults = {
            "string": "value",
            "number": 42,
            "boolean": True,
            "array": [],
            "object": {},
            "unknown": None,
        }
        return defaults.get(type_)

    def _call_faker(self, method: str) -> Any:
        if self._faker is None:
            return "value"
        try:
            fn = getattr(self._faker, method, None)
            return fn() if fn else self._faker.word()
        except Exception:
            return self._faker.word()

    # ------------------------------------------------------------------
    # LLM-generated fallback
    # ------------------------------------------------------------------

    def _llm_generated(self, tool: Tool, args: dict) -> Optional[dict]:
        """Ask the LLM to generate a plausible JSON response."""
        prompt = (
            f"You are simulating the response of an API endpoint.\n\n"
            f"Tool: {tool.id}\n"
            f"Description: {tool.description}\n"
            f"Input arguments: {json.dumps(args, indent=2)}\n\n"
            f"Generate a realistic JSON response for this API call. "
            f"Return only valid JSON, no prose."
        )
        try:
            response = self._llm.chat.completions.create(
                model=self._llm_model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content
            return json.loads(content)
        except Exception as e:
            logger.warning(f"LLM mock generation failed for {tool.id}: {e}")
            return None

    # ------------------------------------------------------------------
    # Generic fallback
    # ------------------------------------------------------------------

    def _generic_fallback(self, tool: Tool) -> dict:
        return {
            "status": "success",
            "tool": tool.id,
            "result": None,
            "message": f"Mock response for {tool.endpoint_name}",
        }

    @staticmethod
    def _short_hash(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:6]
