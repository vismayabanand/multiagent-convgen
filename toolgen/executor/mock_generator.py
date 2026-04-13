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
# "name" is intentionally absent — handled contextually in _scalar_value
_FIELD_HINTS: dict[str, str] = {
    "city": "city",
    "country": "country",
    "address": "address",
    "email": "email",
    "phone": "phone_number",
    "url": "url",
    "description": "sentence",
    "message": "sentence",
    "title": "sentence",
    "currency": "currency_code",
    "author": "name",
    "venue": "company",
    "airline": "company",
    "source": "company",
    "note": "sentence",
}

# Field name fragments that should always use a person's name regardless of category
_PERSON_NAME_FRAGMENTS = ("reviewer", "author", "customer", "guest", "passenger", "user_name", "buyer")

# Field name fragments for integer counts (never floats)
_INTEGER_COUNT_FRAGMENTS = ("rooms", "spots", "guests", "stops", "seats", "items", "quantity", "count", "nights", "volume")

# Realistic review comment templates — far better than faker.sentence() gibberish
# Realistic weather descriptions for mock responses
_WEATHER_DESCRIPTIONS = [
    "Partly cloudy with light breeze",
    "Clear skies and sunny",
    "Overcast with chance of rain",
    "Light rain showers expected",
    "Mostly sunny with some clouds",
    "Foggy in the morning, clearing by afternoon",
    "Heavy rain and thunderstorms possible",
    "Cool and breezy",
    "Warm and humid",
    "Windy with scattered clouds",
    "Mild and pleasant",
    "Cold with possible frost overnight",
]

# Realistic price range values
_PRICE_RANGES = ["$", "$$", "$$$", "$$$$"]

_REVIEW_COMMENTS = [
    "Great experience overall, would definitely come back.",
    "Exactly what I needed. Fast and straightforward.",
    "Good value for the price. Staff was helpful.",
    "Smooth process from start to finish. No complaints.",
    "Decent but nothing exceptional. Gets the job done.",
    "Highly recommend — exceeded my expectations.",
    "Average experience. Room was clean but location was tricky.",
    "Very convenient. Booking was easy and check-in was quick.",
    "Not bad, but I've had better. Would try again with lower expectations.",
    "Fantastic stay. Will definitely book again next time I'm in town.",
]

# Tool categories that represent businesses/places — "name" should be company, not person
_BUSINESS_CATEGORIES = {
    "Food", "Travel", "Hotel", "Hotels", "Restaurant", "Restaurants",
    "Shopping", "Entertainment", "Sports", "eCommerce", "Business",
    "Finance", "Transportation",
}

# Realistic hotel/property names — faker.company() produces law firms and tech startups
_HOTEL_NAMES = [
    "The Grand Meridian", "Parkview Suites", "Harbor Inn & Suites",
    "The Riverside Hotel", "Skyline Boutique Hotel", "Central Plaza Hotel",
    "The Summit Lodge", "Lakeview Inn", "The Metropolitan Hotel",
    "Oceanfront Suites", "The Heritage Inn", "Landmark Hotel & Spa",
    "The Westgate Hotel", "City Center Suites", "The Belmont",
    "Rosewood Residences", "The Cliffside Inn", "Midtown Suites",
    "The Windsor Hotel", "Bayshore Resort",
]

# Realistic restaurant names
_RESTAURANT_NAMES = [
    "The Golden Fork", "Bistro 42", "Casa Mia", "The Rustic Table",
    "Harbor Grill", "Sakura Garden", "The Spice Route", "Olive & Vine",
    "The Corner Bistro", "Blue Plate Kitchen", "Ciao Bella",
    "The Oak Room", "Coastal Kitchen", "La Petite Maison", "The Copper Pot",
]

# Realistic product/device names for electronics, laptops, etc.
_PRODUCT_NAMES = [
    "ProBook 450", "UltraSlim X1", "NovaPad Pro", "Zenith 15",
    "Swift 3", "IdeaPad 5", "ToughBook G2", "VivoBook 14",
    "Precision 5570", "MateBook D15", "Surface Laptop 4", "Inspiron 15",
]

# Realistic calendar event titles
_EVENT_TITLES = [
    "Team standup", "Project review meeting", "Lunch with client",
    "Doctor appointment", "Quarterly planning session", "Coffee with Alex",
    "Flight to New York", "Interview — Senior Engineer", "Dentist checkup",
    "Budget review", "Product demo", "Birthday dinner", "Gym session",
    "Call with vendor", "Annual performance review",
]

# Realistic short descriptions / notes (for event descriptions, notes fields)
_SHORT_DESCRIPTIONS = [
    "Discuss Q3 roadmap and upcoming milestones.",
    "Review project status and next steps.",
    "Follow up on pending items from last week.",
    "Confirm details before the trip.",
    "Bring all relevant documents.",
    "Check availability and confirm time.",
    "Important — do not reschedule.",
    "Prep slides before the meeting.",
    "Book conference room in advance.",
    "Coordinate with the team beforehand.",
]

# Fixed status vocabularies by domain keyword
_STATUS_VALUES = ["confirmed", "pending", "active", "success", "completed", "scheduled"]

# Timestamp-like field name fragments → use a recent datetime string
_TIMESTAMP_FRAGMENTS = ("_at", "_time", "created", "updated", "timestamp", "booked", "added")

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
    "exchange_rate": (0.5, 1.8),   # realistic FX: EUR/USD ~0.9, GBP/USD ~0.8
    "converted_amount": (10.0, 5000.0),  # dollar-like amounts after conversion
    "precipitation_chance": (0.0, 100.0),
    "available_rooms": (1.0, 50.0),
    "available_spots": (1.0, 200.0),
    "party_size": (1.0, 12.0),
    "table_number": (1.0, 50.0),
    "num_guests": (1.0, 10.0),
    "stops": (0.0, 3.0),   # converted to int in _scalar_value
    "parking_rate": (5.0, 40.0),
    "hourly_rate": (5.0, 50.0),
    "daily_rate": (20.0, 150.0),
    "shipping_days": (1.0, 14.0),
    "review_count": (1.0, 5000.0),
    "cart_total": (10.0, 1000.0),
    "average_price": (5.0, 500.0),
    # Generic "rate" fallback — must come LAST so specific entries above win first
    "rate": (0.5, 2.0),
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
        return self._apply_coherence_rules(result, args)

    def _apply_coherence_rules(self, result: dict, args: dict) -> dict:
        """
        Fix impossible value combinations that arise from independent field generation.

        Handles:
        - Stock OHLC: high >= max(open, close) >= min(open, close) >= low
        - Stock price: current price consistent with OHLC range (symbol-seeded)
        - Price/total consistency: total_price = price × actual_nights from args
        """
        lower_keys = {k.lower(): k for k in result}

        # --- Symbol-consistent stock base price ---
        # Use the symbol from args to seed a stable price range so that
        # get_stock_quote and get_stock_history for the same symbol
        # return prices in the same ballpark across separate mock calls.
        symbol = args.get("symbol", "")
        stock_base = self._symbol_base_price(symbol) if symbol else None

        # --- Stock OHLC coherence ---
        ohlc_map = {}
        for canonical in ("open", "high", "low", "close"):
            if canonical in lower_keys:
                ohlc_map[canonical] = lower_keys[canonical]

        if len(ohlc_map) >= 3:
            base = stock_base or self._rng.uniform(10.0, 1000.0)
            daily_range = base * self._rng.uniform(0.005, 0.03)
            open_p = round(base + self._rng.uniform(-daily_range, daily_range), 2)
            close_p = round(base + self._rng.uniform(-daily_range, daily_range), 2)
            high_p = round(max(open_p, close_p) + self._rng.uniform(0.01, daily_range), 2)
            low_p = round(min(open_p, close_p) - self._rng.uniform(0.01, daily_range), 2)
            coherent = {"open": open_p, "close": close_p, "high": high_p, "low": low_p}
            for canonical, original_key in ohlc_map.items():
                result[original_key] = coherent[canonical]

        # --- Current stock price consistent with base ---
        price_key = lower_keys.get("price")
        if price_key and stock_base and "open" not in lower_keys:
            # This is a quote response (no OHLC) — pin price near base
            daily_move = stock_base * self._rng.uniform(-0.03, 0.03)
            result[price_key] = round(stock_base + daily_move, 2)

        # --- price / total_price coherence using actual booking duration ---
        price_key = lower_keys.get("price_per_night") or lower_keys.get("price")
        total_key = lower_keys.get("total_price")
        if price_key and total_key:
            price_val = result[price_key]
            nights = self._booking_nights(args)
            result[total_key] = round(price_val * nights, 2)
        elif total_key and not price_key:
            # Booking confirmation that has total_price but no per-night price in response:
            # derive a coherent total from nights × a generated nightly rate.
            nights = self._booking_nights(args)
            if nights > 1:
                per_night = round(self._rng.uniform(50.0, 400.0), 2)
                result[lower_keys["total_price"]] = round(per_night * nights, 2)

        # --- Weather temperature coherence: high_temp > low_temp, feels_like near temp ---
        high_key = lower_keys.get("high_temp")
        low_key = lower_keys.get("low_temp")
        feels_key = lower_keys.get("feels_like")
        temp_key = lower_keys.get("temperature")

        if high_key and low_key:
            # Ensure high > low with a realistic spread (3–15 degrees)
            base_temp = self._rng.uniform(0.0, 35.0)
            spread = self._rng.uniform(3.0, 15.0)
            result[high_key] = round(base_temp + spread / 2, 1)
            result[low_key] = round(base_temp - spread / 2, 1)

        if temp_key and feels_key:
            temp_val = result[temp_key]
            # feels_like within ±8°C of actual temperature
            result[feels_key] = round(temp_val + self._rng.uniform(-8.0, 8.0), 1)

        # --- Currency conversion coherence: converted_amount = input_amount × exchange_rate ---
        converted_key = lower_keys.get("converted_amount")
        exchange_key = lower_keys.get("exchange_rate")
        if converted_key and exchange_key:
            try:
                amount = float(args.get("amount", 100))
                rate = float(result[lower_keys["exchange_rate"]])
                result[lower_keys["converted_amount"]] = round(amount * rate, 2)
            except (TypeError, ValueError):
                pass

        # --- Weather description coherent with precipitation_chance ---
        desc_key = lower_keys.get("description")
        precip_key = lower_keys.get("precipitation_chance")
        if desc_key and precip_key:
            precip = result[precip_key]
            if precip >= 70:
                result[desc_key] = self._rng.choice([
                    "Heavy rain and thunderstorms possible",
                    "Light rain showers expected",
                    "Overcast with chance of rain",
                ])
            elif precip >= 30:
                result[desc_key] = self._rng.choice([
                    "Partly cloudy with light breeze",
                    "Overcast with chance of rain",
                    "Foggy in the morning, clearing by afternoon",
                    "Windy with scattered clouds",
                ])
            else:
                result[desc_key] = self._rng.choice([
                    "Clear skies and sunny",
                    "Mostly sunny with some clouds",
                    "Mild and pleasant",
                    "Cool and breezy",
                ])

        return result

    def _symbol_base_price(self, symbol: str) -> float:
        """Derive a stable base price for a stock symbol using its hash.

        Same symbol always maps to same price range — ensures quote and
        history calls for TSLA return prices in the same ballpark.
        """
        h = int(hashlib.md5(symbol.upper().encode()).hexdigest()[:8], 16)
        return round(20.0 + (h % 98000) / 100.0, 2)

    def _booking_nights(self, args: dict) -> int:
        """Calculate nights from check_in_date / check_out_date in args, fallback to random."""
        import datetime as dt
        try:
            ci = dt.date.fromisoformat(str(args.get("check_in_date", "")))
            co = dt.date.fromisoformat(str(args.get("check_out_date", "")))
            nights = (co - ci).days
            if 1 <= nights <= 30:
                return nights
        except (ValueError, TypeError):
            pass
        return self._rng.randint(1, 7)

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

        # Numeric range check comes FIRST for number fields — this prevents
        # fields like "table_number" (which ends in "_number") from being
        # mistakenly treated as ID fields by is_id_field().
        if field.type == "number":
            for range_key, (lo, hi) in _NUMERIC_RANGES.items():
                if self._range_key_matches(range_key, name_lower):
                    val = self._rng.uniform(lo, hi)
                    if range_key in ("stops", "volume") or "number" in name_lower or any(
                        f in name_lower for f in _INTEGER_COUNT_FRAGMENTS
                    ):
                        return int(round(val))
                    return round(val, 2)
            # Generic number fallback
            return round(self._rng.uniform(1.0, 100.0), 2)

        # If it's an ID field, generate a stable domain-prefixed ID
        if field.is_id_field():
            prefix = tool.category.lower()[:3]
            return f"{prefix}_{self._short_hash(name + tool.id + str(self._rng.random()))}"

        # Echo input arg value when response field name matches an input arg
        # (e.g. tool called with symbol=MSFT should return symbol=MSFT, not AMZN)
        if name in args and isinstance(args[name], (str, int, float, bool)):
            return args[name]

        # Status fields — fixed domain-appropriate vocabulary, never faker.word()
        if "status" in name_lower:
            return self._rng.choice(_STATUS_VALUES)

        # Timestamp/datetime fields — recent ISO datetime, never faker.word()
        if any(frag in name_lower for frag in _TIMESTAMP_FRAGMENTS):
            return self._recent_datetime()

        # (number-type fields already handled above)
        if field.type == "number":
            # Use word-boundary matching to prevent substrings like "change"
            # from matching "exchange_rate" (naive `in` would match wrongly).
            for range_key, (lo, hi) in _NUMERIC_RANGES.items():
                if self._range_key_matches(range_key, name_lower):
                    val = self._rng.uniform(lo, hi)
                    # Integer counts — never emit floats for room/spot/stop/volume counts
                    if range_key in ("stops", "volume") or any(
                        f in name_lower for f in _INTEGER_COUNT_FRAGMENTS
                    ):
                        return int(round(val))
                    return round(val, 2)
            # Generic number fallback — cap at a sane default range
            return round(self._rng.uniform(1.0, 100.0), 2)

        # "name" field — context-sensitive
        if name_lower == "name" or name_lower.endswith("_name"):
            if any(frag in name_lower for frag in _PERSON_NAME_FRAGMENTS):
                return self._call_faker("name")
            # Use domain-specific curated names instead of faker.company()
            # which produces law firms and tech startups
            tool_id_lower = tool.id.lower()
            if any(k in tool_id_lower for k in ("hotel", "booking_com", "airbnb", "hostel")):
                return self._rng.choice(_HOTEL_NAMES)
            if any(k in tool_id_lower for k in ("restaurant", "yelp", "opentable", "food")):
                return self._rng.choice(_RESTAURANT_NAMES)
            # Product/device names — laptops, electronics, items
            if any(k in tool_id_lower for k in ("laptop", "product", "item", "device", "equipment")):
                return self._rng.choice(_PRODUCT_NAMES)
            # Maps/nearby search returns place names (coffee shops, stores) — use company names
            if any(k in tool_id_lower for k in ("maps", "google_maps", "search_nearby", "places")):
                return self._call_faker("company")
            if tool.category in _BUSINESS_CATEGORIES:
                return self._call_faker("company")
            return self._call_faker("name")

        # Domain-specific string overrides
        if "symbol" in name_lower:
            return args.get("symbol") or self._rng.choice(
                ["AAPL", "GOOG", "MSFT", "AMZN", "META", "TSLA"]
            )
        if "currency" in name_lower or name_lower == "code":
            return self._rng.choice(["USD", "EUR", "GBP", "JPY", "CAD"])
        if "country" in name_lower:
            return self._rng.choice(["US", "UK", "FR", "DE", "JP", "CA"])

        # Event/calendar title fields — curated realistic titles, not faker gibberish
        if any(frag in name_lower for frag in ("title", "event_name", "subject", "summary")):
            if any(k in tool.id.lower() for k in ("calendar", "event", "schedule", "meeting")):
                return self._rng.choice(_EVENT_TITLES)

        # Note/description fields — curated short descriptions, not faker.sentence() gibberish
        if name_lower in ("description", "notes", "note", "details") and not any(
            k in tool.id.lower() for k in ("weather", "forecast", "climate")
        ):
            return self._rng.choice(_SHORT_DESCRIPTIONS)

        # Directions steps field — human-readable turn-by-turn, never faker gibberish
        if name_lower == "steps" and any(
            k in tool.id.lower() for k in ("directions", "route", "navigate", "maps")
        ):
            return "Head north, then turn right at the main intersection and continue straight."

        # Review/comment fields — use curated realistic phrases, not faker gibberish
        if any(frag in name_lower for frag in ("comment", "review", "feedback")):
            return self._rng.choice(_REVIEW_COMMENTS)

        # Date fields — booking/travel/forecast dates should be future; record dates can be past
        if "date" in name_lower:
            if any(frag in name_lower for frag in ("check_in", "check_out", "departure", "arrival", "booking", "travel", "start", "end", "event")):
                return self._future_date(max_days=90)
            if any(k in tool.id.lower() for k in ("forecast", "schedule", "event", "upcoming")):
                return self._future_date(max_days=14)  # forecasts max 2 weeks out
            return self._recent_date()

        # Weather description — curated human-readable phrases, not faker gibberish
        if name_lower == "description" and any(
            k in tool.id.lower() for k in ("weather", "forecast", "climate")
        ):
            # Description is picked per-field; we don't have precip_chance here,
            # so use a uniform random pick. Coherence with precip is handled
            # in _apply_coherence_rules after all fields are generated.
            return self._rng.choice(_WEATHER_DESCRIPTIONS)

        # Price range for restaurants/hotels — always $/$$/$$$/$$$$, never faker.word()
        if "price_range" in name_lower:
            return self._rng.choice(_PRICE_RANGES)

        # General field name hints
        for hint_key, faker_method in _FIELD_HINTS.items():
            if hint_key in name_lower:
                return self._call_faker(faker_method)

        # Fall back on type
        return self._type_value(field.type)

    def _recent_date(self) -> str:
        """Return a date string within the last 90 days."""
        import datetime
        days_ago = self._rng.randint(0, 90)
        d = datetime.date.today() - datetime.timedelta(days=days_ago)
        return d.isoformat()

    def _future_date(self, max_days: int = 30) -> str:
        """Return a date string 1–max_days in the future (for bookings, travel, forecasts)."""
        import datetime
        days_ahead = self._rng.randint(1, max_days)
        d = datetime.date.today() + datetime.timedelta(days=days_ahead)
        return d.isoformat()

    def _recent_datetime(self) -> str:
        """Return an ISO datetime string within the last 30 days."""
        import datetime
        days_ago = self._rng.randint(0, 30)
        hours_ago = self._rng.randint(0, 23)
        dt = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(
            days=days_ago, hours=hours_ago
        )
        return dt.isoformat()

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
    def _range_key_matches(range_key: str, name_lower: str) -> bool:
        """
        Match range_key against a field name using word-boundary semantics.

        Prevents "change" from matching "exchange_rate" by requiring the key
        to appear as a complete underscore-delimited segment (or exact match).

        Examples:
            "change"        vs "change_percent"  → True  (starts the name)
            "change"        vs "exchange_rate"   → False (mid-word: "xchange")
            "exchange_rate" vs "exchange_rate"   → True  (exact match)
            "rate"          vs "exchange_rate"   → True  (_rate at end)
        """
        import re
        return bool(re.search(
            r'(^|_)' + re.escape(range_key) + r'($|_)', name_lower
        ))

    @staticmethod
    def _short_hash(s: str) -> str:
        return hashlib.md5(s.encode()).hexdigest()[:6]
