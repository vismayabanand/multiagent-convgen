"""Unit tests for the offline execution model (session + mock generator)."""

import pytest

from toolgen.executor.mock_generator import MockGenerator
from toolgen.executor.session import ExecutionSession
from toolgen.registry.models import Parameter, ResponseField, Tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_tool(tool_id="Travel/hotels/search", params=None, response_fields=None):
    params = params or [
        Parameter("city", "string", "City", required=True),
        Parameter("hotel_id", "string", "Hotel ID", required=False),
    ]
    # Use `is None` so that passing an explicit empty list [] is honoured
    if response_fields is None:
        response_fields = [
            ResponseField("hotel_id", "string", "Hotel ID"),
            ResponseField("name", "string", "Hotel name"),
            ResponseField("price", "number", "Price"),
        ]
    return Tool(
        id=tool_id,
        category="Travel",
        api_name="hotels",
        endpoint_name="search",
        description="Search hotels",
        parameters=params,
        required_params=[p.name for p in params if p.required],
        response_fields=response_fields,
        response_schema=None,
        raw={},
    )


@pytest.fixture
def mock_gen():
    return MockGenerator(seed=42)


@pytest.fixture
def session(mock_gen):
    return ExecutionSession(mock_gen)


# ---------------------------------------------------------------------------
# MockGenerator tests
# ---------------------------------------------------------------------------

class TestMockGenerator:

    def test_generates_dict_response(self, mock_gen):
        tool = make_tool()
        response = mock_gen.generate(tool, {"city": "Paris"})
        assert isinstance(response, dict)

    def test_schema_derived_includes_all_response_fields(self, mock_gen):
        tool = make_tool()
        response = mock_gen.generate(tool, {"city": "Paris"})
        assert "hotel_id" in response
        assert "name" in response
        assert "price" in response

    def test_id_fields_get_prefixed_values(self, mock_gen):
        tool = make_tool()
        response = mock_gen.generate(tool, {"city": "Paris"})
        hotel_id = response.get("hotel_id", "")
        assert isinstance(hotel_id, str)
        assert len(hotel_id) > 0

    def test_number_fields_are_numeric(self, mock_gen):
        tool = make_tool(response_fields=[
            ResponseField("price", "number", "Price"),
        ])
        response = mock_gen.generate(tool, {})
        assert isinstance(response["price"], (int, float))

    def test_boolean_fields_are_bool(self, mock_gen):
        tool = make_tool(response_fields=[
            ResponseField("available", "boolean", "Availability"),
        ])
        response = mock_gen.generate(tool, {})
        assert isinstance(response["available"], bool)

    def test_array_fields_are_lists(self, mock_gen):
        tool = make_tool(response_fields=[
            ResponseField("amenities", "string", "Amenities", is_array=True),
        ])
        response = mock_gen.generate(tool, {})
        assert isinstance(response["amenities"], list)

    def test_generic_fallback_when_no_schema(self, mock_gen):
        tool = make_tool(response_fields=[])
        response = mock_gen.generate(tool, {})
        assert "status" in response

    def test_same_seed_produces_same_id(self):
        gen1 = MockGenerator(seed=42)
        gen2 = MockGenerator(seed=42)
        tool = make_tool()
        r1 = gen1.generate(tool, {"city": "Paris"})
        r2 = gen2.generate(tool, {"city": "Paris"})
        assert r1.get("hotel_id") == r2.get("hotel_id")


# ---------------------------------------------------------------------------
# ExecutionSession tests
# ---------------------------------------------------------------------------

class TestExecutionSession:

    def test_execute_returns_dict(self, session):
        tool = make_tool()
        response = session.execute(tool, {"city": "Paris"})
        assert isinstance(response, dict)

    def test_state_updated_after_execution(self, session):
        tool = make_tool()
        session.execute(tool, {"city": "Paris"})
        # hotel_id should now be in session state
        assert "hotel_id" in session.state

    def test_arg_grounded_from_state(self, session):
        # First call populates hotel_id in state
        search_tool = make_tool("Travel/hotels/search", response_fields=[
            ResponseField("hotel_id", "string", "Hotel ID"),
        ])
        session.execute(search_tool, {"city": "Paris"})
        actual_hotel_id = session.state.get("hotel_id")

        # Second call with a "wrong" hotel_id — should be overridden by state
        book_tool = make_tool("Travel/hotels/book", params=[
            Parameter("hotel_id", "string", "Hotel ID", required=True),
        ])
        book_response = session.execute(book_tool, {"hotel_id": "hallucinated_id_xyz"})

        # The resolved arg in history should be the state value, not the hallucination
        last_record = session.history[-1]
        assert last_record.resolved_args.get("hotel_id") == actual_hotel_id

    def test_history_records_each_call(self, session):
        tool = make_tool()
        session.execute(tool, {"city": "Paris"})
        session.execute(tool, {"city": "London"})
        assert len(session.history) == 2

    def test_non_id_args_not_overridden(self, session):
        tool = make_tool(params=[
            Parameter("city", "string", "City", required=True),
        ])
        # city is not an ID field; state should not interfere
        session.state["city"] = "London"  # manually set state
        response = session.execute(tool, {"city": "Paris"})
        # city should NOT be overridden (it's not an ID field)
        last = session.history[-1]
        assert last.resolved_args.get("city") == "Paris"

    def test_state_summary_is_string(self, session):
        tool = make_tool()
        session.execute(tool, {"city": "Paris"})
        summary = session.get_state_summary()
        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_nested_response_ids_extracted(self, session):
        tool = make_tool(response_fields=[
            ResponseField("result", "object", "Result object"),
        ])
        # Manually override generate to return nested structure
        session.mock_generator.generate = lambda t, a: {
            "result": {"booking_id": "bk_123", "name": "Hotel"}
        }
        session.execute(tool, {})
        assert "booking_id" in session.state
        assert session.state["booking_id"] == "bk_123"

    def test_empty_state_summary_message(self, mock_gen):
        fresh_session = ExecutionSession(mock_gen)
        summary = fresh_session.get_state_summary()
        assert "No prior" in summary
