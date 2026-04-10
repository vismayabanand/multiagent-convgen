"""Unit tests for the Tool Registry loader and data model."""

import json
import tempfile
from pathlib import Path

import pytest

from toolgen.registry.loader import ToolBenchLoader
from toolgen.registry.models import Parameter, Tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_raw_tool(
    category="Travel",
    tool_name="hotels",
    api_name="search_hotels",
    description="Search for hotels",
    required_params=None,
    optional_params=None,
    response=None,
):
    return {
        "category_name": category,
        "tool_name": tool_name,
        "api_name": api_name,
        "api_description": description,
        "required_parameters": required_params or [
            {"name": "city", "type": "string", "description": "City name", "default": None}
        ],
        "optional_parameters": optional_params or [
            {"name": "max_price", "type": "number", "description": "Max price", "default": 500}
        ],
        "response": response or {
            "properties": {
                "hotel_id": {"type": "string", "description": "Hotel identifier"},
                "name": {"type": "string", "description": "Hotel name"},
                "price": {"type": "number", "description": "Price per night"},
            }
        },
    }


@pytest.fixture
def loader():
    return ToolBenchLoader()


@pytest.fixture
def tmp_json_file(tmp_path):
    def _make(data):
        f = tmp_path / "tools.json"
        f.write_text(json.dumps(data))
        return f
    return _make


# ---------------------------------------------------------------------------
# Loader tests
# ---------------------------------------------------------------------------

class TestToolBenchLoader:

    def test_loads_single_tool(self, loader, tmp_json_file):
        raw = make_raw_tool()
        path = tmp_json_file(raw)
        tools = loader.load(path)
        assert len(tools) == 1
        assert isinstance(tools[0], Tool)

    def test_loads_list_of_tools(self, loader, tmp_json_file):
        raw = [make_raw_tool(), make_raw_tool(tool_name="flights", api_name="search_flights")]
        path = tmp_json_file(raw)
        tools = loader.load(path)
        assert len(tools) == 2

    def test_loads_tools_key(self, loader, tmp_json_file):
        raw = {"tools": [make_raw_tool()]}
        path = tmp_json_file(raw)
        tools = loader.load(path)
        assert len(tools) == 1

    def test_canonical_id_format(self, loader, tmp_json_file):
        raw = make_raw_tool(category="Travel", tool_name="hotels", api_name="search")
        path = tmp_json_file(raw)
        tools = loader.load(path)
        assert "/" in tools[0].id
        parts = tools[0].id.split("/")
        assert len(parts) == 3

    def test_required_params_derived_correctly(self, loader, tmp_json_file):
        raw = make_raw_tool(
            required_params=[{"name": "city", "type": "string", "description": "City"}],
            optional_params=[{"name": "price", "type": "number", "description": "Price"}],
        )
        path = tmp_json_file(raw)
        tools = loader.load(path)
        tool = tools[0]
        assert "city" in tool.required_params
        assert "price" not in tool.required_params

    def test_null_type_becomes_unknown(self, loader, tmp_json_file):
        raw = make_raw_tool(
            required_params=[{"name": "query", "type": None, "description": "Search query"}]
        )
        path = tmp_json_file(raw)
        tools = loader.load(path)
        param = next(p for p in tools[0].parameters if p.name == "query")
        assert param.type == "unknown"

    def test_type_normalization(self, loader, tmp_json_file):
        raw = make_raw_tool(
            required_params=[
                {"name": "count", "type": "integer", "description": "Count"},
                {"name": "flag", "type": "bool", "description": "Flag"},
                {"name": "items", "type": "list", "description": "Items"},
            ]
        )
        path = tmp_json_file(raw)
        tools = loader.load(path)
        param_types = {p.name: p.type for p in tools[0].parameters}
        assert param_types["count"] == "number"
        assert param_types["flag"] == "boolean"
        assert param_types["items"] == "array"

    def test_missing_description_falls_back(self, loader, tmp_json_file):
        raw = make_raw_tool()
        del raw["api_description"]
        path = tmp_json_file(raw)
        tools = loader.load(path)
        assert tools[0].description != ""

    def test_duplicate_ids_are_deduplicated(self, loader, tmp_json_file):
        raw = [make_raw_tool(), make_raw_tool()]  # identical, will clash
        path = tmp_json_file(raw)
        tools = loader.load(path)
        ids = [t.id for t in tools]
        assert len(set(ids)) == len(ids), "Duplicate IDs should be deduplicated"

    def test_malformed_json_skipped_gracefully(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        loader = ToolBenchLoader(strict=False)
        tools = loader.load(bad_file)
        assert tools == []

    def test_strict_mode_raises_on_bad_json(self, tmp_path):
        bad_file = tmp_path / "bad.json"
        bad_file.write_text("{not valid json")
        loader = ToolBenchLoader(strict=True)
        with pytest.raises(Exception):
            loader.load(bad_file)

    def test_response_fields_parsed(self, loader, tmp_json_file):
        raw = make_raw_tool()
        path = tmp_json_file(raw)
        tools = loader.load(path)
        tool = tools[0]
        field_names = {f.name for f in tool.response_fields}
        assert "hotel_id" in field_names

    def test_loads_directory(self, tmp_path):
        for i in range(3):
            f = tmp_path / f"tool_{i}.json"
            f.write_text(json.dumps(make_raw_tool(api_name=f"api_{i}")))
        loader = ToolBenchLoader()
        tools = loader.load(tmp_path)
        assert len(tools) == 3

    def test_to_schema_dict(self, loader, tmp_json_file):
        raw = make_raw_tool()
        path = tmp_json_file(raw)
        tools = loader.load(path)
        schema = tools[0].to_schema_dict()
        assert "name" in schema
        assert "description" in schema
        assert "parameters" in schema
        assert schema["parameters"]["type"] == "object"
        assert "city" in schema["parameters"]["properties"]


# ---------------------------------------------------------------------------
# Parameter model tests
# ---------------------------------------------------------------------------

class TestParameter:

    def test_is_id_field_detection(self):
        assert Parameter("hotel_id", "string", "", True).is_id_field()
        assert Parameter("booking_ref", "string", "", True).is_id_field()
        assert Parameter("access_token", "string", "", True).is_id_field()
        assert not Parameter("city", "string", "", True).is_id_field()
        assert not Parameter("price", "number", "", False).is_id_field()
