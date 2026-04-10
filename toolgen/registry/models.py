"""
Internal data model for the Tool Registry.

ToolBench JSON is inconsistent — this module defines the canonical normalized
representation that the rest of the pipeline depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Optional


# Normalized parameter types — all ToolBench types are mapped to these five.
# "unknown" is used when the original schema omits or cannot be parsed.
ParameterType = Literal["string", "number", "boolean", "array", "object", "unknown"]


@dataclass
class Parameter:
    """A single tool parameter, normalized from ToolBench JSON."""

    name: str
    type: ParameterType
    description: str
    required: bool
    default: Optional[Any] = None
    enum: Optional[list[Any]] = None

    def is_id_field(self) -> bool:
        """Heuristic: is this parameter likely an ID/reference field?"""
        id_suffixes = ("_id", "_key", "_token", "_ref", "_code", "_number")
        return any(self.name.endswith(s) for s in id_suffixes)


@dataclass
class ResponseField:
    """A single field in a response schema."""

    name: str
    type: ParameterType
    description: str
    is_array: bool = False
    nested_fields: list["ResponseField"] = field(default_factory=list)

    def is_id_field(self) -> bool:
        id_suffixes = ("_id", "_key", "_token", "_ref", "_code", "_number")
        return any(self.name.endswith(s) for s in id_suffixes)


@dataclass
class Tool:
    """
    Canonical representation of a ToolBench API endpoint.

    The ID format is "{category}/{api_name}/{endpoint_name}" — readable,
    filterable by prefix, and unique within the registry.
    """

    id: str                                   # canonical: category/api_name/endpoint_name
    category: str                             # ToolBench top-level category
    api_name: str
    endpoint_name: str
    description: str
    parameters: list[Parameter]
    required_params: list[str]                # derived, for fast lookup
    response_fields: list[ResponseField]      # parsed from response_schema; may be empty
    response_schema: Optional[dict]           # raw schema; None if absent
    raw: dict                                 # original JSON for debugging

    @property
    def optional_params(self) -> list[Parameter]:
        return [p for p in self.parameters if not p.required]

    @property
    def id_output_fields(self) -> list[ResponseField]:
        """Fields in the response that look like IDs/references."""
        return [f for f in self.response_fields if f.is_id_field()]

    @property
    def id_input_params(self) -> list[Parameter]:
        """Parameters that look like IDs/references."""
        return [p for p in self.parameters if p.is_id_field()]

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Tool) and self.id == other.id

    def to_schema_dict(self) -> dict:
        """
        Serialize to the format passed to the assistant agent as its tool list.
        Follows the OpenAI function-calling schema convention.
        """
        properties = {}
        for param in self.parameters:
            prop: dict[str, Any] = {
                "type": param.type if param.type != "unknown" else "string",
                "description": param.description,
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            properties[param.name] = prop

        return {
            "name": self.id,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": self.required_params,
            },
        }
