"""
ToolBench JSON loader.

Handles the real-world inconsistencies in ToolBench data:
  - Missing or null parameter types
  - required field absent (inferred from description text)
  - Response schema absent or malformed
  - Duplicate endpoint names within an API
  - Parameters encoded as a comma-separated string instead of a list
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Optional

from .models import Parameter, ParameterType, ResponseField, Tool

logger = logging.getLogger(__name__)

# Map raw ToolBench type strings to our normalized ParameterType
_TYPE_MAP: dict[str, ParameterType] = {
    "string": "string",
    "str": "string",
    "text": "string",
    "integer": "number",
    "int": "number",
    "float": "number",
    "double": "number",
    "number": "number",
    "boolean": "boolean",
    "bool": "boolean",
    "array": "array",
    "list": "array",
    "object": "object",
    "dict": "object",
    "json": "object",
}

# Keywords that suggest a parameter is required when there's no explicit field
_REQUIRED_KEYWORDS = ("required", "must", "mandatory", "necessary")


class ToolBenchLoader:
    """
    Load and normalize ToolBench JSON data into the internal Tool registry.

    ToolBench data can be structured in several ways:
      1. A directory of per-category JSON files
      2. A single JSON file with a top-level list of tools
      3. A single JSON file with a "tools" key

    Usage:
        loader = ToolBenchLoader()
        tools = loader.load("/path/to/toolbench/data")
    """

    def __init__(self, strict: bool = False):
        """
        Args:
            strict: if True, raise on malformed entries instead of skipping.
        """
        self.strict = strict
        self._seen_ids: set[str] = set()

    def load(self, source: str | Path) -> list[Tool]:
        """Load all tools from a file or directory."""
        source = Path(source)
        tools: list[Tool] = []

        if source.is_dir():
            json_files = sorted(source.rglob("*.json"))
            logger.info(f"Loading {len(json_files)} JSON files from {source}")
            for f in json_files:
                tools.extend(self._load_file(f))
        elif source.is_file():
            tools.extend(self._load_file(source))
        else:
            raise FileNotFoundError(f"Source not found: {source}")

        logger.info(f"Loaded {len(tools)} tools from {source}")
        return tools

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_file(self, path: Path) -> list[Tool]:
        try:
            with open(path) as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            if self.strict:
                raise
            logger.warning(f"Skipping {path}: {e}")
            return []

        # Normalize to a flat list of raw tool dicts
        if isinstance(data, list):
            raw_tools = data
        elif isinstance(data, dict) and "tools" in data:
            raw_tools = data["tools"]
        elif isinstance(data, dict):
            # Single tool
            raw_tools = [data]
        else:
            logger.warning(f"Unrecognized structure in {path}, skipping")
            return []

        tools = []
        for raw in raw_tools:
            try:
                tool = self._parse_tool(raw, source_file=path)
                if tool is not None:
                    tools.append(tool)
            except Exception as e:
                if self.strict:
                    raise
                logger.warning(f"Skipping malformed tool in {path}: {e}")

        return tools

    def _parse_tool(self, raw: dict, source_file: Path) -> Optional[Tool]:
        """Parse a single raw tool dict into our normalized model."""
        if not isinstance(raw, dict):
            return None

        # Extract core fields with fallbacks
        category = str(raw.get("category_name") or raw.get("category") or
                       source_file.stem or "unknown")
        api_name = str(raw.get("tool_name") or raw.get("api_name") or "unknown")
        endpoint_name = str(raw.get("api_name") or raw.get("endpoint_name") or
                            raw.get("name") or "unknown")
        description = str(
            raw.get("api_description") or raw.get("description") or
            f"{api_name} {endpoint_name}"
        ).strip()

        # Build canonical ID, deduplicating if needed
        base_id = f"{category}/{api_name}/{endpoint_name}"
        tool_id = self._unique_id(base_id)

        # Parse parameters
        raw_params = raw.get("required_parameters", []) + raw.get("optional_parameters", [])
        if isinstance(raw_params, str):
            # Comma-separated string fallback
            raw_params = [{"name": n.strip(), "type": None, "description": ""}
                          for n in raw_params.split(",")]

        required_names = {p.get("name") for p in raw.get("required_parameters", [])
                          if isinstance(p, dict) and p.get("name")}

        parameters = []
        for p in raw_params:
            if not isinstance(p, dict):
                continue
            param = self._parse_parameter(p, required_names)
            if param is not None:
                parameters.append(param)

        required_params = [p.name for p in parameters if p.required]

        # Parse response schema
        response_schema = raw.get("response") or raw.get("response_schema") or None
        response_fields = self._parse_response_schema(response_schema)

        return Tool(
            id=tool_id,
            category=category,
            api_name=api_name,
            endpoint_name=endpoint_name,
            description=description,
            parameters=parameters,
            required_params=required_params,
            response_fields=response_fields,
            response_schema=response_schema,
            raw=raw,
        )

    def _parse_parameter(self, raw: dict, required_names: set[str]) -> Optional[Parameter]:
        name = raw.get("name")
        if not name:
            return None
        name = str(name).strip()

        # Explicit parentheses: don't let the ternary swallow raw.get("type")
        raw_type = raw.get("type") or (
            raw.get("schema", {}).get("type") if isinstance(raw.get("schema"), dict) else None
        )
        param_type = self._normalize_type(raw_type)

        description = str(raw.get("description") or "").strip()

        # Determine required: explicit field, membership in required_parameters list,
        # or keyword inference from description
        explicit_required = raw.get("required")
        if explicit_required is not None:
            required = bool(explicit_required)
        elif name in required_names:
            required = True
        else:
            required = any(kw in description.lower() for kw in _REQUIRED_KEYWORDS)

        default = raw.get("default")

        # Parse enum
        enum = raw.get("enum") or raw.get("options")
        if isinstance(enum, str):
            enum = [e.strip() for e in enum.split(",")]

        return Parameter(
            name=name,
            type=param_type,
            description=description,
            required=required,
            default=default,
            enum=enum if isinstance(enum, list) else None,
        )

    def _parse_response_schema(self, schema: Any) -> list[ResponseField]:
        """Best-effort parse of a response schema into ResponseField list."""
        if not schema:
            return []
        if not isinstance(schema, dict):
            return []

        fields = []
        # Handle both {"properties": {...}} and flat {"field": {"type": ...}}
        properties = schema.get("properties") or schema
        if not isinstance(properties, dict):
            return []

        for name, field_def in properties.items():
            if not isinstance(field_def, dict):
                continue
            raw_type = field_def.get("type")
            is_array = raw_type == "array"
            field_type = self._normalize_type(
                field_def.get("items", {}).get("type") if is_array else raw_type
            )
            fields.append(ResponseField(
                name=name,
                type=field_type,
                description=str(field_def.get("description") or ""),
                is_array=is_array,
            ))

        return fields

    def _normalize_type(self, raw: Any) -> ParameterType:
        if raw is None:
            return "unknown"
        normalized = _TYPE_MAP.get(str(raw).lower().strip())
        if normalized is None:
            logger.debug(f"Unknown parameter type '{raw}', mapping to 'unknown'")
            return "unknown"
        return normalized

    def _unique_id(self, base_id: str) -> str:
        """Ensure the ID is unique by appending a suffix if needed."""
        if base_id not in self._seen_ids:
            self._seen_ids.add(base_id)
            return base_id
        for i in range(2, 100):
            candidate = f"{base_id}_v{i}"
            if candidate not in self._seen_ids:
                self._seen_ids.add(candidate)
                logger.debug(f"Duplicate tool ID '{base_id}' → '{candidate}'")
                return candidate
        raise RuntimeError(f"Could not generate unique ID for '{base_id}'")
