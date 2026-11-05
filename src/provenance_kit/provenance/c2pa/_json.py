"""The JSON value type used for manifest definitions and parsed assertions (internal)."""

from __future__ import annotations

from typing import TypeAlias

__all__ = ["JsonObject", "JsonValue"]

#: A value decoded from (or encodable as) JSON.
JsonValue: TypeAlias = "bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None"

#: A JSON object.
JsonObject: TypeAlias = "dict[str, JsonValue]"
