"""Deterministic mock tools for benchmarks that ship no executable tools."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from tpj.tools.spec import ToolSpec


def _digest(*parts: Any) -> str:
    """Short stable hash (10 hex chars) of JSON-serialisable ``parts``; used as a deterministic seed."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def mock_value(type_name: str, seed: str) -> Any:
    """A fake value of type ``type_name`` (e.g. ``"int"``, ``"list[Flight]"``), fully determined by the hex ``seed``."""
    type_lower = (type_name or "Any").lower()
    seed_int = int(seed, 16)
    if type_lower.startswith("list"):
        # "list[x]" -> three values of type x, each with its own derived seed
        inner = type_lower[5:-1] if type_lower.endswith("]") else "any"
        return [mock_value(inner, _digest(seed, i)) for i in range(3)]
    if type_lower in {"int", "integer"}:
        return seed_int % 1000
    if type_lower in {"float", "number"}:
        return round((seed_int % 100000) / 100, 2)
    if type_lower in {"bool", "boolean"}:
        return bool(seed_int % 2)
    if type_lower in {"str", "string", "text"}:
        return f"<{seed}>"
    # any other (object) type: a small dict that is recognisable in traces
    return {"id": seed, "value": f"<{type_name}:{seed}>"}


def make_mock(spec: ToolSpec, table: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Return a callable producing stable outputs for identical calls.

    ``table`` maps ``_digest(args)`` to a fixed output (e.g. gold replay).
    If the spec has an object ``output_schema``, a dict with those keys is returned.
    """
    output_fields = spec.output_schema.get("properties") if spec.output_schema else None

    def _tool(**kwargs: Any) -> Any:
        args_key = _digest(kwargs)
        if table and args_key in table:
            return table[args_key]
        seed = _digest(spec.name, kwargs)
        if output_fields:
            output = {}
            for field_name, field_schema in output_fields.items():
                field_type = field_schema.get("type", "Any") if isinstance(field_schema, dict) else "Any"
                output[field_name] = mock_value(field_type, _digest(seed, field_name))
            return output
        return mock_value(spec.returns, seed)

    _tool.__name__ = spec.name
    return _tool
