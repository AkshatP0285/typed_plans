"""Deterministic mock tools for benchmarks that ship no executable tools."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from tpj.tools.spec import ToolSpec


def _digest(*parts: Any) -> str:
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:10]


def mock_value(type_name: str, seed: str) -> Any:
    t = (type_name or "Any").lower()
    h = int(seed, 16)
    if t.startswith("list"):
        inner = t[5:-1] if t.endswith("]") else "any"
        return [mock_value(inner, _digest(seed, i)) for i in range(3)]
    if t in {"int", "integer"}:
        return h % 1000
    if t in {"float", "number"}:
        return round((h % 100000) / 100, 2)
    if t in {"bool", "boolean"}:
        return bool(h % 2)
    if t in {"str", "string", "text"}:
        return f"<{seed}>"
    return {"id": seed, "value": f"<{type_name}:{seed}>"}


def make_mock(spec: ToolSpec, table: dict[str, Any] | None = None) -> Callable[..., Any]:
    """Return a callable producing stable outputs for identical calls.

    ``table`` maps ``_digest(args)`` to a fixed output (e.g. gold replay).
    If the spec has an object ``output_schema``, a dict with those keys is returned.
    """
    props = (spec.output_schema or {}).get("properties") if spec.output_schema else None

    def _tool(**kwargs: Any) -> Any:
        key = _digest(kwargs)
        if table and key in table:
            return table[key]
        seed = _digest(spec.name, kwargs)
        if props:
            return {k: mock_value(v.get("type", "Any") if isinstance(v, dict) else "Any", _digest(seed, k))
                    for k, v in props.items()}
        return mock_value(spec.returns, seed)

    _tool.__name__ = spec.name
    return _tool
