"""Reference handling inside step arguments.

A reference is a dict ``{"$ref": "<step id>", "path": "a.b.0"}``. References may
appear anywhere inside argument values (nested lists / dicts).
"""
from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

REF_KEY = "$ref"
_MISSING = object()


class RefResolutionError(Exception):
    pass


def is_ref(value: Any) -> bool:
    return isinstance(value, dict) and REF_KEY in value


def iter_refs(value: Any) -> Iterator[tuple[str, str | None]]:
    if is_ref(value):
        yield value[REF_KEY], value.get("path")
    elif isinstance(value, Mapping):
        for v in value.values():
            yield from iter_refs(v)
    elif isinstance(value, (list, tuple)):
        for v in value:
            yield from iter_refs(v)


def get_path(value: Any, path: str | None, lenient: bool = True) -> Any:
    """Follow a dotted path (``first.id``, ``items.0.price``) into a tool output.

    ``first`` / ``last`` index lists. With ``lenient``, a single-key path on a
    scalar returns the scalar itself (tools often return ``5`` where the schema
    promised ``{"result": 5}``).
    """
    if not path:
        return value
    parts = path.split(".")
    cur = value
    for i, part in enumerate(parts):
        nxt = _step_into(cur, part)
        if nxt is _MISSING:
            if lenient and i == len(parts) - 1 and not isinstance(cur, (dict, list, tuple)):
                return cur
            raise RefResolutionError(f"cannot follow '{part}' in path '{path}' on {type(cur).__name__}")
        cur = nxt
    return cur


def _step_into(cur: Any, part: str) -> Any:
    if isinstance(cur, Mapping):
        return cur.get(part, _MISSING)
    if isinstance(cur, (list, tuple)):
        if part == "first":
            return cur[0] if cur else _MISSING
        if part == "last":
            return cur[-1] if cur else _MISSING
        if part.lstrip("-").isdigit():
            idx = int(part)
            return cur[idx] if -len(cur) <= idx < len(cur) else _MISSING
        return _MISSING
    return getattr(cur, part, _MISSING)


def resolve(value: Any, outputs: Mapping[str, Any], lenient: bool = True) -> Any:
    """Replace every reference in ``value`` with the referenced output."""
    if is_ref(value):
        ref = value[REF_KEY]
        if ref not in outputs:
            raise RefResolutionError(f"unknown or unfinished step '{ref}'")
        return get_path(outputs[ref], value.get("path"), lenient)
    if isinstance(value, Mapping):
        return {k: resolve(v, outputs, lenient) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, outputs, lenient) for v in value]
    return value
