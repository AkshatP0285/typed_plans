from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any


def values_equal(a: Any, b: Any, rel_tol: float = 1e-3) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(values_equal(a[k], b[k], rel_tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(values_equal(x, y, rel_tol) for x, y in zip(a, b))
    return a == b


def unwrap(v: Any) -> Any:
    """``{"result": 5}`` -> ``5`` (single-key dicts), recursively."""
    while isinstance(v, dict) and len(v) == 1:
        v = next(iter(v.values()))
    return v


def f1(pred: Iterable[Any], gold: Iterable[Any]) -> dict[str, float]:
    p, g = Counter(pred), Counter(gold)
    if not p and not g:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    tp = sum((p & g).values())
    prec = tp / sum(p.values()) if p else 0.0
    rec = tp / sum(g.values()) if g else 0.0
    return {"precision": prec, "recall": rec, "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0}
