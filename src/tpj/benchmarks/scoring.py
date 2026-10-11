"""Small comparison / metric helpers shared by the benchmark adapters."""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable
from typing import Any


def values_equal(a: Any, b: Any, rel_tol: float = 1e-3) -> bool:
    """Lenient deep equality for comparing a predicted answer with a gold answer.

    Numbers compare with a tolerance, strings ignore case and surrounding whitespace,
    and dicts / lists are compared element by element with the same rules.
    """
    # bool is a subclass of int, so check it first: True must not "equal" 1.0 here.
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=rel_tol, abs_tol=1e-6)
    if isinstance(a, str) and isinstance(b, str):
        return a.strip().lower() == b.strip().lower()
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return False
        return all(values_equal(a[key], b[key], rel_tol) for key in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            return False
        return all(values_equal(x, y, rel_tol) for x, y in zip(a, b))
    return a == b


def unwrap(value: Any) -> Any:
    """``{"result": 5}`` -> ``5`` (single-key dicts), recursively."""
    while isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    return value


def f1(pred: Iterable[Any], gold: Iterable[Any]) -> dict[str, float]:
    """Precision / recall / F1 over multisets (duplicates count, order does not)."""
    pred_counts, gold_counts = Counter(pred), Counter(gold)
    if not pred_counts and not gold_counts:
        # Nothing predicted and nothing expected: a perfect (empty) answer.
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    true_positives = sum((pred_counts & gold_counts).values())  # & = per-item minimum count
    precision = true_positives / sum(pred_counts.values()) if pred_counts else 0.0
    recall = true_positives / sum(gold_counts.values()) if gold_counts else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1_score}
