"""Dependency graph utilities: wave levels, descendants, dirty regions."""
from __future__ import annotations

from collections import defaultdict

from tpj.ir.schema import Plan


class PlanStructureError(Exception):
    """The plan cannot be executed at all (dangling reference, cycle, duplicate id)."""


def check_structure(plan: Plan) -> None:
    ids = plan.ids()
    if len(ids) != len(set(ids)):
        raise PlanStructureError("duplicate step ids")
    known = set(ids)
    for s in plan.steps:
        missing = s.deps() - known
        if missing:
            raise PlanStructureError(f"step {s.id} references unknown steps {sorted(missing)}")
    for f in plan.final:
        if f not in known:
            raise PlanStructureError(f"final references unknown step {f}")
    waves(plan)  # raises on cycles


def waves(plan: Plan) -> list[list[str]]:
    """Group steps into levels: a step joins the first wave after all its deps.

    Reproduces Figure 3 of the proposal for the Goa-trip IR.
    """
    deps = {s.id: s.deps() for s in plan.steps}
    level: dict[str, int] = {}
    remaining = dict(deps)
    while remaining:
        ready = [sid for sid, d in remaining.items() if all(x in level for x in d)]
        if not ready:
            raise PlanStructureError(f"cycle or dangling reference among {sorted(remaining)}")
        for sid in ready:
            level[sid] = 1 + max((level[x] for x in remaining[sid]), default=-1)
            del remaining[sid]
    out: dict[int, list[str]] = defaultdict(list)
    for sid in plan.ids():
        out[level[sid]].append(sid)
    return [out[i] for i in sorted(out)]


def children(plan: Plan) -> dict[str, set[str]]:
    ch: dict[str, set[str]] = defaultdict(set)
    for s in plan.steps:
        for d in s.deps():
            ch[d].add(s.id)
    return ch


def descendants(plan: Plan, roots: set[str]) -> set[str]:
    ch = children(plan)
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        cur = stack.pop()
        for c in ch.get(cur, ()):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return seen


def changed_steps(old: Plan, new: Plan) -> set[str]:
    old_by_id = {s.id: s.model_dump() for s in old.steps}
    return {s.id for s in new.steps if old_by_id.get(s.id) != s.model_dump()}


def dirty_region(old: Plan, new: Plan, extra: set[str] | None = None) -> set[str]:
    """Steps of ``new`` that must (re-)run: changed/added steps, ``extra`` and all descendants."""
    roots = changed_steps(old, new) | ((extra or set()) & set(new.ids()))
    return roots | descendants(new, roots)
