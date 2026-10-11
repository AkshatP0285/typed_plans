"""Dependency graph utilities: wave levels, descendants, dirty regions."""
from __future__ import annotations

from collections import defaultdict

from tpj.ir.schema import Plan


class PlanStructureError(Exception):
    """The plan cannot be executed at all (dangling reference, cycle, duplicate id)."""


def check_structure(plan: Plan) -> None:
    """Raise PlanStructureError unless every reference points to an existing step and there is no cycle."""
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

    A step with no dependencies is in wave 0; otherwise its wave is one more
    than the latest wave among its dependencies. Raises on cycles.
    """
    level: dict[str, int] = {}
    remaining = {step.id: step.deps() for step in plan.steps}  # step id -> ids it depends on
    while remaining:
        ready = [step_id for step_id, deps in remaining.items() if all(dep in level for dep in deps)]
        if not ready:  # every remaining step waits on another remaining step
            raise PlanStructureError(f"cycle or dangling reference among {sorted(remaining)}")
        for step_id in ready:
            level[step_id] = 1 + max((level[dep] for dep in remaining[step_id]), default=-1)
            del remaining[step_id]

    by_level: dict[int, list[str]] = defaultdict(list)
    for step_id in plan.ids():  # plan order within a wave
        by_level[level[step_id]].append(step_id)
    return [by_level[i] for i in sorted(by_level)]


def children(plan: Plan) -> dict[str, set[str]]:
    """step id -> ids of the steps that depend on it directly."""
    children_of: dict[str, set[str]] = defaultdict(set)
    for step in plan.steps:
        for dep in step.deps():
            children_of[dep].add(step.id)
    return children_of


def descendants(plan: Plan, roots: set[str]) -> set[str]:
    """Every step that depends on ``roots``, directly or indirectly (roots excluded)."""
    children_of = children(plan)
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        current = stack.pop()
        for child in children_of.get(current, ()):
            if child not in seen:
                seen.add(child)
                stack.append(child)
    return seen


def changed_steps(old: Plan, new: Plan) -> set[str]:
    """Steps of ``new`` that are new or differ from the step with the same id in ``old``."""
    old_by_id = {s.id: s.model_dump() for s in old.steps}
    return {s.id for s in new.steps if old_by_id.get(s.id) != s.model_dump()}


def dirty_region(old: Plan, new: Plan, extra: set[str] | None = None) -> set[str]:
    """Steps of ``new`` that must (re-)run: changed/added steps, ``extra`` and all descendants."""
    roots = changed_steps(old, new) | ((extra or set()) & set(new.ids()))
    return roots | descendants(new, roots)
