"""Controlled fault injection.

Injected faults are the ground-truth labels for evaluating the gate
(precision / recall / F1, detection rate, false intervention rate). All
choices are deterministic given (seed, task id, step id, attempt).

Two levels:
- plan faults corrupt the planner's plan before the pre-execution gate sees it
- execution faults make a tool call raise, or corrupt its output, before the
  post-execution gate sees it
"""
from __future__ import annotations

import json
import random
from collections.abc import Callable
from typing import Any

from tpj.eval.trace import emit
from tpj.ir.dag import descendants
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan
from tpj.tools.spec import ToolSpec

PLAN_FAULTS = ["drop_step", "wrong_ref", "wrong_tool", "swap_args", "drop_arg", "wrong_literal"]
EXEC_FAULTS = ["exception", "timeout", "empty", "wrong_type", "schema_drift", "wrong_value"]
RAISING = {"exception", "timeout"}  # execution faults that raise instead of returning a corrupted output


class InjectedFault(Exception):
    pass


# ---------------------------------------------------------------- plan-level corruptions
# Each function corrupts ``plan`` in place and returns the id of the step it changed,
# or None if the fault does not apply to this plan (e.g. no step has two arguments).

def _drop_step(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Remove a step that nothing depends on."""
    used = {dep for step in plan.steps for dep in step.deps()}
    leaves = [step for step in plan.steps if step.id not in used]
    if len(plan.steps) < 2 or not leaves:
        return None
    victim = rng.choice(leaves)
    plan.steps = [step for step in plan.steps if step.id != victim.id]
    plan.final = [step_id for step_id in plan.final if step_id != victim.id]
    return victim.id


def _wrong_ref(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Point one ``$ref`` argument at a different step (never one that would create a cycle)."""
    candidates = [step for step in plan.steps if any(is_ref(value) for value in step.args.values())]
    rng.shuffle(candidates)
    for step in candidates:
        forbidden = descendants(plan, {step.id}) | {step.id}
        ref_args = [name for name, value in step.args.items() if is_ref(value)]
        arg = rng.choice(ref_args)
        current_target = step.args[arg]["$ref"]
        options = [other.id for other in plan.steps if other.id not in forbidden and other.id != current_target]
        if options:
            step.args[arg] = {"$ref": rng.choice(options)}
            return step.id
    return None


def _wrong_tool(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Replace a step's tool with another available tool."""
    names = [tool.name for tool in tools]
    if len(names) < 2:
        return None
    step = rng.choice(plan.steps)
    step.tool = rng.choice([name for name in names if name != step.tool])
    return step.id


def _swap_args(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Swap the values of two arguments of one step."""
    candidates = [step for step in plan.steps if len(step.args) >= 2]
    if not candidates:
        return None
    step = rng.choice(candidates)
    first, second = rng.sample(list(step.args), 2)
    if step.args[first] == step.args[second]:
        return None
    step.args[first], step.args[second] = step.args[second], step.args[first]
    return step.id


def _drop_arg(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Delete one argument of one step."""
    candidates = [step for step in plan.steps if step.args]
    if not candidates:
        return None
    step = rng.choice(candidates)
    del step.args[rng.choice(list(step.args))]
    return step.id


def _wrong_literal(plan: Plan, tools: list[ToolSpec], rng: random.Random) -> str | None:
    """Change a literal number (add 1-9) or string (reverse it and append 'x')."""
    candidates = [(step, name) for step in plan.steps for name, value in step.args.items()
                  if isinstance(value, (int, float, str)) and not isinstance(value, bool)]
    if not candidates:
        return None
    step, name = rng.choice(candidates)
    value = step.args[name]
    step.args[name] = value + rng.randint(1, 9) if isinstance(value, (int, float)) else value[::-1] + "x"
    return step.id


PlanCorruptor = Callable[[Plan, list[ToolSpec], random.Random], "str | None"]

_PLAN_CORRUPTORS: dict[str, PlanCorruptor] = {
    "drop_step": _drop_step,
    "wrong_ref": _wrong_ref,
    "wrong_tool": _wrong_tool,
    "swap_args": _swap_args,
    "drop_arg": _drop_arg,
    "wrong_literal": _wrong_literal,
}


# ---------------------------------------------------------------- injector
class FaultInjector:
    def __init__(
        self,
        plan_rate: float = 0.0,
        exec_rate: float = 0.0,
        plan_types: list[str] | None = None,
        exec_types: list[str] | None = None,
        seed: int = 0,
        first_attempt_only: bool = True,
    ):
        self.plan_rate = plan_rate  # probability that a task's plan gets one fault
        self.exec_rate = exec_rate  # probability that a single tool call gets a fault
        self.plan_types = list(plan_types or PLAN_FAULTS)
        self.exec_types = list(exec_types or EXEC_FAULTS)
        self.seed = seed
        self.first_attempt_only = first_attempt_only  # re-runs after a repair are clean

    def _rng(self, *parts: Any) -> random.Random:
        """A random generator seeded by (seed, *parts): same inputs, same faults, every run."""
        return random.Random(":".join(str(p) for p in (self.seed, *parts)))

    def corrupt_plan(self, plan: Plan, tools: list[ToolSpec], task_id: str) -> tuple[Plan, list[dict]]:
        """Maybe inject one plan fault. Returns the (possibly corrupted) plan and the fault labels."""
        rng = self._rng(task_id, "plan")
        if not plan.steps or rng.random() >= self.plan_rate:
            return plan, []
        kinds = self.plan_types[:]
        rng.shuffle(kinds)
        for kind in kinds:  # try fault types in random order; use the first one that applies
            corrupted = plan.model_copy(deep=True)
            step_id = _PLAN_CORRUPTORS[kind](corrupted, tools, rng)
            if step_id is not None:
                emit("fault", level="plan", fault_type=kind, step=step_id)
                return corrupted, [{"type": kind, "step": step_id}]
        return plan, []

    def exec_fault(self, task_id: str, step_id: str, attempt: int) -> str | None:
        """Decide whether this tool call gets a fault, and which one (None = run normally)."""
        if self.exec_rate <= 0 or (self.first_attempt_only and attempt > 0):
            return None
        rng = self._rng(task_id, step_id, attempt)
        if rng.random() >= self.exec_rate:
            return None
        return rng.choice(self.exec_types)

    def apply_exec(self, kind: str, output: Any, task_id: str, step_id: str) -> Any:
        """Corrupt ``output`` (or raise) according to ``kind``."""
        rng = self._rng(task_id, step_id, kind)
        if kind == "exception":
            raise InjectedFault("503 Service Unavailable: upstream error")
        if kind == "timeout":
            raise TimeoutError("tool call timed out")
        if kind == "empty":
            return [] if isinstance(output, list) else ({} if isinstance(output, dict) else None)
        if kind == "wrong_type":
            return json.dumps(output, default=str)  # e.g. a list arrives as a JSON string
        if kind == "schema_drift":
            return _rename_keys(output)
        if kind == "wrong_value":
            return _perturb(output, rng)
        raise ValueError(kind)


def _rename_keys(value: Any) -> Any:
    """``{"price": 5}`` -> ``{"price_v2": 5}``; scalars get wrapped as ``{"data": value}``."""
    if isinstance(value, dict):
        return {f"{key}_v2": item for key, item in value.items()}
    if isinstance(value, list):
        return [_rename_keys(item) for item in value]
    return {"data": value}


def _perturb(value: Any, rng: random.Random) -> Any:
    """Same structure, wrong values: flip bools, scale numbers, reverse strings, shuffle lists."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return type(value)(value * rng.choice([0.1, 1.7, 10]) + rng.randint(1, 9))
    if isinstance(value, str):
        return value[::-1] if len(value) > 1 else value + "?"
    if isinstance(value, list):
        perturbed = [_perturb(item, rng) for item in value]
        rng.shuffle(perturbed)
        return perturbed
    if isinstance(value, dict):
        return {key: _perturb(item, rng) for key, item in value.items()}
    return value
