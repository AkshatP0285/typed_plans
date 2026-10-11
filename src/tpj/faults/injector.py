"""Controlled fault injection.

Injected faults are the ground-truth labels for evaluating the gate
(precision / recall / F1, detection rate, false intervention rate). All
choices are deterministic given (seed, task id, step id, attempt).
"""
from __future__ import annotations

import json
import random
from typing import Any

from tpj.eval.trace import emit
from tpj.ir.dag import descendants
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan
from tpj.tools.spec import ToolSpec

PLAN_FAULTS = ["drop_step", "wrong_ref", "wrong_tool", "swap_args", "drop_arg", "wrong_literal"]
EXEC_FAULTS = ["exception", "timeout", "empty", "wrong_type", "schema_drift", "wrong_value"]
RAISING = {"exception", "timeout"}


class InjectedFault(Exception):
    pass


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
        self.plan_rate = plan_rate
        self.exec_rate = exec_rate
        self.plan_types = list(plan_types or PLAN_FAULTS)
        self.exec_types = list(exec_types or EXEC_FAULTS)
        self.seed = seed
        self.first_attempt_only = first_attempt_only

    def _rng(self, *parts: Any) -> random.Random:
        return random.Random(":".join(str(p) for p in (self.seed, *parts)))

    # ---------------- plan-level ----------------
    def corrupt_plan(self, plan: Plan, tools: list[ToolSpec], task_id: str) -> tuple[Plan, list[dict]]:
        rng = self._rng(task_id, "plan")
        if not plan.steps or rng.random() >= self.plan_rate:
            return plan, []
        kinds = self.plan_types[:]
        rng.shuffle(kinds)
        for kind in kinds:  # first applicable fault type
            new = plan.model_copy(deep=True)
            step_id = getattr(self, f"_plan_{kind}")(new, tools, rng)
            if step_id is not None:
                fault = {"type": kind, "step": step_id}
                emit("fault", level="plan", fault_type=kind, step=step_id)
                return new, [fault]
        return plan, []

    @staticmethod
    def _plan_drop_step(plan: Plan, tools, rng) -> str | None:
        used = {d for s in plan.steps for d in s.deps()}
        leaves = [s for s in plan.steps if s.id not in used]
        if len(plan.steps) < 2 or not leaves:
            return None
        victim = rng.choice(leaves)
        plan.steps = [s for s in plan.steps if s.id != victim.id]
        plan.final = [f for f in plan.final if f != victim.id]
        return victim.id

    @staticmethod
    def _plan_wrong_ref(plan: Plan, tools, rng) -> str | None:
        cands = [s for s in plan.steps if any(is_ref(v) for v in s.args.values())]
        rng.shuffle(cands)
        for s in cands:
            forbidden = descendants(plan, {s.id}) | {s.id}
            keys = [k for k, v in s.args.items() if is_ref(v)]
            k = rng.choice(keys)
            options = [o.id for o in plan.steps if o.id not in forbidden and o.id != s.args[k]["$ref"]]
            if options:
                s.args[k] = {"$ref": rng.choice(options)}
                return s.id
        return None

    @staticmethod
    def _plan_wrong_tool(plan: Plan, tools, rng) -> str | None:
        names = [t.name for t in tools]
        if len(names) < 2:
            return None
        s = rng.choice(plan.steps)
        s.tool = rng.choice([n for n in names if n != s.tool])
        return s.id

    @staticmethod
    def _plan_swap_args(plan: Plan, tools, rng) -> str | None:
        cands = [s for s in plan.steps if len(s.args) >= 2]
        if not cands:
            return None
        s = rng.choice(cands)
        a, b = rng.sample(list(s.args), 2)
        if s.args[a] == s.args[b]:
            return None
        s.args[a], s.args[b] = s.args[b], s.args[a]
        return s.id

    @staticmethod
    def _plan_drop_arg(plan: Plan, tools, rng) -> str | None:
        cands = [s for s in plan.steps if s.args]
        if not cands:
            return None
        s = rng.choice(cands)
        del s.args[rng.choice(list(s.args))]
        return s.id

    @staticmethod
    def _plan_wrong_literal(plan: Plan, tools, rng) -> str | None:
        cands = [(s, k) for s in plan.steps for k, v in s.args.items()
                 if isinstance(v, (int, float, str)) and not isinstance(v, bool)]
        if not cands:
            return None
        s, k = rng.choice(cands)
        v = s.args[k]
        s.args[k] = v + rng.randint(1, 9) if isinstance(v, (int, float)) else v[::-1] + "x"
        return s.id

    # ---------------- execution-level ----------------
    def exec_fault(self, task_id: str, step_id: str, attempt: int) -> str | None:
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
            return json.dumps(output, default=str)
        if kind == "schema_drift":
            return _rename_keys(output)
        if kind == "wrong_value":
            return _perturb(output, rng)
        raise ValueError(kind)


def _rename_keys(v: Any) -> Any:
    if isinstance(v, dict):
        return {f"{k}_v2": val for k, val in v.items()}
    if isinstance(v, list):
        return [_rename_keys(x) for x in v]
    return {"data": v}


def _perturb(v: Any, rng: random.Random) -> Any:
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return type(v)(v * rng.choice([0.1, 1.7, 10]) + rng.randint(1, 9))
    if isinstance(v, str):
        return v[::-1] if len(v) > 1 else v + "?"
    if isinstance(v, list):
        out = [_perturb(x, rng) for x in v]
        rng.shuffle(out)
        return out
    if isinstance(v, dict):
        return {k: _perturb(x, rng) for k, x in v.items()}
    return v
