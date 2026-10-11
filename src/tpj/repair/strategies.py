"""How a judge decision changes the plan and which steps must re-run."""
from __future__ import annotations

from abc import ABC, abstractmethod

from tpj.benchmarks.base import Task
from tpj.eval.trace import emit
from tpj.ir.dag import PlanStructureError, check_structure, descendants, dirty_region
from tpj.ir.schema import Plan
from tpj.planner.base import Planner
from tpj.repair.patch import JudgeDecision, PatchError, apply_patch


class AbortTask(Exception):
    """The judge decided the task cannot be completed."""


class RepairStrategy(ABC):
    """Turns a JudgeDecision into a new plan.

    Verdicts: ``repair`` (apply the judge's ops), ``retry`` (re-run the flagged
    steps unchanged), ``false_alarm`` (nothing to do), ``abort`` (raise AbortTask).
    """

    @abstractmethod
    async def apply(self, task: Task, plan: Plan, decision: JudgeDecision, flagged: set[str]) -> tuple[Plan, set[str]]:
        """Return the new plan and the set of step ids that must (re-)run."""


class LocalRepair(RepairStrategy):
    """Apply the judge's patch; re-run only changed steps and their descendants."""

    async def apply(self, task, plan, decision, flagged):
        if decision.verdict == "abort":
            raise AbortTask(decision.diagnosis)
        if decision.verdict == "false_alarm":
            return plan, set()
        if decision.verdict == "retry" or not decision.ops:
            return plan, flagged | descendants(plan, flagged)
        # An invalid patch falls back to a retry rather than failing the task.
        try:
            new = apply_patch(plan, decision)
            check_structure(new)
        except (PatchError, PlanStructureError) as e:
            emit("repair_invalid", error=str(e))
            return plan, flagged | descendants(plan, flagged)
        dirty = dirty_region(plan, new, extra=flagged)
        emit("repair", strategy="local", dirty=sorted(dirty), n_steps=len(new.steps))
        return new, dirty


class FullReplan(RepairStrategy):
    """Baseline: discard the plan and ask the planner again with the diagnosis."""

    def __init__(self, planner: Planner):
        self.planner = planner

    async def apply(self, task, plan, decision, flagged):
        if decision.verdict == "abort":
            raise AbortTask(decision.diagnosis)
        if decision.verdict == "false_alarm":
            return plan, set()
        feedback = f"Plan:\n{plan.to_prompt_json()}\nProblem at {sorted(flagged)}: {decision.diagnosis}"
        new = await self.planner.plan(task, feedback=feedback)
        emit("repair", strategy="full_replan", dirty=new.ids(), n_steps=len(new.steps))
        return new, set(new.ids())


class NoRepair(RepairStrategy):
    """Accept flagged outputs as they are (ablation)."""

    async def apply(self, task, plan, decision, flagged):
        return plan, set()
