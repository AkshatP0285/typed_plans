from __future__ import annotations

from abc import ABC, abstractmethod

from tpj.benchmarks.base import Task
from tpj.ir.schema import Plan


class Planner(ABC):
    @abstractmethod
    async def plan(self, task: Task, feedback: str | None = None) -> Plan:
        """Produce a plan. ``feedback`` describes a failed previous attempt (full re-plan)."""


class GoldPlanner(Planner):
    """Returns the benchmark's reference plan. Oracle upper bound and smoke tests."""

    async def plan(self, task: Task, feedback: str | None = None) -> Plan:
        if task.gold_plan is None:
            raise ValueError(f"task {task.id} has no gold plan")
        return task.gold_plan.model_copy(deep=True)
