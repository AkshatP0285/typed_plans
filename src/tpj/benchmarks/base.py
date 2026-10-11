from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from tpj.ir.schema import Plan
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec


@dataclass
class Task:
    id: str
    query: str
    tools: list[ToolSpec]
    gold: dict[str, Any] = field(default_factory=dict)
    gold_plan: Plan | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class CallRecord:
    step_id: str | None
    tool: str
    args: dict[str, Any]
    output: Any = None
    error: str | None = None
    cached: bool = False


@dataclass
class AgentResult:
    task_id: str
    status: str  # completed | failed | plan_error
    final: dict[str, Any] = field(default_factory=dict)
    calls: list[CallRecord] = field(default_factory=list)
    plan: Plan | None = None
    initial_plan: Plan | None = None
    answer_text: str | None = None
    error: str | None = None

    def successful_calls(self) -> list[CallRecord]:
        return [c for c in self.calls if c.error is None]

    def trajectory(self) -> list[CallRecord]:
        """Latest call per step, in first-seen order.

        A step re-executed after a repair counts once, and steps the judge removed
        from the plan are dropped. Without a plan (ReAct) every call is kept.
        """
        latest: dict[str, CallRecord] = {}
        for i, c in enumerate(self.calls):
            latest[c.step_id or f"_{i}"] = c
        if self.plan is not None:
            live = set(self.plan.ids())
            return [c for k, c in latest.items() if k in live]
        return list(latest.values())

    def final_value(self) -> Any:
        """Single final output if the plan names one step, else the dict."""
        if len(self.final) == 1:
            return next(iter(self.final.values()))
        return self.final


class Benchmark(ABC):
    name: str = "base"

    @abstractmethod
    def load(self) -> list[Task]: ...

    @abstractmethod
    def registry(self, task: Task) -> ToolRegistry: ...

    @abstractmethod
    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        """Must include ``success`` in [0, 1]."""
