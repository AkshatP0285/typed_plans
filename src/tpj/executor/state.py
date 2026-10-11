"""Runtime status of every step in the plan while it executes."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tpj.ir.schema import Plan

Status = Literal["pending", "ok", "failed", "skipped"]


@dataclass
class NodeState:
    """One step's status. A "node" is a step of the plan seen as a node of the dependency graph."""

    status: Status = "pending"
    output: Any = None
    error: str | None = None
    args: dict[str, Any] | None = None
    cached: bool = False  # output came from the cache instead of a tool call
    attempts: int = 0  # how many times the tool was actually called
    fault: str | None = None  # injected execution fault, if any


@dataclass
class ExecutionState:
    nodes: dict[str, NodeState] = field(default_factory=dict)

    @classmethod
    def for_plan(cls, plan: Plan) -> "ExecutionState":
        st = cls()
        st.sync(plan)
        return st

    def sync(self, plan: Plan) -> None:
        """Add new steps as pending and forget removed ones."""
        ids = set(plan.ids())
        for step_id in list(self.nodes):
            if step_id not in ids:
                del self.nodes[step_id]
        for step_id in plan.ids():
            self.nodes.setdefault(step_id, NodeState())

    def reset(self, ids: set[str]) -> None:
        """Mark steps as pending again so they re-run. The attempt count is kept."""
        for step_id in ids:
            if step_id in self.nodes:
                self.nodes[step_id] = NodeState(attempts=self.nodes[step_id].attempts)

    def outputs(self) -> dict[str, Any]:
        """Values visible to references: ok outputs, ``None`` for skipped steps."""
        return {step_id: (node.output if node.status == "ok" else None)
                for step_id, node in self.nodes.items() if node.status in ("ok", "skipped")}

    def summary(self) -> dict[str, dict[str, Any]]:
        """Status, output and error per step, as shown to the LLM judge."""
        return {step_id: {"status": node.status, "output": node.output, "error": node.error}
                for step_id, node in self.nodes.items()}
