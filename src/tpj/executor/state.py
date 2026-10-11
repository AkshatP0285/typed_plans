from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from tpj.ir.schema import Plan

Status = Literal["pending", "ok", "failed", "skipped"]


@dataclass
class NodeState:
    status: Status = "pending"
    output: Any = None
    error: str | None = None
    args: dict[str, Any] | None = None
    cached: bool = False
    attempts: int = 0
    fault: str | None = None


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
        for sid in list(self.nodes):
            if sid not in ids:
                del self.nodes[sid]
        for sid in plan.ids():
            self.nodes.setdefault(sid, NodeState())

    def reset(self, ids: set[str]) -> None:
        for sid in ids:
            if sid in self.nodes:
                n = self.nodes[sid]
                self.nodes[sid] = NodeState(attempts=n.attempts)

    def outputs(self) -> dict[str, Any]:
        """Values visible to references: ok outputs, ``None`` for skipped steps."""
        return {sid: (n.output if n.status == "ok" else None)
                for sid, n in self.nodes.items() if n.status in ("ok", "skipped")}

    def summary(self) -> dict[str, dict[str, Any]]:
        return {sid: {"status": n.status, "output": n.output, "error": n.error} for sid, n in self.nodes.items()}
