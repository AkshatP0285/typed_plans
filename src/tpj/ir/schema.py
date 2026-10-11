"""Typed intermediate representation (IR) for tool-calling plans.

A plan is a list of typed steps. Dependencies are expressed only through
references inside arguments (``{"$ref": "s1", "path": "first.id"}``) and
optional conditions; execution order and parallelism are derived later
(see :mod:`tpj.ir.dag`).
"""
from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field

from tpj.ir.refs import iter_refs

Safety = Literal["safe", "danger"]


class Condition(BaseModel):
    """Run the step only if ``resolve(ref, path) == equals``."""

    ref: str
    path: str | None = None
    equals: Any = True


class Step(BaseModel):
    id: str = Field(description="Step id, e.g. 's1'")
    tool: str = Field(description="Name of the tool to call")
    args: dict[str, Any] = Field(
        default_factory=dict,
        description='Literal values or references: {"$ref": "s1", "path": "field.sub"}',
    )
    returns: str = Field(default="Any", description="Declared return type, e.g. 'list[Flight]'")
    safety: Safety = Field(default="safe", description="'danger' for side-effecting steps")
    when: Condition | None = Field(default=None, description="Optional execution condition")
    after: list[str] = Field(
        default_factory=list,
        description="Ordering-only dependencies (no data flow). Rarely needed in typed IR.",
    )

    def deps(self) -> set[str]:
        deps = {ref for ref, _ in iter_refs(self.args)} | set(self.after)
        if self.when is not None:
            deps.add(self.when.ref)
        return deps


class PlanDraft(BaseModel):
    """What the planner LLM emits (the query is attached afterwards)."""

    steps: list[Step]
    final: list[str] = Field(default_factory=list, description="Step ids whose outputs answer the query")


class Plan(PlanDraft):
    query: str

    def step(self, step_id: str) -> Step:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(step_id)

    def ids(self) -> list[str]:
        return [s.id for s in self.steps]

    def to_prompt_json(self) -> str:
        return json.dumps(self.model_dump(exclude={"query"}, exclude_none=True), indent=1)

    @classmethod
    def from_draft(cls, draft: PlanDraft, query: str) -> "Plan":
        return cls(query=query, steps=draft.steps, final=draft.final)
