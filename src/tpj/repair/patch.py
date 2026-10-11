"""Local IR edits produced by the LLM judge."""
from __future__ import annotations

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field

from tpj.ir.schema import Condition, Plan, Step


class EditOp(BaseModel):
    """Change some fields of an existing step; fields left as None stay unchanged."""

    op: Literal["edit"] = "edit"
    id: str
    tool: str | None = None
    args: dict[str, Any] | None = None
    returns: str | None = None
    safety: Literal["safe", "danger"] | None = None
    when: Condition | None = None
    clear_when: bool = False


class RemoveOp(BaseModel):
    op: Literal["remove"] = "remove"
    id: str


class ReplaceOp(BaseModel):
    op: Literal["replace"] = "replace"
    id: str
    step: Step


class AddOp(BaseModel):
    op: Literal["add"] = "add"
    step: Step


# Pydantic picks the right op class from the "op" field when parsing the judge's JSON.
PatchOp = Annotated[Union[EditOp, RemoveOp, ReplaceOp, AddOp], Field(discriminator="op")]


class JudgeDecision(BaseModel):
    verdict: Literal["repair", "retry", "false_alarm", "abort"] = Field(
        description="repair: apply ops; retry: transient failure, re-run as is; "
        "false_alarm: outputs are fine; abort: the task cannot be completed"
    )
    diagnosis: str = ""
    ops: list[PatchOp] = Field(default_factory=list)
    final: list[str] | None = Field(default=None, description="New final step ids, if they change")


class PatchError(Exception):
    pass


def apply_patch(plan: Plan, decision: JudgeDecision) -> Plan:
    """Apply the judge's ops in order to a copy of ``plan``. Raises PatchError on an invalid op."""
    new = plan.model_copy(deep=True)
    for op in decision.ops:
        # Recomputed per op because earlier ops may have added or removed steps.
        position = {step.id: i for i, step in enumerate(new.steps)}

        if isinstance(op, AddOp):
            if op.step.id in position:
                raise PatchError(f"add: id {op.step.id} already exists")
            new.steps.append(op.step)
            continue

        if op.id not in position:
            raise PatchError(f"{op.op}: unknown step {op.id}")
        i = position[op.id]

        if isinstance(op, RemoveOp):
            del new.steps[i]
            new.final = [step_id for step_id in new.final if step_id != op.id]
        elif isinstance(op, ReplaceOp):
            new.steps[i] = op.step.model_copy(update={"id": op.id})  # keep the id so refs still point here
        else:  # EditOp: overwrite only the fields the judge filled in
            changes = {field: value
                       for field, value in op.model_dump(exclude={"op", "id", "clear_when"}).items()
                       if value is not None}
            if "when" in changes:
                changes["when"] = op.when  # keep it a Condition object, not the dumped dict
            step = new.steps[i].model_copy(update=changes)
            if op.clear_when:
                step.when = None
            new.steps[i] = step

    if decision.final is not None:
        new.final = list(decision.final)
    return new
