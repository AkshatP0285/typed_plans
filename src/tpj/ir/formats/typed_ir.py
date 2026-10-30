"""Typed IR plan format: the planner writes our IR directly as JSON (see :class:`~tpj.ir.schema.PlanDraft`).

Data flow is explicit via ``{"$ref": "sK"}`` references, and each step declares its
return type and safety, which the other two formats lack.
"""
from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from tpj.ir.formats.base import PlanFormat, PlanParseError
from tpj.ir.schema import Plan, PlanDraft
from tpj.llm.client import parse_json

INSTRUCTIONS = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
Use the minimum number of steps that fulfils the request - never repeat the same tool call,
or a near-duplicate of one, more than once with the same intent.
Each step has:
- "id": "s1", "s2", ... (unique)
- "tool": exact tool name from the tool list
- "args": argument values. To use the output of an earlier step write {"$ref": "sK"} or
  {"$ref": "sK", "path": "field.sub"} ("first"/"last"/integers index lists). Never copy
  values you have not seen; reference them.
- "returns": the type the step produces, e.g. "list[Flight]", "Booking", "bool", "float"
- "safety": "danger" if the call changes the world (booking, payment, sending, deleting,
  writing), otherwise "safe"
- "when": optional {"ref": "sK", "equals": true|false} to run the step only under a condition
- "after": list of ids of EVERY step this one depends on, e.g. ["s1"] - required whenever a
  dependency exists. Include a step here even if you also reference it with $ref in args;
  "after" is the complete, authoritative dependency list, $ref is just how you also use its
  data. Do not leave this empty if the step has any real dependency.
"final": ids of the steps whose outputs answer the request.
List every dependency in "after" - never describe order only in prose or by list position.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRFormat(PlanFormat):
    name = "typed_ir"

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        return plan.to_prompt_json()

    def json_schema(self) -> dict[str, Any]:
        return PlanDraft.model_json_schema()

    async def parse(self, text: str, query: str) -> Plan:
        try:
            draft = PlanDraft.model_validate(parse_json(text))
        except (ValueError, ValidationError) as exc:  # ValueError: not JSON at all
            raise PlanParseError(str(exc)) from exc
        return Plan.from_draft(draft, query)
