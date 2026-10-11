from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from tpj.ir.formats.base import PlanFormat, PlanParseError
from tpj.ir.schema import Plan, PlanDraft
from tpj.llm.client import parse_json

INSTRUCTIONS = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
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
"final": ids of the steps whose outputs answer the request.
Dependencies are expressed only through references; do not describe execution order.
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
        except (ValueError, ValidationError) as e:
            raise PlanParseError(str(e)) from e
        return Plan.from_draft(draft, query)
