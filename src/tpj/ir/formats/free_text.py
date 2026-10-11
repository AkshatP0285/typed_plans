"""Free-text plan format.

The planner writes numbered natural-language steps. A converter LLM then maps
the text to the IR so it can run on the same executor. Converter tokens are
traced under role ``converter`` and reported separately.
"""
from __future__ import annotations

import json

from tpj.ir.formats.base import PlanFormat, PlanParseError
from tpj.ir.formats.typed_ir import INSTRUCTIONS as IR_INSTRUCTIONS
from tpj.ir.formats.typed_ir import TypedIRFormat
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan
from tpj.llm.client import LLMClient

INSTRUCTIONS = """\
Write the plan as a numbered list of steps in plain English, one tool call per line.
Name the tool and its arguments, and say which earlier step's result is used where.
End with a line "Answer: steps <numbers>" naming the steps that answer the request."""

CONVERT_PROMPT = """\
Convert the numbered plan below into JSON, exactly following the format rules.
Do not add, drop or reorder tool calls; translate faithfully.

{rules}

Plan:
{plan}"""


def _describe(value) -> str:
    if is_ref(value):
        step = value["$ref"][1:]
        return f"the result of step {step}" + (f" (field {value['path']})" if value.get("path") else "")
    if isinstance(value, list):
        return "[" + ", ".join(_describe(v) for v in value) + "]"
    return json.dumps(value)


class FreeTextFormat(PlanFormat):
    name = "free_text"

    def __init__(self, converter: LLMClient):
        self.converter = converter
        self._ir = TypedIRFormat()

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        lines = []
        for i, s in enumerate(plan.steps, 1):
            args = ", ".join(f"{k} = {_describe(v)}" for k, v in s.args.items())
            cond = ""
            if s.when is not None:
                cond = f" only if step {s.when.ref[1:]} is {s.when.equals}"
            lines.append(f"{i}. Call {s.tool} with {args or 'no arguments'}{cond}.")
        lines.append("Answer: steps " + ", ".join(f[1:] for f in plan.final))
        return "\n".join(lines)

    async def parse(self, text: str, query: str) -> Plan:
        prompt = CONVERT_PROMPT.format(rules=IR_INSTRUCTIONS, plan=text)
        res = await self.converter.chat(
            [{"role": "user", "content": prompt}], role="converter", json_schema=self._ir.json_schema()
        )
        try:
            return await self._ir.parse(res.text, query)
        except PlanParseError as e:
            raise PlanParseError(f"free-text conversion failed: {e}") from e
