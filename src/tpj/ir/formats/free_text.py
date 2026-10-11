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
from tpj.ir.schema import Plan, Step
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


def _step_number(step_id: str) -> str:
    """'s3' -> '3'. Step ids are assumed to be ``s<N>``."""
    return step_id[1:]


def _describe(value) -> str:
    """English description of an argument value, e.g. ``the result of step 1 (field first.id)``."""
    if is_ref(value):
        description = f"the result of step {_step_number(value['$ref'])}"
        if value.get("path"):
            description += f" (field {value['path']})"
        return description
    if isinstance(value, list):
        return "[" + ", ".join(_describe(item) for item in value) + "]"
    return json.dumps(value)


def _describe_step(number: int, step: Step) -> str:
    """One numbered line, e.g. ``2. Call book with flight = the result of step 1 only if step 1 is True.``"""
    args = ", ".join(f"{name} = {_describe(value)}" for name, value in step.args.items())
    condition = ""
    if step.when is not None:
        condition = f" only if step {_step_number(step.when.ref)} is {step.when.equals}"
    return f"{number}. Call {step.tool} with {args or 'no arguments'}{condition}."


class FreeTextFormat(PlanFormat):
    name = "free_text"

    def __init__(self, converter: LLMClient):
        self.converter = converter
        self._ir = TypedIRFormat()  # the converter LLM writes typed IR, which this parses

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        lines = [_describe_step(number, step) for number, step in enumerate(plan.steps, 1)]
        final_numbers = [_step_number(step_id) for step_id in plan.final]
        lines.append("Answer: steps " + ", ".join(final_numbers))
        return "\n".join(lines)

    async def parse(self, text: str, query: str) -> Plan:
        # Second LLM call: translate the English plan into typed IR (constrained to the IR JSON schema).
        prompt = CONVERT_PROMPT.format(rules=IR_INSTRUCTIONS, plan=text)
        reply = await self.converter.chat(
            [{"role": "user", "content": prompt}], role="converter", json_schema=self._ir.json_schema()
        )
        try:
            return await self._ir.parse(reply.text, query)
        except PlanParseError as exc:
            raise PlanParseError(f"free-text conversion failed: {exc}") from exc
