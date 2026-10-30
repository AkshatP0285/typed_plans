"""Free-text plan format.

The planner writes numbered natural-language steps. A converter LLM then maps
the text to the IR so it can run on the same executor. Converter tokens are
traced under role ``converter`` and reported separately.
"""
from __future__ import annotations

import json
import re

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
    order = f" after step {', '.join(_step_number(a) for a in step.after)}" if step.after else ""
    condition = ""
    if step.when is not None:
        condition = f" only if step {_step_number(step.when.ref)} is {step.when.equals}"
    return f"{number}. Call {step.tool} with {args or 'no arguments'}{order}{condition}."


RULE_INSTRUCTIONS = """\
Write the plan as plain text, one tool call per line, exactly in this form:
1. Call <tool name> with <arg> = <JSON value>, <arg> = <JSON value>.
2. Call <tool name> with <arg> = the result of step 1.
Write "with no arguments" if a call takes none. Add "after step 1, 2" if a call must wait for
steps whose results it does not use. End with a line "Answer: steps <numbers>".
If no tool is needed, write only: No tool needed."""

_LINE = re.compile(r"^\s*(\d+)[.)]\s*Call\s+(.+?)\s+with\s+(.*?)(?:\s+after steps? ([\d,\s]+?))?"
                   r"(?:\s+only if step (\d+) is (true|false))?\.?\s*$", re.I)
_ARG = re.compile(r"(?:^|,\s*)([A-Za-z_][\w.\-]*)\s*=\s*")
_RESULT = re.compile(r"^the result of step (\d+)(?: \(field ([^)]+)\))?$", re.I)


def _value(text: str):
    text = text.strip()
    m = _RESULT.match(text)
    if m:
        return {"$ref": f"s{m.group(1)}", **({"path": m.group(2)} if m.group(2) else {})}
    try:
        return json.loads(text)
    except ValueError:
        return text  # unquoted string


def _args(text: str) -> dict:
    if text.strip().lower() in ("no arguments", ""):
        return {}
    marks = list(_ARG.finditer(text))
    return {m.group(1): _value(text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)])
            for i, m in enumerate(marks)}


def parse_rule(text: str, query: str) -> Plan:
    """Read the plain-text plan with fixed rules - no LLM - so the text is judged as written."""
    steps, final = [], []
    for line in text.splitlines():
        if line.strip().lower().startswith("answer:"):
            final = [f"s{n}" for n in re.findall(r"\d+", line)]
        elif m := _LINE.match(line):
            when = {"ref": f"s{m.group(5)}", "equals": m.group(6).lower() == "true"} if m.group(5) else None
            after = [f"s{n}" for n in re.findall(r"\d+", m.group(4) or "")]
            steps.append(Step(id=f"s{m.group(1)}", tool=m.group(2), args=_args(m.group(3)), when=when, after=after))
    if not steps and "no tool" not in text.lower():
        raise PlanParseError("no step lines in the form '1. Call <tool> with ...'")
    ids = {s.id for s in steps}
    return Plan(query=query, steps=steps, final=[f for f in final if f in ids] or [s.id for s in steps])


class FreeTextFormat(PlanFormat):
    name = "free_text"

    def __init__(self, converter: LLMClient | None = None):
        # converter=None: parse the text with fixed rules instead of a converter LLM.
        self.converter = converter
        self._ir = TypedIRFormat()  # the converter LLM writes typed IR, which this parses

    def instructions(self) -> str:
        return INSTRUCTIONS if self.converter else RULE_INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        lines = [_describe_step(number, step) for number, step in enumerate(plan.steps, 1)]
        final_numbers = [_step_number(step_id) for step_id in plan.final]
        lines.append("Answer: steps " + ", ".join(final_numbers))
        return "\n".join(lines)

    async def parse(self, text: str, query: str) -> Plan:
        if self.converter is None:
            return parse_rule(text, query)
        # Second LLM call: translate the English plan into typed IR (constrained to the IR JSON schema).
        prompt = CONVERT_PROMPT.format(rules=IR_INSTRUCTIONS, plan=text)
        reply = await self.converter.chat(
            [{"role": "user", "content": prompt}], role="converter", json_schema=self._ir.json_schema()
        )
        try:
            return await self._ir.parse(reply.text, query)
        except PlanParseError as exc:
            raise PlanParseError(f"free-text conversion failed: {exc}") from exc
