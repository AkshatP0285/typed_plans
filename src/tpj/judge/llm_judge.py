"""LLM components that intervene only when the gate flags something.

- :class:`LLMIREditor`: pre-execution, rewrites a flagged plan (nothing has run yet).
- :class:`LLMJudge`: post-execution, diagnoses flagged nodes and returns local patch ops.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from pydantic import ValidationError

from tpj.benchmarks.base import Task
from tpj.eval.trace import emit
from tpj.ir.formats.typed_ir import INSTRUCTIONS as IR_RULES
from tpj.ir.formats.typed_ir import TypedIRFormat
from tpj.ir.formats.base import PlanParseError
from tpj.ir.schema import Plan
from tpj.llm.client import LLMClient, parse_json
from tpj.repair.patch import JudgeDecision


@dataclass
class Flag:
    """A step the gate flagged (or whose tool raised), passed to the judge."""

    node_id: str
    reason: str  # gate label or exception message
    args: Any = None
    output: Any = None


def _clip(v: Any, n: int = 1500) -> str:
    s = v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
    return s if len(s) <= n else s[:n] + " ...[truncated]"


def _tools_json(task: Task) -> str:
    return json.dumps([t.to_prompt() for t in task.tools], indent=1)


EDITOR_PROMPT = """\
A verifier flagged the following plan as possibly wrong before execution.
Verifier verdict: {verdict}
{extra}
Fix the plan so it fulfils the request. Keep correct steps unchanged.

{rules}

Tools:
{tools}

Request: {query}

Current plan:
{plan}

Reply with the full corrected plan as {reply}."""


class LLMIREditor:
    """Pre-execution: rewrites the whole plan when the gate flags it (nothing has run yet)."""

    enabled = True

    def __init__(self, llm: LLMClient, format=None, preserve_steps: bool = False):
        self.llm = llm
        # Edit in the planner's own format (rules, schema, parser) so edits follow the plan's conventions.
        self._fmt = format if format is not None else TypedIRFormat()
        self._rules = self._fmt.instructions()
        # Opt-in (off for BFCL, where deleting spurious calls is the right fix for irrelevance):
        # for benchmarks where every request needs a tool, never let an edit wipe out the plan.
        self.preserve_steps = preserve_steps

    async def revise(self, task: Task, plan: Plan, verdict: str, extra: str = "") -> Plan:
        if self.preserve_steps:
            extra = (extra + "\nSome tools list no parameters; positional arguments such as \"arg0\" are "
                     "valid for them - keep them. Fix only the specific problem; do not delete steps that "
                     "perform an action the request asks for.").strip()
        prompt = EDITOR_PROMPT.format(
            verdict=verdict, extra=extra, rules=self._rules,
            reply="JSON" if self._fmt.json_schema() else "plain text in the format above", tools=_tools_json(task),
            query=task.query, plan=self._fmt.render(plan),
        )
        res = await self.llm.chat([{"role": "user", "content": prompt}], role="ir_editor",
                                  json_schema=self._fmt.json_schema(), schema_name="plan")
        try:
            new = await self._fmt.parse(res.text, task.query)
        except PlanParseError as e:
            emit("ir_edit_failed", error=str(e))
            return plan
        if self.preserve_steps and plan.steps and not new.steps:
            emit("ir_edit_rejected", reason="edit removed every step")
            return plan
        emit("plan", source="ir_editor", plan=new.model_dump())
        return new


JUDGE_PROMPT = """\
You supervise a tool-calling agent that executes a typed plan in waves. A lightweight
verifier flagged some executed steps. Decide what is wrong and fix only the affected part
of the plan. Steps that already ran correctly are cached and will not be re-run unless
you change them or something they depend on.

Plan rules:
{rules}

Tools:
{tools}

Request: {query}

Current plan:
{plan}

Execution so far (step: status -> output):
{status}

Flagged steps:
{flags}

Reply as JSON with:
- "verdict": "repair" (apply ops), "retry" (transient failure; re-run unchanged),
  "false_alarm" (the flagged outputs are actually fine) or "abort" (cannot be done)
- "diagnosis": one or two sentences
- "ops": list of edits, each one of
  {{"op": "edit", "id": "sK", ...fields to change (tool/args/returns/safety/when)}}
  {{"op": "replace", "id": "sK", "step": {{full step}}}}
  {{"op": "remove", "id": "sK"}}
  {{"op": "add", "step": {{full step with a new id}}}}
- "final": new final step ids, only if they change"""


class LLMJudge:
    """Post-execution: diagnoses flagged steps and returns a JudgeDecision with patch ops."""

    enabled = True

    def __init__(self, llm: LLMClient, max_output_chars: int = 1500, format=None):
        self.llm = llm
        self.max_output_chars = max_output_chars
        # The judge sees the plan as IR JSON, so non-IR formats get the IR rules.
        self._rules = format.instructions() if isinstance(format, TypedIRFormat) else IR_RULES

    async def review(self, task: Task, plan: Plan, flags: list[Flag], status: dict[str, Any]) -> JudgeDecision:
        status_lines = []
        for step_id, node in status.items():
            shown = node.get("output") if node.get("error") is None else node["error"]
            status_lines.append(f"{step_id}: {node['status']} -> {_clip(shown, 400)}")
        status_txt = "\n".join(status_lines)

        flag_lines = []
        for flag in flags:
            flag_lines.append(f"{flag.node_id}: {flag.reason}\n  args: {_clip(flag.args, 600)}"
                              f"\n  output: {_clip(flag.output, self.max_output_chars)}")
        flags_txt = "\n".join(flag_lines)

        prompt = JUDGE_PROMPT.format(
            rules=self._rules, tools=_tools_json(task), query=task.query,
            plan=plan.to_prompt_json(), status=status_txt or "(nothing yet)", flags=flags_txt,
        )
        res = await self.llm.chat([{"role": "user", "content": prompt}], role="judge",
                                  json_schema=JudgeDecision.model_json_schema(), schema_name="decision")
        try:
            decision = JudgeDecision.model_validate(parse_json(res.text))
        except (ValueError, ValidationError) as e:
            # An unreadable reply is treated as "retry" so the run can continue.
            emit("judge_parse_error", error=str(e), text=res.text[:2000])
            decision = JudgeDecision(verdict="retry", diagnosis=f"unparseable judge output: {e}")
        emit("judge", flagged=[f.node_id for f in flags], verdict=decision.verdict,
             diagnosis=decision.diagnosis, ops=[op.model_dump() for op in decision.ops])
        return decision


class NullEditor:
    """No pre-execution editing (ablation)."""

    enabled = False

    async def revise(self, task: Task, plan: Plan, verdict: str, extra: str = "") -> Plan:
        return plan


class NullJudge:
    """No post-execution intervention (ablation / direct baseline)."""

    enabled = False

    async def review(self, task, plan, flags, status) -> JudgeDecision:
        return JudgeDecision(verdict="false_alarm", diagnosis="judge disabled")
