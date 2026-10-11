"""LLM planner: query + tools -> plan in the configured format.

The same class serves the prompted baseline and the fine-tuned (LoRA) planner;
only the ``llm`` config differs. :meth:`build_messages` is also used to build
fine-tuning examples so training and inference prompts stay identical.
"""
from __future__ import annotations

import json

from tpj.benchmarks.base import Task
from tpj.eval.trace import emit
from tpj.ir.examples import GOA_PLAN, GOA_QUERY
from tpj.ir.formats.base import PlanFormat, PlanParseError
from tpj.ir.schema import Plan
from tpj.llm.client import LLMClient
from tpj.planner.base import Planner
from tpj.tools.spec import ToolSpec

SYSTEM = """\
You are a planner for a tool-calling agent. Given a user request and the available tools,
write a complete plan of tool calls that fulfils the request. Use only listed tools and
their exact parameter names.

{format_rules}"""


class PromptedPlanner(Planner):
    def __init__(
        self,
        llm: LLMClient,
        format: PlanFormat,
        few_shot: bool = True,
        constrained: bool = True,
        max_parse_retries: int = 1,
    ):
        self.llm = llm
        self.format = format
        self.few_shot = few_shot
        self.constrained = constrained
        self.max_parse_retries = max_parse_retries

    def build_messages(self, query: str, tools: list[ToolSpec], feedback: str | None = None) -> list[dict]:
        msgs = [{"role": "system", "content": SYSTEM.format(format_rules=self.format.instructions())}]
        if self.few_shot:
            msgs.append({"role": "user", "content": f"Request: {GOA_QUERY}\n\n(tools omitted in this example)"})
            msgs.append({"role": "assistant", "content": self.format.render(GOA_PLAN)})
        tool_json = json.dumps([t.to_prompt() for t in tools], indent=1)
        content = f"Tools:\n{tool_json}\n\nRequest: {query}"
        if feedback:
            content += f"\n\nA previous plan for this request failed:\n{feedback}\nWrite a corrected plan."
        msgs.append({"role": "user", "content": content})
        return msgs

    async def plan(self, task: Task, feedback: str | None = None) -> Plan:
        msgs = self.build_messages(task.query, task.tools, feedback)
        schema = self.format.json_schema() if self.constrained else None
        last_err = None
        for attempt in range(self.max_parse_retries + 1):
            res = await self.llm.chat(msgs, role="planner", json_schema=schema, schema_name="plan")
            try:
                plan = await self.format.parse(res.text, task.query)
                emit("plan", source="planner", attempt=attempt, plan=plan.model_dump())
                return plan
            except PlanParseError as e:
                last_err = e
                emit("plan_parse_error", attempt=attempt, error=str(e), text=res.text[:2000])
                msgs = msgs + [
                    {"role": "assistant", "content": res.text},
                    {"role": "user", "content": f"That output could not be parsed: {e}. Reply with the plan only."},
                ]
        raise PlanParseError(str(last_err))
