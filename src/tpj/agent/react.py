"""ReAct-style baseline: native tool calling, no plan, no verification."""
from __future__ import annotations

import json

from tpj.agent.base import Agent
from tpj.benchmarks.base import AgentResult, CallRecord, Task
from tpj.eval.trace import emit
from tpj.faults.injector import RAISING, FaultInjector
from tpj.llm.client import LLMClient
from tpj.tools.registry import ToolRegistry

SYSTEM = ("You are a tool-calling assistant. Use the tools to complete the user's request. "
          "Call tools until the request is done, then give a short final answer.")


class ReActAgent(Agent):
    def __init__(self, llm: LLMClient, faults: FaultInjector, max_turns: int = 15):
        self.llm = llm
        self.faults = faults
        self.max_turns = max_turns

    async def run(self, task: Task, registry: ToolRegistry) -> AgentResult:
        msgs: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.query}]
        tools = [t.to_openai() for t in task.tools]
        calls: list[CallRecord] = []
        answer = None
        n = 0
        for _ in range(self.max_turns):
            res = await self.llm.chat(msgs, role="react", tools=tools)
            if not res.tool_calls:
                answer = res.text
                break
            msgs.append({
                "role": "assistant", "content": res.text or None,
                "tool_calls": [{"id": tc["id"], "type": "function",
                                "function": {"name": tc["name"], "arguments": tc["arguments"]}}
                               for tc in res.tool_calls],
            })
            for tc in res.tool_calls:
                n += 1
                sid = f"c{n}"
                try:
                    args = json.loads(tc["arguments"] or "{}")
                except json.JSONDecodeError:
                    args = {}
                fault = self.faults.exec_fault(task.id, sid, 0)
                if fault:
                    emit("fault", level="exec", fault_type=fault, step=sid)
                try:
                    if fault in RAISING:
                        self.faults.apply_exec(fault, None, task.id, sid)
                    out = await registry.call(tc["name"], args)
                    if fault:
                        out = self.faults.apply_exec(fault, out, task.id, sid)
                    calls.append(CallRecord(sid, tc["name"], args, out))
                    content = json.dumps(out, default=str)
                except Exception as e:
                    calls.append(CallRecord(sid, tc["name"], args, error=str(e)))
                    content = f"ERROR: {e}"
                emit("node_result", node_id=sid, tool=tc["name"], status="ok" if calls[-1].error is None else "failed",
                     fault=fault, gated=False, gate_label=None, flagged=False, attempt=1)
                msgs.append({"role": "tool", "tool_call_id": tc["id"], "content": content[:8000]})
        ok_calls = [c for c in calls if c.error is None]
        final = {"last": ok_calls[-1].output} if ok_calls else {}
        return AgentResult(task.id, "completed" if answer is not None else "failed", final=final,
                           calls=calls, answer_text=answer)
