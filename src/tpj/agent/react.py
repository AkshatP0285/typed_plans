"""ReAct-style baseline: native tool calling, no plan, no verification.

Loop: ask the model; if it requests tool calls, run them (with any injected
execution faults) and feed the results back as ``tool`` messages; stop when the
model answers without calling a tool or after ``max_turns`` model calls.
"""
from __future__ import annotations

import json
from typing import Any

from tpj.agent.base import Agent
from tpj.benchmarks.base import AgentResult, CallRecord, Task
from tpj.eval.trace import emit
from tpj.faults.injector import RAISING, FaultInjector
from tpj.llm.client import LLMClient
from tpj.tools.registry import ToolRegistry

SYSTEM = ("You are a tool-calling assistant. Use the tools to complete the user's request. "
          "Call tools until the request is done, then give a short final answer.")


def _assistant_message(reply) -> dict[str, Any]:
    """Echo the model's tool calls back in OpenAI format; the following ``tool`` messages refer to their ids."""
    tool_calls = []
    for tool_call in reply.tool_calls:
        tool_calls.append({
            "id": tool_call["id"],
            "type": "function",
            "function": {"name": tool_call["name"], "arguments": tool_call["arguments"]},
        })
    return {"role": "assistant", "content": reply.text or None, "tool_calls": tool_calls}


def _parse_arguments(raw: str | None) -> Any:
    """Tool-call arguments arrive as a JSON string; malformed JSON becomes ``{}`` (the call then usually fails)."""
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        return {}


class ReActAgent(Agent):
    def __init__(self, llm: LLMClient, faults: FaultInjector, max_turns: int = 15):
        self.llm = llm
        self.faults = faults
        self.max_turns = max_turns

    async def run(self, task: Task, registry: ToolRegistry) -> AgentResult:
        messages: list[dict] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": task.query}]
        tools = [tool.to_openai() for tool in task.tools]
        calls: list[CallRecord] = []
        answer = None
        call_count = 0
        for _ in range(self.max_turns):
            reply = await self.llm.chat(messages, role="react", tools=tools)
            if not reply.tool_calls:  # the model answered in text: done
                answer = reply.text
                break
            messages.append(_assistant_message(reply))
            for tool_call in reply.tool_calls:
                call_count += 1
                call_id = f"c{call_count}"  # stands in for a plan step id in traces and fault seeding
                content = await self._execute(task, registry, tool_call, call_id, calls)
                messages.append({"role": "tool", "tool_call_id": tool_call["id"], "content": content[:8000]})

        # ReAct has no "final" steps, so the output of the last successful call is taken as the result.
        ok_calls = [call for call in calls if call.error is None]
        final = {"last": ok_calls[-1].output} if ok_calls else {}
        status = "completed" if answer is not None else "failed"  # failed = ran out of turns
        return AgentResult(task.id, status, final=final, calls=calls, answer_text=answer)

    async def _execute(self, task: Task, registry: ToolRegistry, tool_call: dict[str, Any], call_id: str,
                       calls: list[CallRecord]) -> str:
        """Run one tool call, append its :class:`CallRecord` to ``calls``, and return the text shown to the model."""
        tool_name = tool_call["name"]
        args = _parse_arguments(tool_call["arguments"])
        fault = self.faults.exec_fault(task.id, call_id, 0)
        if fault:
            emit("fault", level="exec", fault_type=fault, step=call_id)
        try:
            if fault in RAISING:
                # exception/timeout faults raise here, before the tool is called
                self.faults.apply_exec(fault, None, task.id, call_id)
            output = await registry.call(tool_name, args)
            if fault:
                output = self.faults.apply_exec(fault, output, task.id, call_id)
            calls.append(CallRecord(call_id, tool_name, args, output))
            content = json.dumps(output, default=str)
        except Exception as exc:
            calls.append(CallRecord(call_id, tool_name, args, error=str(exc)))
            content = f"ERROR: {exc}"
        emit("node_result", node_id=call_id, tool=tool_name, status="ok" if calls[-1].error is None else "failed",
             fault=fault, gated=False, gate_label=None, flagged=False, attempt=1)
        return content
