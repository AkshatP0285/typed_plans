"""tau2-bench integration: our pipeline as a tau2 half-duplex agent.

tau2 owns the conversation loop, the user simulator, tool execution and scoring.
For every user turn we plan, then emit one wave per assistant message as
parallel tool calls; tool results come back as ToolMessage / MultiToolMessage
and go through the same gate -> judge -> local repair path. When the plan is
finished, a responder LLM writes the reply to the user.

Interfaces follow ``src/tau2/agent/README.md`` (HalfDuplexAgent,
get_init_state, generate_next_message, registry.register_agent_factory).
Re-check them against the tau2 version you install.
"""
from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field

from tpj.agent.run import Outcome, PendingCall, PlanRun
from tpj.benchmarks.base import Task
from tpj.eval.trace import Tracer, activate, write_jsonl
from tpj.ir.dag import PlanStructureError
from tpj.ir.formats.base import PlanParseError
from tpj.llm.client import LLMClient
from tpj.tools.spec import ToolSpec

try:
    from tau2.agent.base_agent import HalfDuplexAgent
    from tau2.data_model.message import (
        AssistantMessage,
        MultiToolMessage,
        SystemMessage,
        ToolCall,
        ToolMessage,
        UserMessage,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError("tau2-bench is not installed: pip install from github.com/sierra-research/tau2-bench") from e

# One background event loop shared by all simulations, so async clients stay on one loop.
_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True).start()


def _run(coro, tracer: Tracer):
    """Run on the shared loop; the loop thread does not inherit our context, so re-activate the tracer."""
    async def wrapped():
        with activate(tracer):
            return await coro

    return asyncio.run_coroutine_threadsafe(wrapped(), _LOOP).result()


class Tau2State(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    system_messages: list[Any] = Field(default_factory=list)
    messages: list[Any] = Field(default_factory=list)


RESPOND_PROMPT = """\
You are a customer-service agent. Follow the policy. Using the conversation and the results of
the tool calls you just made, write your next message to the user. Be concise. If you need more
information from the user, ask for it.

Policy:
{policy}

Conversation:
{conversation}

Tool results this turn:
{results}"""


def _spec(tool: Any) -> ToolSpec:
    fn = tool.openai_schema["function"]
    return ToolSpec(name=fn["name"], description=fn.get("description", ""), parameters=fn.get("parameters", {}))


def _render(messages: list[Any]) -> str:
    lines = []
    for m in messages:
        if isinstance(m, UserMessage) and m.content:
            lines.append(f"User: {m.content}")
        elif isinstance(m, AssistantMessage) and m.content:
            lines.append(f"Agent: {m.content}")
    return "\n".join(lines)


class TypedPlanTau2Agent(HalfDuplexAgent[Tau2State]):
    def __init__(self, tools: list[Any], domain_policy: str, agent: Any, responder: LLMClient,
                 trace_path: str | None = None):
        super().__init__(tools, domain_policy)
        self.tools = tools
        self.domain_policy = domain_policy
        self.specs = [_spec(t) for t in tools]
        self.agent = agent
        self.responder = responder
        self.trace_path = trace_path
        self.tracer = Tracer(task_id=f"tau2-{uuid.uuid4().hex[:8]}")
        self._run_: PlanRun | None = None
        self._wave: list[PendingCall] = []
        self._ids: dict[str, str] = {}  # tool_call id -> step id

    def get_init_state(self, message_history: list[Any] | None = None) -> Tau2State:
        return Tau2State(system_messages=[SystemMessage(role="system", content=self.domain_policy)],
                         messages=list(message_history or []))

    def generate_next_message(self, message: Any, state: Tau2State) -> tuple[Any, Tau2State]:
        state.messages.append(message)
        with activate(self.tracer):  # for synchronous emits
            if isinstance(message, (ToolMessage, MultiToolMessage)) and self._run_ is not None:
                self._observe(message)
            elif isinstance(message, UserMessage):
                self._start(state)
            reply = self._next(state)
        if self.trace_path:
            write_jsonl(self.trace_path, self.tracer.events)
            self.tracer.events.clear()
        state.messages.append(reply)
        return reply, state

    # ---------------------------------------------------------------
    def _start(self, state: Tau2State) -> None:
        query = f"Policy:\n{self.domain_policy}\n\nConversation so far:\n{_render(state.messages)}\n\n" \
                "Plan the tool calls needed to handle the latest user message (none if you only need to reply)."
        task = Task(id=self.tracer.task_id, query=query, tools=self.specs)
        self._run_ = PlanRun(self.agent, task)
        try:
            _run(self._run_.prepare(), self.tracer)
        except (PlanParseError, PlanStructureError) as e:
            self.tracer.emit("plan_error", error=str(e))
            self._run_ = None

    def _observe(self, message: Any) -> None:
        msgs = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        outcomes: dict[str, Outcome] = {}
        for m in msgs:
            sid = self._ids.get(m.id)
            if sid is None:
                continue
            try:
                out = json.loads(m.content) if m.content else None
            except (json.JSONDecodeError, TypeError):
                out = m.content
            outcomes[sid] = Outcome(error=str(out) if getattr(m, "error", False) else None,
                                    output=None if getattr(m, "error", False) else out)
        wave = [c for c in self._wave if c.step.id in outcomes]
        _run(self._run_.observe(wave, outcomes), self.tracer)

    def _next(self, state: Tau2State) -> Any:
        if self._run_ is not None and not self._run_.done:
            wave = self._run_.next_wave()
            if wave:
                self._wave, self._ids = wave, {}
                calls = []
                for c in wave:
                    cid = f"call_{uuid.uuid4().hex[:12]}"
                    self._ids[cid] = c.step.id
                    calls.append(ToolCall(id=cid, name=c.step.tool, arguments=c.args, requestor="assistant"))
                return AssistantMessage(role="assistant", content=None, tool_calls=calls)
        results = self._run_.state.summary() if self._run_ is not None else {}
        self._run_ = None
        prompt = RESPOND_PROMPT.format(policy=self.domain_policy, conversation=_render(state.messages),
                                       results=json.dumps(results, default=str)[:12000] or "(none)")
        res = _run(self.responder.chat([{"role": "user", "content": prompt}], role="responder"), self.tracer)
        return AssistantMessage(role="assistant", content=res.text)


def register(build_agent: Callable[[], Any], responder: LLMClient, name: str = "tpj_agent",
             trace_path: str | None = None) -> None:
    """Register our agent with tau2's registry under ``name``."""
    from tau2.registry import registry

    agent = build_agent()  # shared across simulations (one Laya load, one client pool)

    def create_agent(tools, domain_policy, **kwargs):
        return TypedPlanTau2Agent(tools, domain_policy, agent, responder, trace_path)

    registry.register_agent_factory(create_agent, name)
    if trace_path:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
