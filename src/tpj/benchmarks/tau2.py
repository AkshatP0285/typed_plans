"""tau2-bench integration: our pipeline as a tau2 half-duplex agent.

tau2 owns the conversation loop, the user simulator, tool execution and scoring.
For every user turn we plan, then emit one wave per assistant message as
parallel tool calls; tool results come back as ToolMessage / MultiToolMessage
and go through the same gate -> judge -> local repair path. When the plan is
finished, a responder LLM writes the reply to the user.

One user turn, as tau2 sees it::

    UserMessage        -> _start(): plan the turn          -> AssistantMessage(tool_calls=wave 1)
    MultiToolMessage   -> _observe(): feed results to plan -> AssistantMessage(tool_calls=wave 2)
    ...
    ToolMessage        -> _observe(); plan is done         -> AssistantMessage(content=reply to user)

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

# tau2 calls our agent synchronously, but our pipeline is async. We run all coroutines on
# one background event loop shared by all simulations, so async clients stay on one loop.
_LOOP = asyncio.new_event_loop()
threading.Thread(target=_LOOP.run_forever, daemon=True).start()

# Cap on the tool-results JSON pasted into the responder prompt (characters).
_MAX_RESULTS_CHARS = 12000


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
    """tau2 tool (exposes an OpenAI-style schema) -> our ToolSpec."""
    function = tool.openai_schema["function"]
    return ToolSpec(name=function["name"], description=function.get("description", ""),
                    parameters=function.get("parameters", {}))


def _render(messages: list[Any]) -> str:
    """Plain-text transcript of the user/agent text messages (tool calls and results are left out)."""
    lines = []
    for message in messages:
        if isinstance(message, UserMessage) and message.content:
            lines.append(f"User: {message.content}")
        elif isinstance(message, AssistantMessage) and message.content:
            lines.append(f"Agent: {message.content}")
    return "\n".join(lines)


def _parse_tool_content(content: Any) -> Any:
    """tau2 tool results are usually JSON strings; fall back to the raw content if not."""
    try:
        return json.loads(content) if content else None
    except (json.JSONDecodeError, TypeError):
        return content


class TypedPlanTau2Agent(HalfDuplexAgent[Tau2State]):
    def __init__(self, tools: list[Any], domain_policy: str, agent: Any, responder: LLMClient,
                 trace_path: str | None = None):
        super().__init__(tools, domain_policy)
        self.tools = tools
        self.domain_policy = domain_policy
        self.specs = [_spec(tool) for tool in tools]
        self.agent = agent
        self.responder = responder
        self.trace_path = trace_path
        self.tracer = Tracer(task_id=f"tau2-{uuid.uuid4().hex[:8]}")
        self._plan_run: PlanRun | None = None  # the plan for the current user turn (None = just reply)
        self._wave: list[PendingCall] = []  # calls sent in the last assistant message
        self._step_of_call: dict[str, str] = {}  # tau2 tool_call id -> our step id

    def get_init_state(self, message_history: list[Any] | None = None) -> Tau2State:
        return Tau2State(system_messages=[SystemMessage(role="system", content=self.domain_policy)],
                         messages=list(message_history or []))

    def generate_next_message(self, message: Any, state: Tau2State) -> tuple[Any, Tau2State]:
        """tau2 entry point: take the incoming message, return our next assistant message."""
        state.messages.append(message)
        with activate(self.tracer):  # for synchronous emits
            if isinstance(message, (ToolMessage, MultiToolMessage)) and self._plan_run is not None:
                self._observe(message)
            elif isinstance(message, UserMessage):
                self._start(state)
            reply = self._next(state)
        if self.trace_path:
            # Append this message's events to the trace file and clear the in-memory buffer.
            write_jsonl(self.trace_path, self.tracer.events)
            self.tracer.events.clear()
        state.messages.append(reply)
        return reply, state

    # ---------------------------------------------------------------
    def _start(self, state: Tau2State) -> None:
        """New user message: build a fresh plan for this turn from the policy and conversation so far."""
        query = f"Policy:\n{self.domain_policy}\n\nConversation so far:\n{_render(state.messages)}\n\n" \
                "Plan the tool calls needed to handle the latest user message (none if you only need to reply)."
        task = Task(id=self.tracer.task_id, query=query, tools=self.specs)
        self._plan_run = PlanRun(self.agent, task)
        try:
            _run(self._plan_run.prepare(), self.tracer)
        except (PlanParseError, PlanStructureError) as e:
            # Unusable plan: drop it and just reply to the user without tool calls.
            self.tracer.emit("plan_error", error=str(e))
            self._plan_run = None

    def _observe(self, message: Any) -> None:
        """Tool results came back: map them to our steps and let the PlanRun gate / judge / repair them."""
        tool_messages = message.tool_messages if isinstance(message, MultiToolMessage) else [message]
        outcomes: dict[str, Outcome] = {}
        for tool_message in tool_messages:
            step_id = self._step_of_call.get(tool_message.id)
            if step_id is None:  # not one of the calls we just sent
                continue
            content = _parse_tool_content(tool_message.content)
            if getattr(tool_message, "error", False):
                outcomes[step_id] = Outcome(error=str(content), output=None)
            else:
                outcomes[step_id] = Outcome(error=None, output=content)
        answered = [call for call in self._wave if call.step.id in outcomes]
        _run(self._plan_run.observe(answered, outcomes), self.tracer)

    def _next(self, state: Tau2State) -> Any:
        """Send the next wave of tool calls if the plan has one, otherwise reply to the user."""
        if self._plan_run is not None and not self._plan_run.done:
            wave = self._plan_run.next_wave()
            if wave:
                return self._tool_call_message(wave)
        return self._reply_to_user(state)

    def _tool_call_message(self, wave: list[PendingCall]) -> Any:
        """One assistant message carrying the whole wave as parallel tool calls."""
        self._wave = wave
        self._step_of_call = {}
        calls = []
        for pending in wave:
            call_id = f"call_{uuid.uuid4().hex[:12]}"
            self._step_of_call[call_id] = pending.step.id
            calls.append(ToolCall(id=call_id, name=pending.step.tool, arguments=pending.args,
                                  requestor="assistant"))
        return AssistantMessage(role="assistant", content=None, tool_calls=calls)

    def _reply_to_user(self, state: Tau2State) -> Any:
        """Plan finished (or none): let the responder LLM write a text reply from this turn's results."""
        results = self._plan_run.state.summary() if self._plan_run is not None else {}
        self._plan_run = None  # the turn is over; the next user message starts a new plan
        results_text = json.dumps(results, default=str)[:_MAX_RESULTS_CHARS] or "(none)"
        prompt = RESPOND_PROMPT.format(policy=self.domain_policy, conversation=_render(state.messages),
                                       results=results_text)
        response = _run(self.responder.chat([{"role": "user", "content": prompt}], role="responder"), self.tracer)
        return AssistantMessage(role="assistant", content=response.text)


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
