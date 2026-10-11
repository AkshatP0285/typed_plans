"""One task's execution-aware run (Figure 1), written as a step-wise state machine.

The run decides *what* to execute next (``next_wave``) and reacts to outcomes
(``observe``); the caller decides *how* tools are executed. The local agent
executes against a tool registry, while the tau2 adapter turns each wave into
one assistant message with parallel tool calls.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tpj.benchmarks.base import AgentResult, CallRecord, Task
from tpj.cache.node_cache import cache_key
from tpj.eval.trace import emit
from tpj.executor.state import ExecutionState
from tpj.gate.base import GateInput
from tpj.ir.dag import PlanStructureError, check_structure
from tpj.ir.refs import RefResolutionError, resolve
from tpj.ir.schema import Plan, Step
from tpj.judge.llm_judge import Flag
from tpj.repair.strategies import AbortTask

if TYPE_CHECKING:
    from tpj.agent.verified import VerifiedAgent


@dataclass
class PendingCall:
    step: Step
    args: dict[str, Any]
    key: str


@dataclass
class Outcome:
    output: Any = None
    error: str | None = None
    fault: str | None = None  # injected fault label, if any


class PlanRun:
    def __init__(self, agent: "VerifiedAgent", task: Task):
        self.a = agent
        self.task = task
        self.plan: Plan | None = None
        self.initial_plan: Plan | None = None
        self.state = ExecutionState()
        self.cache = agent.cache.fresh()
        self.calls: list[CallRecord] = []
        self.repairs = 0
        self.waves = 0
        self.aborted: str | None = None

    # ------------------------------------------------------------ planning
    async def prepare(self) -> Plan:
        """Plan, inject plan faults, then gate + edit before execution."""
        a, task = self.a, self.task
        plan = await a.planner.plan(task)
        plan, plan_faults = a.faults.corrupt_plan(plan, task.tools, task.id)
        self.initial_plan = plan.model_copy(deep=True)

        for round_ in range(a.max_plan_edits + 1):
            structure_err = None
            try:
                check_structure(plan)
            except PlanStructureError as e:
                structure_err = str(e)
                emit("plan_structure_error", round=round_, error=structure_err)
            flagged, verdict = False, None
            if a.gate.enabled("plan"):
                out = await a.gate.classify(GateInput(
                    phase="plan", query=task.query, plan=plan, tools=task.tools,
                    meta={"fault": plan_faults if round_ == 0 else []},
                ))
                flagged, verdict = out.flagged, out.label
            if not (flagged or structure_err) or not a.ir_editor.enabled or round_ == a.max_plan_edits:
                break
            plan = await a.ir_editor.revise(
                task, plan, verdict=verdict or "STRUCTURE_ERROR",
                extra=f"Structural problem: {structure_err}" if structure_err else "",
            )
            emit("plan_edit", round=round_)

        check_structure(plan)  # unexecutable plans end the task here
        self.plan = plan
        self.state = ExecutionState.for_plan(plan)
        emit("plan_final", plan=plan.model_dump())
        return plan

    # ------------------------------------------------------------ scheduling
    @property
    def done(self) -> bool:
        return self.aborted is not None or self.waves >= self.a.max_waves or not self._ready()

    def _ready(self) -> list[Step]:
        assert self.plan is not None
        nodes = self.state.nodes
        return [s for s in self.plan.steps
                if nodes[s.id].status == "pending" and all(nodes[d].status != "pending" for d in s.deps())]

    def next_wave(self) -> list[PendingCall]:
        """Resolve ready steps (conditions, upstream failures, cache hits) and return the next wave."""
        if self.aborted is not None or self.waves >= self.a.max_waves:
            return []
        while True:
            ready = self._ready()
            if not ready:
                return []
            calls: list[PendingCall] = []
            progressed = False
            outputs = self.state.outputs()
            for s in ready:
                node = self.state.nodes[s.id]
                failed = [d for d in s.deps() if self.state.nodes[d].status == "failed"]
                if failed:
                    node.status, node.error = "failed", f"upstream failed: {failed}"
                    progressed = True
                    continue
                try:
                    if s.when is not None and resolve({"$ref": s.when.ref, "path": s.when.path}, outputs) != s.when.equals:
                        node.status = "skipped"
                        progressed = True
                        emit("node_skipped", node_id=s.id)
                        continue
                    args = resolve(s.args, outputs, self.a.lenient_paths)
                except RefResolutionError as e:
                    node.status, node.error = "failed", f"argument resolution: {e}"
                    progressed = True
                    continue
                key = cache_key(s.tool, args)
                hit, value = self.cache.get(key)
                if hit:
                    node.status, node.output, node.args, node.cached = "ok", value, args, True
                    self.calls.append(CallRecord(s.id, s.tool, args, value, cached=True))
                    emit("cache_hit", node_id=s.id, tool=s.tool)
                    progressed = True
                    continue
                calls.append(PendingCall(s, args, key))
            if calls:
                chosen = set(self.a.scheduler.select([c.step.id for c in calls]))
                self.waves += 1
                wave = [c for c in calls if c.step.id in chosen]
                emit("wave", index=self.waves, nodes=[c.step.id for c in wave])
                return wave
            if not progressed:
                return []

    # ------------------------------------------------------------ verification
    async def observe(self, wave: list[PendingCall], outcomes: dict[str, Outcome]) -> None:
        a, task = self.a, self.task
        assert self.plan is not None
        for c in wave:
            o = outcomes[c.step.id]
            node = self.state.nodes[c.step.id]
            node.attempts += 1
            node.args, node.fault = c.args, o.fault
            node.status = "failed" if o.error else "ok"
            node.output, node.error = o.output, o.error
            self.calls.append(CallRecord(c.step.id, c.step.tool, c.args, o.output, o.error))

        ok = [c for c in wave if outcomes[c.step.id].error is None]
        gated: dict[str, Any] = {}
        if ok and a.gate.enabled("post"):
            results = await asyncio.gather(*[
                a.gate.classify(GateInput(
                    phase="post", query=task.query, plan=self.plan, tools=task.tools, step=c.step,
                    args=c.args, output=outcomes[c.step.id].output,
                    meta={"fault": outcomes[c.step.id].fault},
                )) for c in ok
            ])
            gated = {c.step.id: r for c, r in zip(ok, results)}

        flags: list[Flag] = []
        for c in wave:
            o = outcomes[c.step.id]
            g = gated.get(c.step.id)
            if o.error is not None:
                flags.append(Flag(c.step.id, f"tool error: {o.error}", c.args, None))
            elif g is not None and g.flagged:
                flags.append(Flag(c.step.id, f"verifier label {g.label} {g.probs}", c.args, o.output))
            emit("node_result", node_id=c.step.id, tool=c.step.tool, safety=c.step.safety,
                 status="failed" if o.error else "ok", fault=o.fault,
                 gated=g is not None, gate_label=g.label if g else None,
                 flagged=(o.error is not None) or bool(g and g.flagged), attempt=self.state.nodes[c.step.id].attempts)

        flagged_ids = {f.node_id for f in flags}
        cleared = {c.step.id for c in ok if c.step.id not in flagged_ids}

        if flags and a.judge.enabled and self.repairs < a.max_repairs:
            self.repairs += 1
            decision = await a.judge.review(task, self.plan, flags, self.state.summary())
            if decision.verdict == "false_alarm":
                cleared |= {f.node_id for f in flags if self.state.nodes[f.node_id].status == "ok"}
            try:
                new_plan, dirty = await a.repair.apply(task, self.plan, decision, flagged_ids)
            except AbortTask as e:
                self.aborted = str(e) or "judge aborted"
                emit("abort", reason=self.aborted)
                new_plan, dirty = self.plan, set()
            if new_plan is not self.plan:
                self.plan = new_plan
                self.state.sync(new_plan)
            self.state.reset(dirty)
            emit("intervention", flagged=sorted(flagged_ids), verdict=decision.verdict, dirty=sorted(dirty))

        for c in wave:
            if c.step.id in cleared and self.state.nodes.get(c.step.id) and self.state.nodes[c.step.id].status == "ok":
                self.cache.put(c.key, outcomes[c.step.id].output)

    # ------------------------------------------------------------ result
    def result(self) -> AgentResult:
        assert self.plan is not None
        nodes = self.state.nodes
        final = {f: nodes[f].output for f in self.plan.final if f in nodes}
        ok = self.aborted is None and all(nodes[f].status in ("ok", "skipped") for f in self.plan.final if f in nodes)
        ok = ok and not any(n.status == "pending" for n in nodes.values())
        return AgentResult(
            task_id=self.task.id,
            status="completed" if ok else "failed",
            final=final,
            calls=self.calls,
            plan=self.plan,
            initial_plan=self.initial_plan,
            error=self.aborted,
        )
