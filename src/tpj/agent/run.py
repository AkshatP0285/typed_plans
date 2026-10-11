"""One task's execution-aware run (Figure 1), written as a step-wise state machine.

The run decides *what* to execute next (``next_wave``) and reacts to outcomes
(``observe``); the caller decides *how* tools are executed. The local agent
executes against a tool registry, while the tau2 adapter turns each wave into
one assistant message with parallel tool calls.

Typical driver loop (see ``VerifiedAgent.run``)::

    run = PlanRun(agent, task)
    await run.prepare()                  # plan -> gate -> (edit) -> final plan
    while not run.done:
        wave = run.next_wave()           # steps whose inputs are ready
        outcomes = <execute the wave>    # caller runs the tools
        await run.observe(wave, outcomes)  # gate -> (judge -> repair) -> cache
    return run.result()
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from tpj.benchmarks.base import AgentResult, CallRecord, Task
from tpj.cache.node_cache import cache_key
from tpj.eval.trace import emit
from tpj.executor.state import ExecutionState
from tpj.gate.base import GateInput, GateOutput
from tpj.ir.dag import PlanStructureError, check_structure
from tpj.ir.refs import RefResolutionError, resolve
from tpj.ir.schema import Plan, Step
from tpj.judge.llm_judge import Flag
from tpj.repair.strategies import AbortTask

if TYPE_CHECKING:
    from tpj.agent.verified import VerifiedAgent


@dataclass
class PendingCall:
    """A step that is ready to execute, with its references already resolved."""

    step: Step
    args: dict[str, Any]
    key: str  # cache key (tool + resolved args)


@dataclass
class Outcome:
    """What happened when a PendingCall was executed."""

    output: Any = None
    error: str | None = None
    fault: str | None = None  # injected fault label, if any


class PlanRun:
    def __init__(self, agent: "VerifiedAgent", task: Task):
        self.agent = agent  # holds the components (planner, gate, judge, ...) and limits
        self.task = task
        self.plan: Plan | None = None
        self.initial_plan: Plan | None = None  # plan before any edits, for analysis
        self.state = ExecutionState()
        self.cache = agent.cache.fresh()  # every task starts with an empty cache
        self.calls: list[CallRecord] = []
        self.repairs = 0
        self.waves = 0
        self.aborted: str | None = None  # reason, once the judge gives up on the task

    # ------------------------------------------------------------ planning
    async def prepare(self) -> Plan:
        """Plan, inject plan faults, then gate + edit before execution.

        The gate and the structure check get up to ``max_plan_edits`` chances to
        send the plan to the IR editor. A plan that is still structurally broken
        afterwards raises ``PlanStructureError`` and ends the task.
        """
        agent, task = self.agent, self.task
        plan = await agent.planner.plan(task)
        plan, plan_faults = agent.faults.corrupt_plan(plan, task.tools, task.id)
        self.initial_plan = plan.model_copy(deep=True)

        for round_ in range(agent.max_plan_edits + 1):
            structure_error = self._structure_error(plan, round_)
            # Fault labels describe the original plan only, so only round 0 carries them.
            gate_flagged, gate_label = await self._gate_plan(plan, plan_faults if round_ == 0 else [])

            needs_edit = gate_flagged or structure_error is not None
            last_round = round_ == agent.max_plan_edits
            if not needs_edit or not agent.ir_editor.enabled or last_round:
                break
            plan = await agent.ir_editor.revise(
                task, plan, verdict=gate_label or "STRUCTURE_ERROR",
                extra=f"Structural problem: {structure_error}" if structure_error else "",
            )
            emit("plan_edit", round=round_)

        check_structure(plan)  # unexecutable plans end the task here
        self.plan = plan
        self.state = ExecutionState.for_plan(plan)
        emit("plan_final", plan=plan.model_dump())
        return plan

    @staticmethod
    def _structure_error(plan: Plan, round_: int) -> str | None:
        """Error message if the plan cannot be executed as is (bad refs, cycles, ...)."""
        try:
            check_structure(plan)
        except PlanStructureError as e:
            emit("plan_structure_error", round=round_, error=str(e))
            return str(e)
        return None

    async def _gate_plan(self, plan: Plan, faults: list[dict]) -> tuple[bool, str | None]:
        """Ask the gate about the whole plan. Returns (flagged, label)."""
        if not self.agent.gate.enabled("plan"):
            return False, None
        verdict = await self.agent.gate.classify(GateInput(
            phase="plan", query=self.task.query, plan=plan, tools=self.task.tools,
            meta={"fault": faults},
        ))
        return verdict.flagged, verdict.label

    # ------------------------------------------------------------ scheduling
    @property
    def done(self) -> bool:
        return self.aborted is not None or self.waves >= self.agent.max_waves or not self._ready_steps()

    def _ready_steps(self) -> list[Step]:
        """Pending steps whose dependencies have all finished (ok, failed or skipped)."""
        assert self.plan is not None
        nodes = self.state.nodes
        ready = []
        for step in self.plan.steps:
            if nodes[step.id].status != "pending":
                continue
            if all(nodes[dep].status != "pending" for dep in step.deps()):
                ready.append(step)
        return ready

    def next_wave(self) -> list[PendingCall]:
        """Return the next batch of tool calls to execute, or ``[]`` if nothing is left.

        Some ready steps need no tool call: their upstream failed, their ``when``
        condition is false, their references cannot be resolved, or the cache
        already has their output. Those are settled right here. Settling them can
        make further steps ready, so we loop until there is something to execute.
        """
        if self.aborted is not None or self.waves >= self.agent.max_waves:
            return []
        while True:
            ready = self._ready_steps()
            if not ready:
                return []
            outputs = self.state.outputs()
            to_execute: list[PendingCall] = []
            settled_any = False
            for step in ready:
                call = self._prepare_call(step, outputs)
                if call is None:
                    settled_any = True
                else:
                    to_execute.append(call)
            if to_execute:
                return self._start_wave(to_execute)
            if not settled_any:  # safety net against looping forever
                return []

    def _prepare_call(self, step: Step, outputs: dict[str, Any]) -> PendingCall | None:
        """Turn a ready step into a call, or settle it without running and return None."""
        node = self.state.nodes[step.id]

        failed_deps = [dep for dep in step.deps() if self.state.nodes[dep].status == "failed"]
        if failed_deps:
            node.status, node.error = "failed", f"upstream failed: {failed_deps}"
            return None

        try:
            if not self._condition_holds(step, outputs):
                node.status = "skipped"
                emit("node_skipped", node_id=step.id)
                return None
            args = resolve(step.args, outputs, self.agent.lenient_paths)
        except RefResolutionError as e:
            node.status, node.error = "failed", f"argument resolution: {e}"
            return None

        key = cache_key(step.tool, args)
        hit, cached_output = self.cache.get(key)
        if hit:
            node.status, node.output, node.args, node.cached = "ok", cached_output, args, True
            self.calls.append(CallRecord(step.id, step.tool, args, cached_output, cached=True))
            emit("cache_hit", node_id=step.id, tool=step.tool)
            return None

        return PendingCall(step, args, key)

    @staticmethod
    def _condition_holds(step: Step, outputs: dict[str, Any]) -> bool:
        """True if the step has no ``when`` condition or the condition is met."""
        if step.when is None:
            return True
        value = resolve({"$ref": step.when.ref, "path": step.when.path}, outputs)
        return value == step.when.equals

    def _start_wave(self, calls: list[PendingCall]) -> list[PendingCall]:
        """Let the scheduler pick which calls run now (all of them, or one when sequential)."""
        chosen = set(self.agent.scheduler.select([call.step.id for call in calls]))
        wave = [call for call in calls if call.step.id in chosen]
        self.waves += 1
        emit("wave", index=self.waves, nodes=[call.step.id for call in wave])
        return wave

    # ------------------------------------------------------------ verification
    async def observe(self, wave: list[PendingCall], outcomes: dict[str, Outcome]) -> None:
        """React to an executed wave: gate the outputs, repair flagged steps, cache good outputs.

        ``outcomes`` maps step id -> Outcome for every call in ``wave``.
        """
        assert self.plan is not None
        self._record_outcomes(wave, outcomes)

        succeeded = [call for call in wave if outcomes[call.step.id].error is None]
        verdicts = await self._gate_outputs(succeeded, outcomes)
        flags = self._collect_flags(wave, outcomes, verdicts)

        # Outputs that ran fine and were not flagged are safe to cache.
        flagged_ids = {flag.node_id for flag in flags}
        cleared = {call.step.id for call in succeeded if call.step.id not in flagged_ids}

        agent = self.agent
        if flags and agent.judge.enabled and self.repairs < agent.max_repairs:
            cleared |= await self._judge_and_repair(flags, flagged_ids)

        self._cache_outputs(wave, outcomes, cleared)

    def _record_outcomes(self, wave: list[PendingCall], outcomes: dict[str, Outcome]) -> None:
        """Store each call's result in the execution state and the call log."""
        for call in wave:
            outcome = outcomes[call.step.id]
            node = self.state.nodes[call.step.id]
            node.attempts += 1
            node.args, node.fault = call.args, outcome.fault
            node.status = "failed" if outcome.error else "ok"
            node.output, node.error = outcome.output, outcome.error
            self.calls.append(CallRecord(call.step.id, call.step.tool, call.args, outcome.output, outcome.error))

    async def _gate_outputs(self, succeeded: list[PendingCall],
                            outcomes: dict[str, Outcome]) -> dict[str, GateOutput]:
        """Run the post-execution gate on every successful call, in parallel.

        Failed calls are not gated: an exception is already a clear signal.
        """
        if not succeeded or not self.agent.gate.enabled("post"):
            return {}
        inputs = [
            GateInput(
                phase="post", query=self.task.query, plan=self.plan, tools=self.task.tools, step=call.step,
                args=call.args, output=outcomes[call.step.id].output,
                meta={"fault": outcomes[call.step.id].fault},
            )
            for call in succeeded
        ]
        verdicts = await asyncio.gather(*[self.agent.gate.classify(inp) for inp in inputs])
        return {call.step.id: verdict for call, verdict in zip(succeeded, verdicts)}

    def _collect_flags(self, wave: list[PendingCall], outcomes: dict[str, Outcome],
                       verdicts: dict[str, GateOutput]) -> list[Flag]:
        """A step is flagged if its tool raised or the gate flagged its output."""
        flags: list[Flag] = []
        for call in wave:
            outcome = outcomes[call.step.id]
            verdict = verdicts.get(call.step.id)  # None when the step was not gated
            failed = outcome.error is not None
            gate_flagged = verdict is not None and verdict.flagged

            if failed:
                flags.append(Flag(call.step.id, f"tool error: {outcome.error}", call.args, None))
            elif gate_flagged:
                flags.append(Flag(call.step.id, f"verifier label {verdict.label} {verdict.probs}",
                                  call.args, outcome.output))

            emit("node_result", node_id=call.step.id, tool=call.step.tool, safety=call.step.safety,
                 status="failed" if failed else "ok", fault=outcome.fault,
                 gated=verdict is not None, gate_label=verdict.label if verdict else None,
                 flagged=failed or gate_flagged, attempt=self.state.nodes[call.step.id].attempts)
        return flags

    async def _judge_and_repair(self, flags: list[Flag], flagged_ids: set[str]) -> set[str]:
        """Ask the judge about flagged steps and apply its repair.

        Returns the flagged steps the judge cleared (a ``false_alarm`` verdict),
        so the caller can cache them too.
        """
        assert self.plan is not None
        self.repairs += 1
        decision = await self.agent.judge.review(self.task, self.plan, flags, self.state.summary())

        cleared: set[str] = set()
        if decision.verdict == "false_alarm":
            cleared = {flag.node_id for flag in flags if self.state.nodes[flag.node_id].status == "ok"}

        try:
            new_plan, dirty = await self.agent.repair.apply(self.task, self.plan, decision, flagged_ids)
        except AbortTask as e:
            self.aborted = str(e) or "judge aborted"
            emit("abort", reason=self.aborted)
            new_plan, dirty = self.plan, set()

        if new_plan is not self.plan:
            self.plan = new_plan
            self.state.sync(new_plan)  # add new steps, drop removed ones
        self.state.reset(dirty)  # "dirty" steps go back to pending and will run again
        emit("intervention", flagged=sorted(flagged_ids), verdict=decision.verdict, dirty=sorted(dirty))
        return cleared

    def _cache_outputs(self, wave: list[PendingCall], outcomes: dict[str, Outcome], cleared: set[str]) -> None:
        """Cache cleared outputs, unless the repair removed the step or reset it to pending."""
        for call in wave:
            node = self.state.nodes.get(call.step.id)
            if call.step.id in cleared and node is not None and node.status == "ok":
                self.cache.put(call.key, outcomes[call.step.id].output)

    # ------------------------------------------------------------ result
    def result(self) -> AgentResult:
        """Completed only if not aborted, every final step is ok/skipped and nothing is still pending."""
        assert self.plan is not None
        nodes = self.state.nodes
        final_ids = [step_id for step_id in self.plan.final if step_id in nodes]
        final = {step_id: nodes[step_id].output for step_id in final_ids}

        finals_ok = all(nodes[step_id].status in ("ok", "skipped") for step_id in final_ids)
        nothing_pending = not any(node.status == "pending" for node in nodes.values())
        completed = self.aborted is None and finals_ok and nothing_pending

        return AgentResult(
            task_id=self.task.id,
            status="completed" if completed else "failed",
            final=final,
            calls=self.calls,
            plan=self.plan,
            initial_plan=self.initial_plan,
            error=self.aborted,
        )
