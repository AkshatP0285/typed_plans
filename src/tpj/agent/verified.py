"""The execution-aware agent: plan -> verify -> wave execution -> selective repair."""
from __future__ import annotations

import asyncio

from tpj.agent.base import Agent
from tpj.agent.run import Outcome, PendingCall, PlanRun
from tpj.benchmarks.base import AgentResult, Task
from tpj.cache.node_cache import NodeCache
from tpj.eval.trace import emit
from tpj.executor.scheduling import WaveScheduler
from tpj.faults.injector import RAISING, FaultInjector
from tpj.gate.base import Gate
from tpj.ir.dag import PlanStructureError
from tpj.ir.formats.base import PlanParseError
from tpj.judge.llm_judge import LLMIREditor, LLMJudge, NullEditor, NullJudge
from tpj.planner.base import Planner
from tpj.repair.strategies import RepairStrategy
from tpj.tools.registry import ToolRegistry


class VerifiedAgent(Agent):
    """The proposed system. Each component is pluggable through its Hydra config group.

    The per-task logic lives in :class:`PlanRun`; this class holds the components
    and executes each wave's tool calls.
    """

    def __init__(
        self,
        planner: Planner,
        gate: Gate,
        ir_editor: LLMIREditor | NullEditor,
        judge: LLMJudge | NullJudge,
        repair: RepairStrategy,
        scheduler: WaveScheduler,
        cache: NodeCache,
        faults: FaultInjector,
        max_plan_edits: int = 2,
        max_repairs: int = 3,
        max_waves: int = 50,
        lenient_paths: bool = True,
    ):
        self.planner = planner
        self.gate = gate
        self.ir_editor = ir_editor
        self.judge = judge
        self.repair = repair
        self.scheduler = scheduler
        self.cache = cache
        self.faults = faults
        self.max_plan_edits = max_plan_edits
        self.max_repairs = max_repairs
        self.max_waves = max_waves
        self.lenient_paths = lenient_paths

    async def run(self, task: Task, registry: ToolRegistry) -> AgentResult:
        run = PlanRun(self, task)
        try:
            await run.prepare()
        except (PlanParseError, PlanStructureError) as e:
            emit("plan_error", error=str(e))
            return AgentResult(task.id, "plan_error", initial_plan=run.initial_plan, error=str(e))

        while not run.done:
            wave = run.next_wave()
            if not wave:
                break
            # All calls in a wave are independent of each other, so they run concurrently.
            results = await asyncio.gather(*[self._execute(task, run, call, registry) for call in wave])
            await run.observe(wave, dict(results))
        return run.result()

    async def _execute(self, task: Task, run: PlanRun, call: PendingCall, registry: ToolRegistry) -> tuple[str, Outcome]:
        """Run one tool call, possibly with an injected fault. Never raises.

        Returns ``(step_id, outcome)``. Faults in ``RAISING`` replace the call with an
        exception; the others let the tool run and then corrupt its output.
        """
        step_id = call.step.id
        fault = self.faults.exec_fault(task.id, step_id, run.state.nodes[step_id].attempts)
        if fault:
            emit("fault", level="exec", fault_type=fault, step=step_id)
        try:
            if fault in RAISING:
                self.faults.apply_exec(fault, None, task.id, step_id)  # raises
            output = await registry.call(call.step.tool, call.args)
            if fault:
                output = self.faults.apply_exec(fault, output, task.id, step_id)
            return step_id, Outcome(output=output, fault=fault)
        except Exception as e:  # tool errors are observations for the judge, not crashes
            return step_id, Outcome(error=f"{type(e).__name__}: {e}", fault=fault)
