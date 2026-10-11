"""End-to-end loop on the toy benchmark without any model: gold planner, oracle gate, scripted judge."""
import asyncio

from tpj.agent import VerifiedAgent
from tpj.benchmarks.toy import ToyBenchmark
from tpj.cache import NodeCache
from tpj.eval.metrics import aggregate, task_metrics
from tpj.eval.trace import tracing
from tpj.executor import WaveScheduler
from tpj.faults import FaultInjector
from tpj.gate.simple import ConstantGate, OracleGate
from tpj.judge import NullEditor, NullJudge
from tpj.planner import GoldPlanner
from tpj.repair import JudgeDecision, LocalRepair, NoRepair


class RetryJudge:
    enabled = True

    async def review(self, task, plan, flags, status):
        return JudgeDecision(verdict="retry", diagnosis="transient")


def _agent(gate, judge, repair, faults, max_repairs=10):
    return VerifiedAgent(planner=GoldPlanner(), gate=gate, ir_editor=NullEditor(), judge=judge, repair=repair,
                         scheduler=WaveScheduler(), cache=NodeCache(), faults=faults, max_repairs=max_repairs)


def _run(agent):
    bench = ToyBenchmark()
    rows, results = [], []
    for task in bench.load():
        with tracing(task.id) as tr:
            res = asyncio.run(agent.run(task, bench.registry(task)))
        score = bench.score(task, res)
        rows.append(task_metrics(tr.events, score))
        results.append((res, score))
    return results, aggregate(rows)


def test_direct_gold_plan_succeeds():
    agent = _agent(ConstantGate("OKAY", check_plan=False, check_post=False), NullJudge(), NoRepair(),
                   FaultInjector())
    results, metrics = _run(agent)
    assert metrics["success_rate"] == 1.0
    goa = results[0][0]
    assert goa.final["s11"]["booking_ids"] == ["BK-T1", "BK-H1", "BK-INDOOR"]
    assert all(c.tool != "book_scooter" for c in goa.calls)  # condition respected


def test_faults_without_recovery_fail():
    faults = FaultInjector(exec_rate=1.0, exec_types=["wrong_value"])
    agent = _agent(ConstantGate("OKAY", check_plan=False, check_post=False), NullJudge(), NoRepair(), faults)
    _, metrics = _run(agent)
    assert metrics["success_rate"] < 1.0


def test_oracle_gate_and_retry_recovers_with_cache():
    faults = FaultInjector(exec_rate=1.0, exec_types=["wrong_value", "exception", "empty"])
    agent = _agent(OracleGate(check_plan=False), RetryJudge(), LocalRepair(), faults)
    _, metrics = _run(agent)
    assert metrics["success_rate"] == 1.0
    assert metrics["gate_node"]["recall"] == 1.0
    assert metrics["avg_interventions"] > 0
