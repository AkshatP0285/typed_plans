"""Evaluate an agent on a benchmark; writes results.jsonl, trace.jsonl and metrics.json."""
from __future__ import annotations

import asyncio
import json
import logging
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hydra.utils import instantiate
from omegaconf import DictConfig

from tpj.agent.base import Agent
from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.eval.metrics import aggregate, task_metrics
from tpj.eval.trace import Tracer, tracing, write_jsonl

log = logging.getLogger(__name__)


async def evaluate(cfg: DictConfig, out_dir: Path) -> dict[str, Any]:
    # `instantiate` builds the objects named by `_target_` in the YAML configs,
    # recursively, so `agent` arrives with its planner, gate, judge, ... already built.
    bench: Benchmark = instantiate(cfg.benchmark, _convert_="all")
    agent: Agent = instantiate(cfg.agent, _convert_="all")
    tasks = bench.load()
    if cfg.limit:
        tasks = tasks[: cfg.limit]
    log.info("benchmark=%s tasks=%d", bench.name, len(tasks))

    limit_concurrency = asyncio.Semaphore(cfg.task_concurrency)
    metric_rows: list[dict[str, Any]] = []

    async def run_one_task(task: Task) -> None:
        async with limit_concurrency:
            with tracing(task.id) as tracer:  # collects every emit() made while running this task
                result = await _run_agent(agent, bench, task, tracer)
            score = bench.score(task, result)
            write_jsonl(out_dir / "trace.jsonl", tracer.events)
            write_jsonl(out_dir / "results.jsonl", [_result_row(task, result, score)])
            metric_rows.append(task_metrics(tracer.events, score))
            log.info("%s %s success=%.2f", task.id, result.status, score.get("success", 0.0))

    await asyncio.gather(*[run_one_task(task) for task in tasks])
    metrics = aggregate(metric_rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("SR=%.3f tokens=%.0f", metrics["success_rate"], metrics["avg_tokens"].get("total", 0))
    return metrics


async def _run_agent(agent: Agent, bench: Benchmark, task: Task, tracer: Tracer) -> AgentResult:
    """Run the agent on one task. A crash fails only this task, so a long sweep keeps going."""
    try:
        return await agent.run(task, bench.registry(task))
    except Exception as e:
        tracer.emit("crash", error=repr(e), tb=traceback.format_exc())
        return AgentResult(task.id, "failed", error=repr(e))


def _result_row(task: Task, result: AgentResult, score: dict[str, float]) -> dict[str, Any]:
    """One line of results.jsonl."""
    return {
        "task_id": task.id,
        "status": result.status,
        "score": score,
        "error": result.error,
        "final": result.final,
        "answer_text": result.answer_text,
        "plan": result.plan.model_dump() if result.plan else None,
        "initial_plan": result.initial_plan.model_dump() if result.initial_plan else None,
        "calls": [asdict(call) for call in result.calls],
    }
