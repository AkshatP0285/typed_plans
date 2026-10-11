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

from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.eval.metrics import aggregate, task_metrics
from tpj.eval.trace import tracing, write_jsonl

log = logging.getLogger(__name__)


async def evaluate(cfg: DictConfig, out_dir: Path) -> dict[str, Any]:
    bench: Benchmark = instantiate(cfg.benchmark, _convert_="all")
    agent = instantiate(cfg.agent, _convert_="all")
    tasks = bench.load()
    if cfg.limit:
        tasks = tasks[: cfg.limit]
    log.info("benchmark=%s tasks=%d", bench.name, len(tasks))

    sem = asyncio.Semaphore(cfg.task_concurrency)
    rows: list[dict[str, Any]] = []

    async def one(task: Task) -> None:
        async with sem:
            with tracing(task.id) as tr:
                try:
                    result = await agent.run(task, bench.registry(task))
                except Exception as e:  # keep the sweep alive; record the crash
                    tr.emit("crash", error=repr(e), tb=traceback.format_exc())
                    result = AgentResult(task.id, "failed", error=repr(e))
            score = bench.score(task, result)
            write_jsonl(out_dir / "trace.jsonl", tr.events)
            write_jsonl(out_dir / "results.jsonl", [{
                "task_id": task.id, "status": result.status, "score": score, "error": result.error,
                "final": result.final, "answer_text": result.answer_text,
                "plan": result.plan.model_dump() if result.plan else None,
                "initial_plan": result.initial_plan.model_dump() if result.initial_plan else None,
                "calls": [asdict(c) for c in result.calls],
            }])
            rows.append(task_metrics(tr.events, score))
            log.info("%s %s success=%.2f", task.id, result.status, score.get("success", 0.0))

    await asyncio.gather(*[one(t) for t in tasks])
    metrics = aggregate(rows)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    log.info("SR=%.3f tokens=%.0f", metrics["success_rate"], metrics["avg_tokens"].get("total", 0))
    return metrics
