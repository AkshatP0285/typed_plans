"""Run tau2-bench with our pipeline registered as a tau2 agent.

    python scripts/run_tau2.py benchmark=tau2 benchmark.domain=airline benchmark.num_tasks=20

tau2 runs the conversation, the user simulator and scoring; our trace is written
to <hydra run dir>/trace.jsonl. Fault injection is not applied (tau2 tools are real).
"""
from __future__ import annotations

import sys
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from omegaconf import DictConfig


def _tau2_cli_args(bench: DictConfig) -> list[str]:
    """Command line for ``tau2 run``, built from the ``benchmark=tau2`` config group."""
    args = ["tau2", "run",
            "--domain", bench.domain,
            "--agent", bench.agent_name,
            "--agent-llm", bench.agent_llm_label,
            "--user-llm", bench.user_llm,
            "--num-trials", str(bench.num_trials)]
    if bench.num_tasks:  # null / 0: leave --num-tasks out and use tau2's default
        args += ["--num-tasks", str(bench.num_tasks)]
    args += list(bench.extra_args)
    return args


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    # Imported here so tau2 (an optional dependency) is only needed when this script runs.
    from tpj.benchmarks.tau2 import register

    bench = cfg.benchmark
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    register(
        build_agent=lambda: instantiate(cfg.agent, _convert_="all"),
        responder=instantiate(cfg.models.responder, _convert_="all"),
        name=bench.agent_name,
        trace_path=str(out_dir / "trace.jsonl"),
    )
    from tau2.cli import main as tau2_main

    # tau2 only exposes a CLI entry point, so we fake its argv and call it in-process
    # (in-process so it can find the agent we just registered).
    sys.argv = _tau2_cli_args(bench)
    tau2_main()


if __name__ == "__main__":
    main()
