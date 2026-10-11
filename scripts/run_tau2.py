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


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    from tpj.benchmarks.tau2 import register

    b = cfg.benchmark
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    register(
        build_agent=lambda: instantiate(cfg.agent, _convert_="all"),
        responder=instantiate(cfg.models.responder, _convert_="all"),
        name=b.agent_name,
        trace_path=str(out_dir / "trace.jsonl"),
    )
    from tau2.cli import main as tau2_main

    sys.argv = ["tau2", "run", "--domain", b.domain, "--agent", b.agent_name,
                "--agent-llm", b.agent_llm_label, "--user-llm", b.user_llm,
                "--num-trials", str(b.num_trials), *(["--num-tasks", str(b.num_tasks)] if b.num_tasks else []),
                *b.extra_args]
    tau2_main()


if __name__ == "__main__":
    main()
