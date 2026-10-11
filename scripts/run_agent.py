"""Run an agent on a benchmark.

    python scripts/run_agent.py benchmark=toy planner=gold
    python scripts/run_agent.py benchmark=nestful gate=laya faults=light
    python scripts/run_agent.py -m +experiment=rq2_gating
"""
from __future__ import annotations

import asyncio
import random
from pathlib import Path

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from tpj.runner import evaluate


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> float:
    random.seed(cfg.seed)
    out_dir = Path(HydraConfig.get().runtime.output_dir)
    metrics = asyncio.run(evaluate(cfg, out_dir))
    return metrics["success_rate"]


if __name__ == "__main__":
    main()
