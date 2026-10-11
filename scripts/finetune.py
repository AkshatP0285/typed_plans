"""QLoRA fine-tuning of the IR planner on the synthetic IR dataset.

    python scripts/finetune.py finetune.train_file=data/ir/train.jsonl finetune.eval_file=data/ir/dev.jsonl
"""
from __future__ import annotations

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig


@hydra.main(version_base="1.3", config_path="../configs", config_name="finetune")
def main(cfg: DictConfig) -> None:
    from tpj.finetune.sft import train  # imported lazily: pulls in torch / transformers / trl

    # cfg.plan_format picks the format whose rendering the model learns to produce (typed_ir by default)
    plan_format = instantiate(cfg.plan_format, _convert_="all")
    train(cfg, plan_format)


if __name__ == "__main__":
    main()
