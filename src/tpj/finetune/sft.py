"""QLoRA supervised fine-tuning of the IR planner.

Input: JSONL where each line is
``{"query": str, "tools": [ToolSpec dict, ...], "plan": {"steps": [...], "final": [...]}}``
(the synthetic IR dataset). Prompts are built by :class:`PromptedPlanner`, so
training and inference see identical messages. Serve the adapter with vLLM
(``--enable-lora --lora-modules ir_planner=<output_dir>``) and select
``model@models.planner=planner_lora``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from tpj.ir.formats.base import PlanFormat
from tpj.ir.schema import Plan
from tpj.planner.prompted import PromptedPlanner
from tpj.tools.spec import ToolSpec


def build_examples(path: str | Path, fmt: PlanFormat, few_shot: bool) -> list[dict[str, Any]]:
    planner = PromptedPlanner(llm=None, format=fmt, few_shot=few_shot)  # type: ignore[arg-type]
    rows = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            tools = [ToolSpec.model_validate(t) for t in ex["tools"]]
            plan = Plan.model_validate({"query": ex["query"], **ex["plan"]})
            # conversational prompt/completion: TRL computes loss on the completion only
            rows.append({
                "prompt": planner.build_messages(ex["query"], tools),
                "completion": [{"role": "assistant", "content": fmt.render(plan)}],
            })
    return rows


def train(cfg: DictConfig, fmt: PlanFormat) -> None:
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    ft = cfg.finetune
    train_ds = Dataset.from_list(build_examples(ft.train_file, fmt, ft.few_shot))
    eval_ds = Dataset.from_list(build_examples(ft.eval_file, fmt, ft.few_shot)) if ft.eval_file else None

    quant = None
    if ft.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tok = AutoTokenizer.from_pretrained(ft.base_model)
    model = AutoModelForCausalLM.from_pretrained(ft.base_model, quantization_config=quant,
                                                 torch_dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation=ft.attn_implementation)
    lora = LoraConfig(r=ft.lora.r, lora_alpha=ft.lora.alpha, lora_dropout=ft.lora.dropout,
                      target_modules=list(ft.lora.target_modules), task_type="CAUSAL_LM")
    args = SFTConfig(
        output_dir=ft.output_dir,
        num_train_epochs=ft.epochs,
        per_device_train_batch_size=ft.batch_size,
        gradient_accumulation_steps=ft.grad_accum,
        learning_rate=ft.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=ft.max_length,
        bf16=True,
        logging_steps=10,
        eval_strategy="epoch" if eval_ds is not None else "no",
        save_strategy="epoch",
        report_to=list(ft.report_to),
        seed=cfg.seed,
    )
    trainer = SFTTrainer(model=model, args=args, train_dataset=train_ds, eval_dataset=eval_ds,
                         processing_class=tok, peft_config=lora)
    trainer.train()
    trainer.save_model(ft.output_dir)
    tok.save_pretrained(ft.output_dir)
