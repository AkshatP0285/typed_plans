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
    """Turn the JSONL dataset into TRL prompt/completion rows (completion = the gold plan rendered in ``fmt``)."""
    # Only build_messages() is used, so the planner needs no LLM.
    planner = PromptedPlanner(llm=None, format=fmt, few_shot=few_shot)  # type: ignore[arg-type]
    rows = []
    with Path(path).open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            example = json.loads(line)
            tools = [ToolSpec.model_validate(tool) for tool in example["tools"]]
            plan = Plan.model_validate({"query": example["query"], **example["plan"]})
            # conversational prompt/completion: TRL computes loss on the completion only
            rows.append({
                "prompt": planner.build_messages(example["query"], tools),
                "completion": [{"role": "assistant", "content": fmt.render(plan)}],
            })
    return rows


def _load_tokenizer_and_model(ft: DictConfig):
    """Load the base model, 4-bit quantised (QLoRA) when ``load_in_4bit`` is set, otherwise plain bf16."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quant = None
    if ft.load_in_4bit:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    tokenizer = AutoTokenizer.from_pretrained(ft.base_model)
    model = AutoModelForCausalLM.from_pretrained(ft.base_model, quantization_config=quant,
                                                 torch_dtype=torch.bfloat16, device_map="auto",
                                                 attn_implementation=ft.attn_implementation)
    return tokenizer, model


def _training_args(cfg: DictConfig, has_eval: bool):
    from trl import SFTConfig

    ft = cfg.finetune
    return SFTConfig(
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
        eval_strategy="epoch" if has_eval else "no",
        save_strategy="epoch",
        report_to=list(ft.report_to),
        seed=cfg.seed,
    )


def train(cfg: DictConfig, fmt: PlanFormat) -> None:
    """Fine-tune a LoRA adapter on ``cfg.finetune.train_file`` and save adapter + tokenizer to ``output_dir``."""
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer

    ft = cfg.finetune
    train_ds = Dataset.from_list(build_examples(ft.train_file, fmt, ft.few_shot))
    eval_ds = None
    if ft.eval_file:
        eval_ds = Dataset.from_list(build_examples(ft.eval_file, fmt, ft.few_shot))

    tokenizer, model = _load_tokenizer_and_model(ft)
    lora = LoraConfig(r=ft.lora.r, lora_alpha=ft.lora.alpha, lora_dropout=ft.lora.dropout,
                      target_modules=list(ft.lora.target_modules), task_type="CAUSAL_LM")
    trainer = SFTTrainer(model=model, args=_training_args(cfg, eval_ds is not None), train_dataset=train_ds,
                         eval_dataset=eval_ds, processing_class=tokenizer, peft_config=lora)
    trainer.train()
    trainer.save_model(ft.output_dir)
    tokenizer.save_pretrained(ft.output_dir)
