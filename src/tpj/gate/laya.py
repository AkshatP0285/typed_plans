"""Laya gate: open-source text classifier run locally with transformers.

The Laya model card (convaiinnovations/layab) is gated, so two input modes are
supported and chosen in config:

- ``choice_lm``: a causal LM scores lettered options; probabilities are the
  softmax over the option-letter logits at the answer position.
- ``seq_cls``: a sequence-classification head; ``label_map`` maps the model's
  ``id2label`` names onto OKAY / ERROR / NEEDS_FORMAT_CHANGE.

Check the model card once you have access and set ``mode`` accordingly.
"""
from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal

from tpj.gate.base import INSTRUCTIONS, LABELS, Gate, GateInput, render_state

_LETTERS = "ABCDEFGH"


class LayaGate(Gate):
    name = "laya"

    def __init__(
        self,
        model_id: str = "convaiinnovations/layab",
        mode: Literal["choice_lm", "seq_cls"] = "choice_lm",
        device: str = "cuda",
        dtype: str = "bfloat16",
        label_map: dict[str, str] | None = None,
        max_length: int = 4096,
        trust_remote_code: bool = True,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model_id = model_id
        self.mode = mode
        self.device = device
        self.dtype = dtype
        self.label_map = label_map or {}
        self.max_length = max_length
        self.trust_remote_code = trust_remote_code
        self._model = None
        self._tok = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        cls = AutoModelForCausalLM if self.mode == "choice_lm" else AutoModelForSequenceClassification
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        self._model = cls.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.dtype),
            trust_remote_code=self.trust_remote_code,
        ).to(self.device).eval()

    def _prompt(self, inp: GateInput) -> tuple[str, list[str]]:
        labels = list(LABELS[inp.phase])
        options = "\n".join(f"{_LETTERS[i]}. {lab}: {LABELS[inp.phase][lab]}" for i, lab in enumerate(labels))
        body = (
            f"{render_state(inp, self.max_chars)}\n\nQuestion: {INSTRUCTIONS[inp.phase]}\n"
            f"Options:\n{options}\nAnswer with the letter of the correct option."
        )
        if getattr(self._tok, "chat_template", None):
            text = self._tok.apply_chat_template(
                [{"role": "user", "content": body}], tokenize=False, add_generation_prompt=True
            )
        else:
            text = body + "\nAnswer:"
        return text, labels

    def _infer(self, inp: GateInput) -> dict[str, float]:
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            tok, model = self._tok, self._model
            if self.mode == "choice_lm":
                text, labels = self._prompt(inp)
                enc = tok(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
                with torch.no_grad():
                    logits = model(**enc).logits[0, -1]
                ids = [tok.encode(_LETTERS[i], add_special_tokens=False)[-1] for i in range(len(labels))]
                p = torch.softmax(logits[ids].float(), dim=-1).tolist()
                return dict(zip(labels, p))
            text = render_state(inp, self.max_chars)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
            with torch.no_grad():
                p = torch.softmax(model(**enc).logits[0].float(), dim=-1).tolist()
            probs: dict[str, float] = {}
            for idx, val in enumerate(p):
                name = model.config.id2label.get(idx, str(idx))
                ours = self.label_map.get(name, name)
                probs[ours] = probs.get(ours, 0.0) + val
            return probs

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        return await asyncio.to_thread(self._infer, inp)
