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

_LETTERS = "ABCDEFGH"  # option letters used in choice_lm prompts (A = first label, B = second, ...)


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
        # The model runs in worker threads (see _classify); the lock serialises inference and the lazy load.
        self._lock = threading.Lock()

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

        model_class = AutoModelForCausalLM if self.mode == "choice_lm" else AutoModelForSequenceClassification
        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        self._model = model_class.from_pretrained(
            self.model_id,
            torch_dtype=getattr(torch, self.dtype),
            trust_remote_code=self.trust_remote_code,
        ).to(self.device).eval()

    def _prompt(self, inp: GateInput) -> tuple[str, list[str]]:
        """Build the multiple-choice prompt for ``choice_lm`` mode.

        Returns the prompt text and the labels in option order (labels[i] is option ``_LETTERS[i]``).
        """
        label_descriptions = LABELS[inp.phase]
        labels = list(label_descriptions)
        option_lines = [f"{_LETTERS[i]}. {label}: {label_descriptions[label]}" for i, label in enumerate(labels)]
        options = "\n".join(option_lines)
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
        with self._lock:
            if self._model is None:
                self._load()
            if self.mode == "choice_lm":
                return self._infer_choice_lm(inp)
            return self._infer_seq_cls(inp)

    def _infer_choice_lm(self, inp: GateInput) -> dict[str, float]:
        """P(label) = softmax over the next-token logits of the option letters only (A, B, C, ...)."""
        import torch

        text, labels = self._prompt(inp)
        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            next_token_logits = self._model(**inputs).logits[0, -1]
        # Token id of each option letter ([-1]: take the last piece if the tokenizer splits it).
        letter_ids = [self._tok.encode(_LETTERS[i], add_special_tokens=False)[-1] for i in range(len(labels))]
        letter_probs = torch.softmax(next_token_logits[letter_ids].float(), dim=-1).tolist()
        return dict(zip(labels, letter_probs))

    def _infer_seq_cls(self, inp: GateInput) -> dict[str, float]:
        """Softmax over the classifier head, with the model's class names mapped onto our labels.

        Several model classes may map to the same gate label, so their probabilities are summed.
        """
        import torch

        text = render_state(inp, self.max_chars)
        inputs = self._tok(text, return_tensors="pt", truncation=True, max_length=self.max_length).to(self.device)
        with torch.no_grad():
            class_probs = torch.softmax(self._model(**inputs).logits[0].float(), dim=-1).tolist()
        probs: dict[str, float] = {}
        for class_idx, prob in enumerate(class_probs):
            model_label = self._model.config.id2label.get(class_idx, str(class_idx))
            gate_label = self.label_map.get(model_label, model_label)
            probs[gate_label] = probs.get(gate_label, 0.0) + prob
        return probs

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        # transformers inference is blocking; run it in a thread so the event loop keeps serving other tasks
        return await asyncio.to_thread(self._infer, inp)
