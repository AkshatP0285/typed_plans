"""Gate variants used for baselines and ablations."""
from __future__ import annotations

from typing import Any

from tpj.gate.base import ERROR, INSTRUCTIONS, LABELS, OKAY, Gate, GateInput, render_state
from tpj.llm.client import LLMClient, parse_json


class ConstantGate(Gate):
    """``label=OKAY``: never intervene. ``label=ERROR``: send everything to the LLM."""

    def __init__(self, label: str = OKAY, **kwargs: Any):
        super().__init__(**kwargs)
        self.label = label
        self.name = f"constant_{label.lower()}"

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        return {self.label: 1.0}


class OracleGate(Gate):
    """Flags exactly the injected faults. Upper bound for any gate."""

    name = "oracle"

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        faulty = bool(inp.meta.get("fault"))
        return {ERROR: 1.0} if faulty else {OKAY: 1.0}


class LLMClassifierGate(Gate):
    """Prompted LLM used as a classifier (fallback when Laya/JEV are unavailable)."""

    name = "llm_classifier"

    def __init__(self, llm: LLMClient, **kwargs: Any):
        super().__init__(**kwargs)
        self.llm = llm

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        labels = LABELS[inp.phase]
        options = "\n".join(f"- {k}: {v}" for k, v in labels.items())
        prompt = (
            f"{render_state(inp, self.max_chars)}\n\n{INSTRUCTIONS[inp.phase]}\n{options}\n"
            'Reply as JSON: {"label": "<one option>"}'
        )
        schema = {"type": "object", "properties": {"label": {"type": "string", "enum": list(labels)}},
                  "required": ["label"]}
        res = await self.llm.chat([{"role": "user", "content": prompt}], role="gate", json_schema=schema)
        try:
            label = parse_json(res.text)["label"]
        except (ValueError, KeyError, TypeError):
            label = ERROR
        return {label if label in labels else ERROR: 1.0}
