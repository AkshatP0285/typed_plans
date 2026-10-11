"""Lightweight gate: classifies a plan or an executed step outcome.

The gate returns a probability distribution over OKAY / ERROR /
NEEDS_FORMAT_CHANGE. The step is flagged (sent to the LLM editor or judge)
when ``1 - P(OKAY) >= threshold``.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal

from tpj.eval.trace import emit
from tpj.ir.schema import Plan, Step
from tpj.tools.spec import ToolSpec

Phase = Literal["plan", "post"]
OKAY, ERROR, NEEDS_FORMAT_CHANGE = "OKAY", "ERROR", "NEEDS_FORMAT_CHANGE"

LABELS: dict[str, dict[str, str]] = {
    "plan": {
        OKAY: "The plan fulfils the request completely with the right tools, arguments and dependencies.",
        ERROR: "The plan uses a wrong tool, wrong argument value, wrong dependency, or misses/adds a step.",
        NEEDS_FORMAT_CHANGE: "The plan is right in substance but an argument's format or type does not "
        "match what the tool expects.",
    },
    "post": {
        OKAY: "The tool output is a valid, plausible result of this call and can be used by later steps.",
        ERROR: "The output shows a failure, is empty, implausible, or inconsistent with the request or arguments.",
        NEEDS_FORMAT_CHANGE: "The output is right in substance but its structure or type does not match "
        "what later steps expect.",
    },
}

INSTRUCTIONS: dict[str, str] = {
    "plan": "Is this tool-calling plan correct for the user's request?",
    "post": "Is the observed outcome of this tool call acceptable?",
}


@dataclass
class GateInput:
    phase: Phase
    query: str
    plan: Plan
    tools: list[ToolSpec] = field(default_factory=list)
    step: Step | None = None
    args: dict[str, Any] | None = None
    output: Any = None
    meta: dict[str, Any] = field(default_factory=dict)  # e.g. injected fault labels (oracle only)


@dataclass
class GateOutput:
    label: str  # most likely label
    probs: dict[str, float]
    flagged: bool  # True -> send to the LLM editor / judge


def _clip(value: Any, limit: int) -> str:
    s = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    return s if len(s) <= limit else s[:limit] + " ...[truncated]"


def render_state(inp: GateInput, max_chars: int = 6000) -> str:
    """Text the gate sees. Shared by all gate implementations."""
    parts = [f"User request: {inp.query}"]
    if inp.phase == "plan":
        tools = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in inp.tools]
        parts.append("Available tools: " + _clip(tools, max_chars // 2))
        parts.append("Plan: " + _clip(inp.plan.to_prompt_json(), max_chars // 2))
    else:
        assert inp.step is not None
        spec = next((t for t in inp.tools if t.name == inp.step.tool), None)
        parts.append(f"Step {inp.step.id}: tool '{inp.step.tool}' (declared return type {inp.step.returns})")
        if spec is not None:
            parts.append(f"Tool description: {_clip(spec.description, 600)}")
        parts.append("Arguments: " + _clip(inp.args, max_chars // 4))
        parts.append("Output: " + _clip(inp.output, max_chars // 2))
    return "\n".join(parts)


class Gate(ABC):
    """Base class: subclasses implement ``_classify``; callers use ``classify``."""

    name = "gate"

    def __init__(self, threshold: float = 0.5, check_plan: bool = True, check_post: bool = True,
                 max_chars: int = 6000):
        self.threshold = threshold
        self.check_plan = check_plan
        self.check_post = check_post
        self.max_chars = max_chars

    def enabled(self, phase: Phase) -> bool:
        return self.check_plan if phase == "plan" else self.check_post

    async def classify(self, inp: GateInput) -> GateOutput:
        """Shared wrapper around :meth:`_classify`: normalise, apply the threshold, trace."""
        start = time.time()
        raw_probs = await self._classify(inp)
        # Keep only this phase's labels (missing ones count as 0) and make them sum to 1.
        probs = {label: float(raw_probs.get(label, 0.0)) for label in LABELS[inp.phase]}
        total = sum(probs.values()) or 1.0
        probs = {label: p / total for label, p in probs.items()}
        label = max(probs, key=probs.get)
        flagged = (1.0 - probs[OKAY]) >= self.threshold
        emit(
            "gate",
            gate=self.name,
            phase=inp.phase,
            node_id=inp.step.id if inp.step else None,
            label=label,
            probs=probs,
            flagged=flagged,
            latency=time.time() - start,
        )
        return GateOutput(label, probs, flagged)

    @abstractmethod
    async def _classify(self, inp: GateInput) -> dict[str, float]:
        """Implemented by each gate: label -> score (need not be normalised)."""
