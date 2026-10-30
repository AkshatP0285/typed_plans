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
    # Ported from the JEV- branch's pre-exec pilot, prompt "fewshot" (the jev_or_fewshot4_v1
    # experiment): F1 .961, detect .984 on the 466-case frozen test set - the best of 3 prompt
    # variants tried. That pilot used binary VALID/INVALID; renamed to OKAY/ERROR here so the
    # wording, the live criteria sent to JEV, and plan_verification_fewshot.json's "verdict"
    # values all agree. NEEDS_FORMAT_CHANGE is intentionally not offered at the plan phase - it
    # was never part of what was actually tested.
    "plan": {
        OKAY: "The plan correctly and completely fulfils the user request: every step uses an "
        "appropriate available tool with arguments that match the tool schema and the request, "
        "dependencies are correct, and no needed step is missing or unneeded step is present.",
        ERROR: "The plan has at least one defect: a needed step is missing, a step is unnecessary "
        "or redundant, a step uses the wrong tool, an argument value is wrong for the request or "
        "violates the tool schema, or a dependency is wrong.",
    },
    # Ported from the JEV- branch's post-exec pilot (post-exec/run_experiments.py) - same 3-way
    # label set already used here, just more precise, disambiguating wording.
    "post": {
        OKAY: "Execution completed successfully AND the returned output satisfies the expected "
        "output schema/requirements provided in the case.",
        ERROR: "The tool call failed or the tool output explicitly reports an execution failure, "
        "exception, timeout, or unsuccessful operation. Use this for execution failure, not "
        "merely a successful result with the wrong format.",
        NEEDS_FORMAT_CHANGE: "Execution appears to have completed, but the returned output is "
        "missing required fields, has incompatible types, or violates the expected output "
        "format/schema. Use this when the main issue is output representation.",
    },
}

INSTRUCTIONS: dict[str, str] = {
    "plan": "Classify the plan in case_to_classify (a PRE-EXECUTION plan: ordered steps; an "
    "argument {\"$ref\": \"sN\", \"path\": \"field.sub\"} means an output of step sN) against its "
    "user_query and available_tools. labeled_examples are solved examples for reference only: "
    "each shows a user_query, a plan and the correct verdict (with a reason when ERROR). Do not "
    "classify the examples. Return OKAY only if the plan in case_to_classify is correct and "
    "complete for its request; otherwise return ERROR.",
    "post": "Classify this POST-EXECUTION tool outcome. Use the user request, tool call, "
    "execution status/output, and expected output requirements. Return exactly one label from "
    "the supplied criteria. If the tool explicitly failed, choose ERROR. If execution succeeded "
    "but its output violates the expected schema, choose NEEDS_FORMAT_CHANGE. Choose OKAY only "
    "when execution succeeded and the output meets the requirements. Do not infer success merely "
    "because output exists. Note: this environment's tools are mocked for benchmarking - a "
    "generic placeholder object such as {\"id\": ..., \"value\": \"<Type:hash>\"} is a normal, "
    "correct mock result for ANY declared return type (including when the placeholder's own "
    "label differs from the declared type, e.g. a step declaring 'bool' legitimately receiving "
    "such a placeholder) and must be classified OKAY. Only choose ERROR or NEEDS_FORMAT_CHANGE "
    "when the tool explicitly reports failure, raises an exception, or the output is otherwise "
    "malformed (e.g. missing entirely, or an error message) - not merely generic-looking.",
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
                 max_chars: int = 6000, post_steps: str = "all"):
        self.threshold = threshold
        # "all": check every executed step; "danger": only steps the plan marks safety=danger
        # (needs a format that carries safety tags, i.e. typed IR).
        self.post_steps = post_steps
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
