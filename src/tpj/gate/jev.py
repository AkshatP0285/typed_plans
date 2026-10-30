"""JEV (TypeSafe AI System One model) gate over the HTTP API.

API: POST {url} with ``{"state": str, "model": str, "questions": {id: {"type":
"choice", "instructions": str, "criteria": {option: description}}}}``. Each
choice answer carries ``choice``, ``confidence`` and ``probabilities``.

``url`` is the full endpoint - different deployments use different paths with
the same request/response shape (TypeSafe direct: .../v1/systemone; JEV via
OpenRouter's Decisions API: .../api/alpha/decisions), so this takes the
complete URL rather than assuming a fixed suffix.

We ask a single question with id ``"outcome"`` whose options are the gate labels, so a
typical response looks like::

    {"model": ..., "input_tokens": 812, "output_tokens": 3,
     "answers": {"outcome": {"choice": "OKAY", "confidence": 0.91,
                             "probabilities": {"OKAY": 0.91, "ERROR": 0.06, "NEEDS_FORMAT_CHANGE": 0.03}}}}
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

from tpj.eval.trace import emit
from tpj.gate.base import INSTRUCTIONS, LABELS, Gate, GateInput, render_state
from tpj.ir.schema import Plan
from tpj.ir.text_view import REF_NOTE as IR_TEXT_NOTE
from tpj.ir.text_view import ir_text

IR_REF_SENTENCE = 'an argument {"$ref": "sN", "path": "field.sub"} means an output of step sN'
assert IR_REF_SENTENCE in INSTRUCTIONS["plan"]


class JEVGate(Gate):
    """Gate backed by the hosted JEV classifier. Responses are cached on disk keyed by the request body."""

    name = "jev"

    def __init__(
        self,
        model: str = "jev-latest",
        url: str = "https://api.typesafe.ai/v1/systemone",
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout: float = 30.0,
        cache_dir: str | None = ".cache/jev",
        plan_fewshot_file: str | None = None,
        compact_plan: bool = False,
        plan_view: str = "json",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.compact_plan = compact_plan  # show the plan without default-valued fields
        self.plan_view = plan_view  # "json" or "ir_text" (typed IR as text lines, see tpj.ir.text_view)
        self.model = model
        self.url = url
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._client = None
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(cache_dir)
        # Few-shot demonstrations for the "plan" phase only (the fewshot4_v1
        # pilot that scored F1 .961 on the 466-case pre-exec test set).
        # Bookkeeping keys starting with "_" are dropped - not sent to JEV.
        self.plan_fewshot: list[dict[str, Any]] = []
        if plan_fewshot_file:
            with open(plan_fewshot_file, encoding="utf-8") as f:
                raw = json.load(f)
            self.plan_fewshot = [{k: v for k, v in ex.items() if not k.startswith("_")} for ex in raw]

    def _http(self):
        """Create the HTTP client on first use, so the API key is only required when the gate actually runs."""
        if self._client is None:
            import httpx

            api_key = os.environ.get(self.api_key_env)
            if not api_key:
                raise RuntimeError(f"set {self.api_key_env} to use the JEV gate")
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            )
        return self._client

    def _plan_state(self, inp: GateInput) -> dict[str, Any]:
        """Structured (dict, not string) state for the plan phase, matching the exact shape the
        fewshot4_v1 pilot was validated against: {user_query, available_tools, plan}, wrapped in
        labeled_examples/case_to_classify when few-shot demonstrations are configured."""
        tools = [{"name": t.name, "description": t.description, "parameters": t.parameters} for t in inp.tools]
        if self.plan_view == "ir_text":
            case = {"user_query": inp.query, "available_tools": tools, "plan": ir_text(inp.plan)}
            fewshot = [{**ex, "plan": ir_text(Plan.model_validate({**ex["plan"], "query": ex["user_query"]}))}
                       for ex in self.plan_fewshot]
        else:
            case = {"user_query": inp.query, "available_tools": tools,
                    "plan": json.loads(inp.plan.to_prompt_json(compact=self.compact_plan))}
            fewshot = self.plan_fewshot
        if not fewshot:
            return case
        return {"labeled_examples": fewshot, "case_to_classify": case}

    def _request_body(self, inp: GateInput) -> dict[str, Any]:
        """One multiple-choice question ("outcome") whose options are this phase's gate labels."""
        state: Any = self._plan_state(inp) if inp.phase == "plan" else render_state(inp, self.max_chars)
        return {
            "state": state,
            "model": self.model,
            "questions": {
                "outcome": {
                    "type": "choice",
                    "instructions": self._instructions(inp.phase),
                    "criteria": LABELS[inp.phase],
                }
            },
        }

    def _instructions(self, phase: str) -> str:
        if phase == "plan" and self.plan_view == "ir_text":
            return INSTRUCTIONS[phase].replace(IR_REF_SENTENCE, IR_TEXT_NOTE)
        return INSTRUCTIONS[phase]

    async def _fetch(self, body: dict[str, Any]) -> tuple[dict[str, Any], bool]:
        """POST ``body`` (or reuse a cached response). Returns ``(response_json, was_cached)``."""
        cache_key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key], True
        response = await self._http().post(self.url, json=body)
        response.raise_for_status()
        data = response.json()
        if self._cache is not None:
            self._cache[cache_key] = data
        return data, False

    @staticmethod
    def _extract_probs(data: dict[str, Any], phase: str) -> dict[str, float]:
        """Pull the label -> probability map for the "outcome" question out of a response."""
        # Answers normally sit under "answers"; fall back to the top level if that wrapper is missing.
        answers = data.get("answers") or data
        answer = answers["outcome"]
        probs = answer.get("probabilities")
        if not probs:  # some responses inline option probabilities in the answer object
            probs = {label: answer[label] for label in LABELS[phase] if label in answer}
        return probs

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        t0 = time.time()
        data, cached = await self._fetch(self._request_body(inp))
        # Usage lives under data["usage"], not top-level - the old code read the wrong
        # path and silently logged 0 tokens for every call.
        usage = data.get("usage") or {}
        emit(
            "gate_api",
            gate=self.name,
            model=data.get("model", self.model),
            input_tokens=usage.get("input_tokens", 0),
            output_tokens=usage.get("output_tokens", 0),
            cost_usd=usage.get("cost"),
            cached=cached,
            latency=time.time() - t0,
        )
        return self._extract_probs(data, inp.phase)
