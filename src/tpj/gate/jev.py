"""JEV (TypeSafe AI System One model) gate over the HTTP API.

API (docs.typesafe.ai): POST {base_url}/systemone with
``{"state": str, "model": str, "questions": {id: {"type": "choice", "instructions": str,
"criteria": {option: description}}}}``. Each choice answer carries ``choice``,
``confidence`` and ``probabilities``.

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
from typing import Any

from tpj.eval.trace import emit
from tpj.gate.base import INSTRUCTIONS, LABELS, Gate, GateInput, render_state


class JEVGate(Gate):
    """Gate backed by the hosted JEV classifier. Responses are cached on disk keyed by the request body."""

    name = "jev"

    def __init__(
        self,
        model: str = "jev-latest",
        base_url: str = "https://api.typesafe.ai/v1",
        api_key_env: str = "TYPESAFE_API_KEY",
        timeout: float = 30.0,
        cache_dir: str | None = ".cache/jev",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.url = base_url.rstrip("/") + "/systemone"
        self.api_key_env = api_key_env
        self.timeout = timeout
        self._client = None
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(cache_dir)

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

    def _request_body(self, inp: GateInput) -> dict[str, Any]:
        """One multiple-choice question ("outcome") whose options are this phase's gate labels."""
        return {
            "state": render_state(inp, self.max_chars),
            "model": self.model,
            "questions": {
                "outcome": {
                    "type": "choice",
                    "instructions": INSTRUCTIONS[inp.phase],
                    "criteria": LABELS[inp.phase],
                }
            },
        }

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
        data, cached = await self._fetch(self._request_body(inp))
        emit(
            "gate_api",
            gate=self.name,
            model=data.get("model", self.model),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached=cached,
        )
        return self._extract_probs(data, inp.phase)
