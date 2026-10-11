"""JEV (TypeSafe AI System One model) gate over the HTTP API.

API (docs.typesafe.ai): POST {base_url}/systemone with
``{"state": str, "model": str, "questions": {id: {"type": "choice", "instructions": str,
"criteria": {option: description}}}}``. Each choice answer carries ``choice``,
``confidence`` and ``probabilities``.
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any

from tpj.eval.trace import emit
from tpj.gate.base import INSTRUCTIONS, LABELS, Gate, GateInput, render_state


class JEVGate(Gate):
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
        if self._client is None:
            import httpx

            key = os.environ.get(self.api_key_env)
            if not key:
                raise RuntimeError(f"set {self.api_key_env} to use the JEV gate")
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            )
        return self._client

    async def _classify(self, inp: GateInput) -> dict[str, float]:
        body = {
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
        key = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        cached = self._cache is not None and key in self._cache
        if cached:
            data = self._cache[key]
        else:
            resp = await self._http().post(self.url, json=body)
            resp.raise_for_status()
            data = resp.json()
            if self._cache is not None:
                self._cache[key] = data
        emit(
            "gate_api",
            gate=self.name,
            model=data.get("model", self.model),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            cached=cached,
        )
        answer = (data.get("answers") or data)["outcome"]
        probs = answer.get("probabilities")
        if not probs:  # some responses inline option probabilities in the answer object
            probs = {k: answer[k] for k in LABELS[inp.phase] if k in answer}
        return probs
