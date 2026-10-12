"""Small client for the hosted Jev System One API."""
from __future__ import annotations

import os
import time
import uuid
from dataclasses import dataclass
from typing import Any

import requests

API_URL = "https://jevmodel.org/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"

@dataclass
class JevResult:
    label: str
    probabilities: dict[str, float]
    confidence: float | None
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    latency_seconds: float
    raw_response: dict[str, Any]

class JevClient:
    def __init__(self, api_key: str | None = None, model: str = DEFAULT_MODEL,
                 timeout: int = 45, max_retries: int = 2):
        self.api_key = api_key or os.getenv("JEVMODEL_API_KEY") or os.getenv("TYPESAFE_API_KEY")
        if not self.api_key:
            raise RuntimeError(
                "Set JEVMODEL_API_KEY in your environment. "
                "TYPESAFE_API_KEY is also accepted for your current terminal setup."
            )
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def classify(self, state: dict[str, Any], criteria: dict[str, str],
                 instructions: str, experiment_id: str) -> JevResult:
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "outcome": {
                    "type": "choice",
                    "instructions": instructions,
                    "criteria": criteria,
                }
            },
        }
        # The API documents a maximum serialized state of 8,000 characters.
        serialized_state = __import__("json").dumps(
            state, ensure_ascii=False, separators=(",", ":")
        )
        if len(serialized_state) > 8000:
            raise ValueError(
                f"Serialized state is {len(serialized_state)} characters; "
                "Jev's documented state limit is 8,000 characters."
            )

        headers = {"Idempotency-Key": f"{experiment_id}-{uuid.uuid4().hex[:12]}"}
        started = time.perf_counter()
        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self.session.post(
                    API_URL, json=payload, headers=headers, timeout=self.timeout
                )
                if response.status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(1.5 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                answer = data["answers"]["outcome"]
                label = answer["choice"]
                probs = answer.get("probabilities", {})
                if label not in criteria:
                    raise ValueError(f"Jev returned unexpected label: {label!r}")
                return JevResult(
                    label=label,
                    probabilities={str(k): float(v) for k, v in probs.items()},
                    confidence=answer.get("confidence"),
                    model=data.get("model"),
                    input_tokens=(data.get("usage") or {}).get("input_tokens"),
                    output_tokens=(data.get("usage") or {}).get("output_tokens"),
                    latency_seconds=round(time.perf_counter() - started, 4),
                    raw_response=data,
                )
            except (requests.RequestException, ValueError, KeyError) as exc:
                last_error = exc
                # Retry only transient HTTP failures; do not retry schema/auth errors.
                if isinstance(exc, requests.HTTPError):
                    status = exc.response.status_code if exc.response is not None else None
                    if status not in (429, 502, 503, 504):
                        raise
                if attempt >= self.max_retries:
                    raise
                time.sleep(1.5 ** attempt)
        raise RuntimeError(f"Jev request failed: {last_error}")
