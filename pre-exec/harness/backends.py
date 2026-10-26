"""Decision-model backends. To add a model: subclass Backend, decorate with @register("name"),
then reference it from experiments.toml (`backend = "name"`, extra keys go under `params`)."""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
BACKENDS: dict[str, type["Backend"]] = {}


def register(name: str):
    def deco(cls):
        cls.backend_name = name
        BACKENDS[name] = cls
        return cls
    return deco


@dataclass
class Prediction:
    label: str
    probabilities: dict[str, float] = field(default_factory=dict)
    confidence: float | None = None
    latency_seconds: float | None = None
    input_tokens: int | None = None
    cost_usd: float | None = None


class Backend:
    """One decision model. `classify` answers a single choice question over `labels`."""
    max_workers = 1  # >1 only for remote APIs; local models run sequentially

    def __init__(self, **params: Any):
        self.params = params

    def setup(self) -> None:
        """Heavy initialisation (load weights, open sessions). Skipped on dry runs."""

    def classify(self, state: dict, labels: dict[str, str], instructions: str,
                 experiment_id: str) -> Prediction:
        raise NotImplementedError


@register("jev_openrouter")
class JevOpenRouterBackend(Backend):
    """JEV through OpenRouter's Decisions API (same typed-question model, different endpoint).
    Needs OPENROUTER_API_KEY in the environment. Model slugs: typesafe/jev-1.13 (matches our earlier
    direct-API runs), typesafe/jev-1.14, ~typesafe/jev-latest. The state limit is far larger than the
    direct API's 8,000 chars (OpenRouter documents 32,000 bytes before it switches to a slower,
    less accurate long-context mode), so keep `max_chars` under that."""

    URL = "https://openrouter.ai/api/alpha/decisions"

    def __init__(self, model: str = "typesafe/jev-1.13", workers: int = 4, timeout: int = 90,
                 max_retries: int = 3, **kw):
        super().__init__(model=model, workers=workers, timeout=timeout, max_retries=max_retries, **kw)
        self.max_workers = workers

    def setup(self):
        import requests
        key = os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("Set OPENROUTER_API_KEY in your environment (never put keys in the config).")
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {key}", "Content-Type": "application/json"})

    def classify(self, state, labels, instructions, experiment_id):
        import requests
        p = self.params
        payload = {"model": p["model"], "state": state,
                   "questions": {"outcome": {"type": "choice", "instructions": instructions,
                                             "criteria": labels}}}
        t0 = time.perf_counter()
        for attempt in range(p["max_retries"] + 1):
            try:
                r = self.session.post(self.URL, json=payload, timeout=p["timeout"])
                if r.status_code in (429, 502, 503, 504) and attempt < p["max_retries"]:
                    time.sleep(1.5 ** attempt)
                    continue
                if not r.ok:   # surface OpenRouter's message (402 = no credits, 401 = bad key)
                    raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
                data = r.json()
                ans = data["answers"]["outcome"]
                if ans["choice"] not in labels:
                    raise ValueError(f"unexpected label {ans['choice']!r}")
                u = data.get("usage") or {}
                return Prediction(ans["choice"], {k: float(v) for k, v in ans.get("probabilities", {}).items()},
                                  ans.get("confidence"), round(time.perf_counter() - t0, 4),
                                  u.get("input_tokens"), u.get("cost"))
            except requests.RequestException:
                if attempt >= p["max_retries"]:
                    raise
                time.sleep(1.5 ** attempt)


@register("laya")
class LayaBackend(Backend):
    """Local Laya checkpoint (pip install laya). ckpt: english | multilingual | typed-decisions.
    max_len / head_max_len override the token budgets (English default 512 truncates our plans)."""

    SUBFOLDER = {"english": None, "multilingual": "multilingual", "typed-decisions": "typed-decisions"}

    def __init__(self, ckpt: str = "english", max_len: int | None = None,
                 head_max_len: int | None = None, device: str = "cpu",
                 repo: str = "convaiinnovations/laya", **kw):
        if ckpt not in self.SUBFOLDER:
            raise ValueError(f"laya ckpt must be one of {list(self.SUBFOLDER)}, got {ckpt!r}")
        super().__init__(ckpt=ckpt, max_len=max_len, head_max_len=head_max_len,
                         device=device, repo=repo, **kw)

    def setup(self):
        os.environ.setdefault("USE_TF", "0")  # model card: avoids a TF/abseil import deadlock
        import laya
        p = self.params
        kw = {"subfolder": self.SUBFOLDER[p["ckpt"]]} if self.SUBFOLDER[p["ckpt"]] else {}
        if p["device"] != "cpu":
            kw["device"] = p["device"]
        self.agent = laya.load(p["repo"], **kw)
        if p["max_len"]:
            self.agent.cfg["max_len"] = p["max_len"]
        if p["head_max_len"]:
            self.agent.cfg["head_max_len"] = p["head_max_len"]
        print(f"  laya {p['ckpt']}: max_len={self.agent.cfg.get('max_len')} "
              f"head_max_len={self.agent.cfg.get('head_max_len')}")

    def classify(self, state, labels, instructions, experiment_id):
        q = {"outcome": {"type": "choice", "instructions": instructions, "criteria": labels}}
        t0 = time.perf_counter()
        ans = self.agent.predict(state, q)["answers"]["outcome"]
        return Prediction(ans["choice"], dict(ans.get("probabilities", {})), ans.get("confidence"),
                          round(time.perf_counter() - t0, 4))


@register("rules")
class RulesBackend(Backend):
    """No-model baseline: deterministic schema/reference validator. Plan-verification task only."""

    def classify(self, state, labels, instructions, experiment_id):
        sys.path.insert(0, str(ROOT))
        from build_pre_cases import validate
        t0 = time.perf_counter()
        bad = bool(validate(state["plan"]["steps"], state["available_tools"]))
        return Prediction("INVALID" if bad else "VALID",
                          {"INVALID": float(bad), "VALID": float(not bad)}, 1.0,
                          round(time.perf_counter() - t0, 6))
