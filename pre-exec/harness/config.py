"""Loads experiments.toml (TOML, stdlib tomllib) and merges defaults into each experiment."""
from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .backends import BACKENDS
from .tasks import TASKS


@dataclass
class Prompt:
    name: str
    instructions: str
    labels: dict[str, str]


@dataclass
class Experiment:
    name: str
    backend: str
    params: dict[str, Any]
    prompt: Prompt
    task: str
    data: Path
    out: Path
    enabled: bool = True
    state_opts: dict[str, Any] = field(default_factory=dict)


@dataclass
class Config:
    root: Path
    experiments: list[Experiment] = field(default_factory=list)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    root = path.parent
    d = raw.get("defaults", {})
    prompts = {n: Prompt(n, p["instructions"], p["labels"]) for n, p in raw.get("prompts", {}).items()}
    cfg, seen = Config(root=root), set()
    for e in raw.get("experiments", []):
        name = e["name"]
        if name in seen:
            raise ValueError(f"duplicate experiment name: {name}")
        seen.add(name)
        if e["backend"] not in BACKENDS:
            raise ValueError(f"{name}: unknown backend {e['backend']!r} (have {sorted(BACKENDS)})")
        task = e.get("task", d.get("task", "pre_exec_plan"))
        if task not in TASKS:
            raise ValueError(f"{name}: unknown task {task!r} (have {sorted(TASKS)})")
        pname = e.get("prompt", d.get("prompt", "default"))
        if pname not in prompts:
            raise ValueError(f"{name}: unknown prompt {pname!r} (have {sorted(prompts)})")
        results_dir = d.get("results_dir", "results")
        cfg.experiments.append(Experiment(
            name=name, backend=e["backend"], params=e.get("params", {}), prompt=prompts[pname],
            task=task, data=root / e.get("data", d.get("data", "test_set/pre_exec_test_set.jsonl")),
            out=root / e.get("out", f"{results_dir}/{name}.jsonl"),
            enabled=e.get("enabled", True), state_opts=e.get("state", {}),
        ))
    return cfg


def select(cfg: Config, only: str | None, include_disabled: bool = False) -> list[Experiment]:
    names = [n.strip() for n in only.split(",")] if only else None
    if names:
        missing = set(names) - {e.name for e in cfg.experiments}
        if missing:
            raise SystemExit(f"unknown experiment(s): {sorted(missing)}")
        return [e for e in cfg.experiments if e.name in names]   # explicit names override `enabled`
    return [e for e in cfg.experiments if e.enabled or include_disabled]
