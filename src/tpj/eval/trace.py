"""Per-task event tracing.

Components call :func:`emit` without passing a tracer around; the active tracer
lives in a context variable that is set per task (and copied into asyncio
sub-tasks automatically). Every metric is computed from these events.
"""
from __future__ import annotations

import contextvars
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_current: contextvars.ContextVar["Tracer | None"] = contextvars.ContextVar("tpj_tracer", default=None)


@dataclass
class Tracer:
    task_id: str
    events: list[dict[str, Any]] = field(default_factory=list)

    def emit(self, type_: str, **data: Any) -> None:
        self.events.append({"task_id": self.task_id, "t": time.time(), "type": type_, **data})

    def of_type(self, type_: str) -> list[dict[str, Any]]:
        return [e for e in self.events if e["type"] == type_]


def emit(type_: str, **data: Any) -> None:
    tracer = _current.get()
    if tracer is not None:
        tracer.emit(type_, **data)


def current() -> Tracer | None:
    return _current.get()


@contextmanager
def tracing(task_id: str):
    tracer = Tracer(task_id)
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


@contextmanager
def activate(tracer: Tracer):
    token = _current.set(tracer)
    try:
        yield tracer
    finally:
        _current.reset(token)


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, default=str, ensure_ascii=False) + "\n")
