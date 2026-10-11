"""NESTFUL: nested / dependent API call sequences (IBM, EMNLP 2025).

HF dataset ``ibm-research/nestful`` (split ``train``): fields ``sample_id``,
``input``, ``tools`` (JSON str), ``output`` (JSON str: calls with ``name``,
``arguments``, ``label``) and ``gold_answer``. References look like
``"$var_1.result$"``. Executable implementations live in the IBM/NESTFUL repo
under ``data_v2/executable_functions``; point ``executable_dir`` at a clone.
"""
from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import random
import re
from pathlib import Path
from typing import Any, Callable

from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.benchmarks.scoring import f1, unwrap, values_equal
from tpj.ir.schema import Plan, Step
from tpj.tools.mock import make_mock
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec

_REF = re.compile(r"^\$(var_?\d+)\.?(.*?)\$$")
_TYPES = {"int": "integer", "integer": "integer", "float": "number", "number": "number", "str": "string",
          "string": "string", "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
          "dict": "object", "object": "object"}


def _loads(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else v


def _gold_answer(v: Any) -> Any:
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (ValueError, SyntaxError):
        return v


def _schema(params: dict[str, Any] | None) -> dict[str, Any]:
    props = {}
    for name, p in (params or {}).items():
        p = p if isinstance(p, dict) else {}
        t = _TYPES.get(str(p.get("type", "")).lower())
        props[name] = {k: v for k, v in {"type": t, "description": p.get("description")}.items() if v}
    return {"type": "object", "properties": props, "required": list(props)}


def _spec(tool: dict[str, Any]) -> ToolSpec:
    out = tool.get("output_parameters")
    return ToolSpec(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=_schema(tool.get("parameters")),
        output_schema=_schema(out) if out else None,
    )


def gold_to_plan(query: str, calls: list[dict[str, Any]]) -> Plan:
    labels: dict[str, str] = {}
    steps = []
    for i, c in enumerate(calls, 1):
        sid = f"s{i}"
        label = str(c.get("label", f"$var_{i}")).strip("$")
        labels[label] = sid

        def conv(v: Any) -> Any:
            if isinstance(v, str):
                m = _REF.match(v)
                if m and m.group(1) in labels:
                    return {"$ref": labels[m.group(1)], "path": m.group(2)} if m.group(2) else {"$ref": labels[m.group(1)]}
            if isinstance(v, list):
                return [conv(x) for x in v]
            return v

        steps.append(Step(id=sid, tool=c["name"], args={k: conv(v) for k, v in (c.get("arguments") or {}).items()}))
    return Plan(query=query, steps=steps, final=[steps[-1].id] if steps else [])


def load_functions(directory: str | Path) -> dict[str, Callable[..., Any]]:
    fns: dict[str, Callable[..., Any]] = {}
    for path in sorted(Path(directory).rglob("*.py")):
        spec = importlib.util.spec_from_file_location(f"nestful_fn_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)
        except Exception:  # noqa: BLE001 - skip modules with missing deps
            continue
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if obj.__module__ == mod.__name__:
                fns.setdefault(name, obj)
    return fns


class NestfulBenchmark(Benchmark):
    name = "nestful"

    def __init__(self, hf_path: str = "ibm-research/nestful", split: str = "train", limit: int | None = None,
                 sample_seed: int | None = 0, executable_dir: str | None = None, executable_only: bool = False):
        self.hf_path = hf_path
        self.split = split
        self.limit = limit
        self.sample_seed = sample_seed
        self.executable_dir = executable_dir
        self.executable_only = executable_only
        self._fns: dict[str, Callable[..., Any]] | None = None

    def _functions(self) -> dict[str, Callable[..., Any]]:
        if self._fns is None:
            self._fns = load_functions(self.executable_dir) if self.executable_dir else {}
        return self._fns

    def load(self) -> list[Task]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, split=self.split)
        tasks = []
        fns = self._functions()
        for row in ds:
            tools = [_spec(t) for t in _loads(row["tools"])]
            calls = _loads(row["output"])
            executable = bool(fns) and all(c["name"] in fns for c in calls)
            if self.executable_only and not executable:
                continue
            tasks.append(Task(
                id=str(row["sample_id"]), query=row["input"], tools=tools,
                gold={"calls": calls, "answer": _gold_answer(row.get("gold_answer"))},
                gold_plan=gold_to_plan(row["input"], calls), meta={"executable": executable},
            ))
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        fns = self._functions()
        impls = {t.name: fns.get(t.name) or make_mock(t) for t in task.tools}
        return ToolRegistry(task.tools, impls)

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        gold_calls = task.gold["calls"]
        pred_names = [c.tool for c in result.trajectory() if c.error is None]
        names = f1(pred_names, [c["name"] for c in gold_calls])
        plan = result.plan
        seq_match = float(plan is not None and [s.tool for s in plan.steps] == [c["name"] for c in gold_calls])
        out = {"tool_f1": names["f1"], "full_sequence_match": seq_match}
        if task.meta.get("executable"):
            pred = unwrap(result.final_value())
            out["answer_match"] = float(values_equal(pred, unwrap(task.gold["answer"])))
            out["success"] = out["answer_match"]
        else:
            out["success"] = seq_match
        return out
