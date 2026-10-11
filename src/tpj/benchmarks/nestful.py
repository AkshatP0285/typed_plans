"""NESTFUL: nested / dependent API call sequences (IBM, EMNLP 2025).

HF dataset ``ibm-research/nestful`` (split ``train``): fields ``sample_id``,
``input``, ``tools`` (JSON str), ``output`` (JSON str: calls with ``name``,
``arguments``, ``label``) and ``gold_answer``. References look like
``"$var_1.result$"``. Executable implementations live in the IBM/NESTFUL repo
under ``data_v2/executable_functions``; point ``executable_dir`` at a clone.

Illustrative row (made-up values; the shape is what the parsing below expects)::

    {"sample_id": "42",
     "input": "What is (3 + 4) * 2?",
     "tools": '[{"name": "add", "description": "...",
                 "parameters": {"arg_0": {"type": "int", "description": "..."}, ...},
                 "output_parameters": {"result": {"type": "int"}}}, ...]',
     "output": '[{"name": "add", "label": "$var_1", "arguments": {"arg_0": 3, "arg_1": 4}},
                 {"name": "multiply", "label": "$var_2", "arguments": {"arg_0": "$var_1.result$", "arg_1": 2}}]',
     "gold_answer": "14"}
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

# Matches a NESTFUL reference string: "$var_1.result$" -> ("var_1", "result"); "$var2$" -> ("var2", "").
_REF = re.compile(r"^\$(var_?\d+)\.?(.*?)\$$")
# NESTFUL uses Python-ish type names; map them to JSON-schema types.
_TYPES = {"int": "integer", "integer": "integer", "float": "number", "number": "number", "str": "string",
          "string": "string", "bool": "boolean", "boolean": "boolean", "list": "array", "array": "array",
          "dict": "object", "object": "object"}


def _loads(value: Any) -> Any:
    """Some fields are stored as JSON strings, others already parsed."""
    return json.loads(value) if isinstance(value, str) else value


def _gold_answer(value: Any) -> Any:
    """Gold answers are Python literals stored as strings (``"14"``, ``"[1, 2]"``); keep the raw string otherwise."""
    if not isinstance(value, str):
        return value
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


def _schema(params: dict[str, Any] | None) -> dict[str, Any]:
    """``{"arg_0": {"type": "int", "description": "..."}}`` -> JSON-schema object with every param required."""
    props = {}
    for name, param in (params or {}).items():
        if not isinstance(param, dict):
            param = {}
        json_type = _TYPES.get(str(param.get("type", "")).lower())
        prop = {}
        # Leave out unknown types / empty descriptions rather than writing None.
        if json_type:
            prop["type"] = json_type
        if param.get("description"):
            prop["description"] = param.get("description")
        props[name] = prop
    return {"type": "object", "properties": props, "required": list(props)}


def _spec(tool: dict[str, Any]) -> ToolSpec:
    output_params = tool.get("output_parameters")
    return ToolSpec(
        name=tool["name"],
        description=tool.get("description", ""),
        parameters=_schema(tool.get("parameters")),
        output_schema=_schema(output_params) if output_params else None,
    )


def _convert_refs(value: Any, label_to_step: dict[str, str]) -> Any:
    """Turn ``"$var_1.result$"`` into our IR ref ``{"$ref": "s1", "path": "result"}`` (recursing into lists).

    Strings that look like refs but name an unknown label are left unchanged.
    """
    if isinstance(value, str):
        match = _REF.match(value)
        if match and match.group(1) in label_to_step:
            label, path = match.group(1), match.group(2)
            if path:
                return {"$ref": label_to_step[label], "path": path}
            return {"$ref": label_to_step[label]}
    if isinstance(value, list):
        return [_convert_refs(item, label_to_step) for item in value]
    return value


def gold_to_plan(query: str, calls: list[dict[str, Any]]) -> Plan:
    """Convert the gold call sequence into a Plan with steps ``s1, s2, ...``; the last step is the final one."""
    label_to_step: dict[str, str] = {}  # "var_1" -> "s1"
    steps = []
    for index, call in enumerate(calls, 1):
        step_id = f"s{index}"
        label = str(call.get("label", f"$var_{index}")).strip("$")
        label_to_step[label] = step_id
        args = {name: _convert_refs(value, label_to_step) for name, value in (call.get("arguments") or {}).items()}
        steps.append(Step(id=step_id, tool=call["name"], args=args))
    final = [steps[-1].id] if steps else []
    return Plan(query=query, steps=steps, final=final)


def load_functions(directory: str | Path) -> dict[str, Callable[..., Any]]:
    """Import every ``*.py`` under ``directory`` and collect its top-level functions by name.

    Modules that fail to import (e.g. a missing third-party dependency) are skipped; if two
    modules define the same name, the first one (in sorted path order) wins.
    """
    functions: dict[str, Callable[..., Any]] = {}
    for path in sorted(Path(directory).rglob("*.py")):
        spec = importlib.util.spec_from_file_location(f"nestful_fn_{path.stem}", path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception:  # noqa: BLE001 - skip modules with missing deps
            continue
        for name, obj in inspect.getmembers(module, inspect.isfunction):
            if obj.__module__ == module.__name__:  # skip functions the module merely imported
                functions.setdefault(name, obj)
    return functions


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
        """Executable tool implementations, loaded once (empty if no ``executable_dir``)."""
        if self._fns is None:
            self._fns = load_functions(self.executable_dir) if self.executable_dir else {}
        return self._fns

    def load(self) -> list[Task]:
        from datasets import load_dataset

        dataset = load_dataset(self.hf_path, split=self.split)
        functions = self._functions()
        tasks = []
        for row in dataset:
            tools = [_spec(tool) for tool in _loads(row["tools"])]
            calls = _loads(row["output"])
            # A task is executable only if we have a real implementation for every gold call.
            executable = bool(functions) and all(call["name"] in functions for call in calls)
            if self.executable_only and not executable:
                continue
            tasks.append(Task(
                id=str(row["sample_id"]), query=row["input"], tools=tools,
                gold={"calls": calls, "answer": _gold_answer(row.get("gold_answer"))},
                gold_plan=gold_to_plan(row["input"], calls), meta={"executable": executable},
            ))
        # Shuffle with a fixed seed so ``limit`` takes the same random subset every run.
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        """Real implementations where available, deterministic mocks for the rest."""
        functions = self._functions()
        impls = {tool.name: functions.get(tool.name) or make_mock(tool) for tool in task.tools}
        return ToolRegistry(task.tools, impls)

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        """Executable tasks: success = final answer matches. Otherwise: success = exact tool sequence."""
        gold_names = [call["name"] for call in task.gold["calls"]]
        pred_names = [call.tool for call in result.trajectory() if call.error is None]
        name_scores = f1(pred_names, gold_names)

        plan = result.plan
        seq_match = float(plan is not None and [step.tool for step in plan.steps] == gold_names)

        scores = {"tool_f1": name_scores["f1"], "full_sequence_match": seq_match}
        if task.meta.get("executable"):
            pred = unwrap(result.final_value())
            scores["answer_match"] = float(values_equal(pred, unwrap(task.gold["answer"])))
            scores["success"] = scores["answer_match"]
        else:
            scores["success"] = seq_match
        return scores
