"""BFCL (Berkeley Function Calling Leaderboard), single-turn categories.

Reads a local clone of ``gorilla/berkeley-function-call-leaderboard/bfcl_eval/data``:
``BFCL_{version}_{category}.json`` (JSON lines: ``id``, ``question``, ``function``)
and ``possible_answer/BFCL_{version}_{category}.json`` (``id``, ``ground_truth``:
``[{func: {param: [allowed values]}}]``). Scoring follows BFCL's AST matching:
every gold call must be matched by one predicted call with the same function
and allowed parameter values; ``""`` in the allowed list marks an optional param.

Illustrative lines (made-up values; the shape is what the parsing below expects)::

    {"id": "simple_python_0",
     "question": [[{"role": "user", "content": "Area of a triangle with base 10 and height 5?"}]],
     "function": [{"name": "math.triangle_area", "description": "...",
                   "parameters": {"type": "dict", "properties": {"base": {"type": "integer"}, ...}}}]}
    {"id": "simple_python_0",
     "ground_truth": [{"math.triangle_area": {"base": [10], "height": [5], "unit": ["", "cm"]}}]}

Multi-turn categories need BFCL's stateful executable backends and are not
wired up here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tpj.benchmarks.base import AgentResult, Benchmark, CallRecord, Task
from tpj.benchmarks.scoring import values_equal
from tpj.tools.mock import make_mock
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec

# BFCL writes Python type names ("dict", "float", ...) in its schemas; map them to JSON-schema types.
_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array",
             "int": "integer", "bool": "boolean", "str": "string"}


def _fix_schema(node: Any) -> Any:
    """Recursively rewrite every ``"type"`` string in a BFCL schema to its JSON-schema name."""
    if isinstance(node, dict):
        fixed = {key: _fix_schema(value) for key, value in node.items()}
        if isinstance(fixed.get("type"), str):
            fixed["type"] = _TYPE_MAP.get(fixed["type"], fixed["type"])
        return fixed
    if isinstance(node, list):
        return [_fix_schema(item) for item in node]
    return node


def _safe_name(name: str) -> str:
    """BFCL names like ``math.hypot`` are not valid tool names for most LLM APIs; use ``math_hypot``."""
    return name.replace(".", "_")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _user_query(question: list[Any]) -> str:
    """Join the user messages of the first turn.

    ``question`` is either a list of turns (``[[msg, ...], ...]``) or, in older files, a flat list of messages.
    """
    if question and isinstance(question[0], list):
        first_turn = question[0]
    else:
        first_turn = question
    return "\n".join(msg["content"] for msg in first_turn if msg.get("role") == "user")


def _tool_spec(function: dict[str, Any]) -> ToolSpec:
    return ToolSpec(name=_safe_name(function["name"]), description=function.get("description", ""),
                    parameters=_fix_schema(function.get("parameters", {"type": "object", "properties": {}})))


class BFCLBenchmark(Benchmark):
    name = "bfcl"

    def __init__(self, data_dir: str, categories: list[str] | None = None, version: str = "v4",
                 limit: int | None = None, sample_seed: int | None = 0):
        self.data_dir = Path(data_dir)
        self.categories = categories or ["simple_python", "multiple", "parallel", "parallel_multiple"]
        self.version = version
        self.limit = limit
        self.sample_seed = sample_seed

    def _load_answers(self, category: str) -> dict[str, list]:
        """``id -> ground_truth``; empty if the category has no answer file (e.g. irrelevance)."""
        answer_path = self.data_dir / "possible_answer" / f"BFCL_{self.version}_{category}.json"
        if not answer_path.exists():
            return {}
        return {answer["id"]: answer["ground_truth"] for answer in _read_jsonl(answer_path)}

    def load(self) -> list[Task]:
        tasks = []
        for category in self.categories:
            rows = _read_jsonl(self.data_dir / f"BFCL_{self.version}_{category}.json")
            answers = self._load_answers(category)
            for row in rows:
                tools = [_tool_spec(function) for function in row["function"]]
                # "irrelevance" categories: none of the given tools fit, so the right answer is no call at all.
                gold = {"calls": answers.get(row["id"], []), "irrelevance": "irrelevance" in category}
                tasks.append(Task(id=row["id"], query=_user_query(row["question"]), tools=tools,
                                  gold=gold, meta={"category": category}))
        # Shuffle with a fixed seed so ``limit`` takes the same random subset every run.
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        return ToolRegistry(task.tools, {tool.name: make_mock(tool) for tool in task.tools})

    @staticmethod
    def _matches(pred_args: dict[str, Any], allowed: dict[str, list[Any]]) -> bool:
        """True if the predicted args fit one gold call's ``{param: [allowed values]}``.

        No unknown params; every param must be present unless ``""`` is among its allowed values
        (BFCL's marker for "optional"); a present param must equal one of the non-``""`` values.
        """
        if set(pred_args) - set(allowed):
            return False
        for param, values in allowed.items():
            if param not in pred_args:
                if "" not in values:
                    return False
                continue
            if not any(values_equal(pred_args[param], value) for value in values if value != ""):
                return False
        return True

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        """Greedy one-to-one matching of gold calls to predicted calls; success needs no extras either."""
        preds = result.trajectory()
        if task.gold.get("irrelevance"):
            return {"success": float(not preds), "ast_match": float(not preds)}

        gold_calls = task.gold["calls"]
        unmatched: list[CallRecord] = list(preds)
        hits = 0
        for gold_call in gold_calls:
            (func_name, allowed), = gold_call.items()  # each gold call is a one-key dict {func: {param: [...]}}
            for pred in unmatched:
                if pred.tool == _safe_name(func_name) and self._matches(pred.args, allowed):
                    unmatched.remove(pred)  # each predicted call can match only one gold call
                    hits += 1
                    break
        ok = hits == len(gold_calls) and not unmatched
        call_recall = hits / len(gold_calls) if gold_calls else 1.0
        return {"success": float(ok), "ast_match": float(ok), "call_recall": call_recall}
