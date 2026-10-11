"""BFCL (Berkeley Function Calling Leaderboard), single-turn categories.

Reads a local clone of ``gorilla/berkeley-function-call-leaderboard/bfcl_eval/data``:
``BFCL_{version}_{category}.json`` (JSON lines: ``id``, ``question``, ``function``)
and ``possible_answer/BFCL_{version}_{category}.json`` (``id``, ``ground_truth``:
``[{func: {param: [allowed values]}}]``). Scoring follows BFCL's AST matching:
every gold call must be matched by one predicted call with the same function
and allowed parameter values; ``""`` in the allowed list marks an optional param.

Multi-turn categories need BFCL's stateful executable backends and are not
wired up here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.benchmarks.scoring import values_equal
from tpj.tools.mock import make_mock
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec

_TYPE_MAP = {"dict": "object", "float": "number", "tuple": "array", "any": "string", "list": "array",
             "int": "integer", "bool": "boolean", "str": "string"}


def _fix_schema(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _fix_schema(v) for k, v in node.items()}
        if isinstance(out.get("type"), str):
            out["type"] = _TYPE_MAP.get(out["type"], out["type"])
        return out
    if isinstance(node, list):
        return [_fix_schema(v) for v in node]
    return node


def _safe_name(name: str) -> str:
    return name.replace(".", "_")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


class BFCLBenchmark(Benchmark):
    name = "bfcl"

    def __init__(self, data_dir: str, categories: list[str] | None = None, version: str = "v4",
                 limit: int | None = None, sample_seed: int | None = 0):
        self.data_dir = Path(data_dir)
        self.categories = categories or ["simple_python", "multiple", "parallel", "parallel_multiple"]
        self.version = version
        self.limit = limit
        self.sample_seed = sample_seed

    def load(self) -> list[Task]:
        tasks = []
        for cat in self.categories:
            rows = _read_jsonl(self.data_dir / f"BFCL_{self.version}_{cat}.json")
            ans_path = self.data_dir / "possible_answer" / f"BFCL_{self.version}_{cat}.json"
            answers = {a["id"]: a["ground_truth"] for a in _read_jsonl(ans_path)} if ans_path.exists() else {}
            for r in rows:
                turn = r["question"][0] if r["question"] and isinstance(r["question"][0], list) else r["question"]
                query = "\n".join(m["content"] for m in turn if m.get("role") == "user")
                tools = [ToolSpec(name=_safe_name(f["name"]), description=f.get("description", ""),
                                  parameters=_fix_schema(f.get("parameters", {"type": "object", "properties": {}})))
                         for f in r["function"]]
                gold = answers.get(r["id"], [])
                tasks.append(Task(id=r["id"], query=query, tools=tools,
                                  gold={"calls": gold, "irrelevance": "irrelevance" in cat}, meta={"category": cat}))
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        return ToolRegistry(task.tools, {t.name: make_mock(t) for t in task.tools})

    @staticmethod
    def _matches(pred_args: dict[str, Any], allowed: dict[str, list[Any]]) -> bool:
        if set(pred_args) - set(allowed):
            return False
        for param, values in allowed.items():
            if param not in pred_args:
                if "" not in values:
                    return False
                continue
            if not any(values_equal(pred_args[param], v) for v in values if v != ""):
                return False
        return True

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        preds = result.trajectory()
        if task.gold.get("irrelevance"):
            return {"success": float(not preds), "ast_match": float(not preds)}
        gold = task.gold["calls"]
        unmatched = list(preds)
        hits = 0
        for g in gold:
            (fname, allowed), = g.items()
            for p in unmatched:
                if p.tool == _safe_name(fname) and self._matches(p.args, allowed):
                    unmatched.remove(p)
                    hits += 1
                    break
        ok = hits == len(gold) and not unmatched
        return {"success": float(ok), "ast_match": float(ok), "call_recall": hits / len(gold) if gold else 1.0}
