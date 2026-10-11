"""TaskBench (NeurIPS 2024): tool-graph planning.

HF dataset ``microsoft/Taskbench``, subsets ``dailylifeapis`` / ``huggingface`` /
``multimedia``, split ``test``. Fields used: ``id``, ``instruction``, ``type``
(single/chain/dag), ``sampled_nodes`` (tools with argument specs),
``tool_nodes`` (gold calls) and ``tool_links`` (gold edges, by tool name).
There are no executable tools, so execution uses deterministic mocks and the
metrics are structural (node / edge / argument F1).
"""
from __future__ import annotations

import json
import random
import re
from typing import Any

from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.benchmarks.scoring import f1
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan, Step
from tpj.tools.mock import make_mock
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec

_NODE_REF = re.compile(r"^<node-(\d+)>$")


def _loads(v: Any) -> Any:
    return json.loads(v) if isinstance(v, str) else (v or [])


def _arg_items(arguments: Any) -> list[tuple[str, Any]]:
    """Arguments are either ``[{"name", "value"}]`` or positional ``["x.jpg", ...]``."""
    items = []
    for i, a in enumerate(arguments or []):
        if isinstance(a, dict):
            items.append((a.get("name", f"arg{i}"), a.get("value", a.get("desc"))))
        else:
            items.append((f"arg{i}", a))
    return items


def _spec_from_node(node: dict[str, Any]) -> ToolSpec:
    props = {}
    for i, a in enumerate(node.get("arguments") or []):
        if isinstance(a, dict):
            props[a.get("name", f"arg{i}")] = {"type": a.get("type", "string"), "description": a.get("desc", "")}
        else:
            props[f"arg{i}"] = {"type": "string", "description": str(a)}
    for p in props.values():
        if p["type"] not in {"string", "integer", "number", "boolean", "array", "object"}:
            p["description"] = f"({p['type']}) {p['description']}".strip()
            p["type"] = "string"
    return ToolSpec(name=node["task"], description=node.get("desc", node["task"]),
                    parameters={"type": "object", "properties": props, "required": list(props)})


def gold_plan(query: str, nodes: list[dict], links: list[dict]) -> Plan:
    steps = []
    for i, n in enumerate(nodes):
        args = {}
        for name, value in _arg_items(n.get("arguments")):
            m = _NODE_REF.match(value) if isinstance(value, str) else None
            args[name] = {"$ref": f"s{int(m.group(1)) + 1}"} if m else value
        steps.append(Step(id=f"s{i + 1}", tool=n["task"], args=args))
    by_tool = {s.tool: s.id for s in steps}
    for link in links:
        src, tgt = by_tool.get(link["source"]), by_tool.get(link["target"])
        if src and tgt:
            step = next(s for s in steps if s.id == tgt)
            if src not in step.deps():
                step.after.append(src)
    used = {d for s in steps for d in s.deps()}
    return Plan(query=query, steps=steps, final=[s.id for s in steps if s.id not in used])


class TaskBenchBenchmark(Benchmark):
    name = "taskbench"

    def __init__(self, hf_path: str = "microsoft/Taskbench", subset: str = "dailylifeapis", split: str = "test",
                 types: list[str] | None = None, limit: int | None = None, sample_seed: int | None = 0):
        self.hf_path = hf_path
        self.subset = subset
        self.split = split
        self.types = set(types) if types else None
        self.limit = limit
        self.sample_seed = sample_seed
        self.catalog: list[ToolSpec] = []

    def load(self) -> list[Task]:
        from datasets import load_dataset

        ds = load_dataset(self.hf_path, self.subset, split=self.split)
        catalog: dict[str, ToolSpec] = {}
        for row in ds:  # full tool catalogue = union of sampled tools
            for n in _loads(row["sampled_nodes"]):
                catalog.setdefault(n["task"], _spec_from_node(n))
        self.catalog = list(catalog.values())
        tasks = []
        for row in ds:
            if self.types and row["type"] not in self.types:
                continue
            nodes, links = _loads(row["tool_nodes"]), _loads(row["tool_links"])
            tasks.append(Task(
                id=str(row["id"]), query=row["instruction"], tools=self.catalog,
                gold={"nodes": nodes, "links": links},
                gold_plan=gold_plan(row["instruction"], nodes, links),
                meta={"type": row["type"], "n_tools": row["n_tools"]},
            ))
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        return ToolRegistry(task.tools, {t.name: make_mock(t) for t in task.tools})

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        gold_nodes = [n["task"] for n in task.gold["nodes"]]
        gold_edges = [(link["source"], link["target"]) for link in task.gold["links"]]
        gold_args = [(n["task"], k, str(v)) for n in task.gold["nodes"] for k, v in _arg_items(n.get("arguments"))
                     if not (isinstance(v, str) and _NODE_REF.match(v))]
        plan = result.plan or result.initial_plan
        if plan is not None:
            tool_of = {s.id: s.tool for s in plan.steps}
            pred_nodes = [s.tool for s in plan.steps]
            pred_edges = [(tool_of[d], s.tool) for s in plan.steps for d in s.deps() if d in tool_of]
            pred_args = [(s.tool, k, str(v)) for s in plan.steps for k, v in s.args.items() if not is_ref(v)]
        else:  # ReAct baseline: no explicit structure
            pred_nodes = [c.tool for c in result.calls]
            pred_edges, pred_args = [], [(c.tool, k, str(v)) for c in result.calls for k, v in c.args.items()]
        node, edge, arg = f1(pred_nodes, gold_nodes), f1(pred_edges, gold_edges), f1(pred_args, gold_args)
        return {"node_f1": node["f1"], "edge_f1": edge["f1"], "arg_f1": arg["f1"],
                "success": float(node["f1"] == 1.0 and edge["f1"] == 1.0)}
