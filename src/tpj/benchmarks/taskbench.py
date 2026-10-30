"""TaskBench (NeurIPS 2024): tool-graph planning.

HF dataset ``microsoft/Taskbench``, subsets ``dailylifeapis`` / ``huggingface`` /
``multimedia``, split ``test``. Fields used: ``id``, ``instruction``, ``type``
(single/chain/dag), ``sampled_nodes`` (tools with argument specs),
``tool_nodes`` (gold calls) and ``tool_links`` (gold edges, by tool name).
There are no executable tools, so execution uses deterministic mocks and the
metrics are structural (node / edge / argument F1).

Illustrative gold fields (made-up values; the shape is what the parsing below expects)::

    tool_nodes = [{"task": "Image-to-Text", "arguments": ["example.jpg"]},
                  {"task": "Translation", "arguments": ["<node-0>", "fr"]}]   # <node-0> = output of node 0
    # or, in dailylifeapis: "arguments": [{"name": "city", "value": "Paris"}, ...]
    tool_links = [{"source": "Image-to-Text", "target": "Translation"}]
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

# "<node-2>" as an argument value means "the output of gold node 2" (0-based).
_NODE_REF = re.compile(r"^<node-(\d+)>$")
_JSON_TYPES = {"string", "integer", "number", "boolean", "array", "object"}


def _loads(value: Any) -> Any:
    """Fields may be JSON strings or already-parsed lists; missing values become ``[]``."""
    return json.loads(value) if isinstance(value, str) else (value or [])


def _arg_items(arguments: Any) -> list[tuple[str, Any]]:
    """Arguments are either ``[{"name", "value"}]`` or positional ``["x.jpg", ...]``.

    Returns ``(name, value)`` pairs; positional args get the names ``arg0, arg1, ...``.
    """
    items = []
    for index, arg in enumerate(arguments or []):
        if isinstance(arg, dict):
            items.append((arg.get("name", f"arg{index}"), arg.get("value", arg.get("desc"))))
        else:
            items.append((f"arg{index}", arg))
    return items


def _spec_from_node(node: dict[str, Any]) -> ToolSpec:
    """Build a ToolSpec from a ``sampled_nodes`` entry (``task`` = tool name, ``arguments`` = param specs)."""
    props = {}
    for index, arg in enumerate(node.get("arguments") or []):
        if isinstance(arg, dict):
            props[arg.get("name", f"arg{index}")] = {"type": arg.get("type", "string"),
                                                     "description": arg.get("desc", "")}
        else:
            props[f"arg{index}"] = {"type": "string", "description": str(arg)}
    # TaskBench uses free-form types ("image", "text", ...). Fall back to string so the
    # schema stays valid JSON-schema, but keep the original type visible in the description.
    for prop in props.values():
        if prop["type"] not in _JSON_TYPES:
            prop["description"] = f"({prop['type']}) {prop['description']}".strip()
            prop["type"] = "string"
    return ToolSpec(name=node["task"], description=node.get("desc", node["task"]),
                    parameters={"type": "object", "properties": props, "required": list(props)})


def gold_plan(query: str, nodes: list[dict], links: list[dict]) -> Plan:
    """Turn gold ``tool_nodes`` + ``tool_links`` into a Plan (node i becomes step ``s{i+1}``).

    ``<node-N>`` argument values become data refs. Links are given by tool *name*, so a link
    without a matching data ref becomes an ordering-only dependency (``after``). Final steps
    are those nothing else depends on.
    """
    steps = []
    for index, node in enumerate(nodes):
        # HuggingFace/Multimedia list some nodes as a bare tool-name string (no args),
        # and a few dict nodes use "tool" instead of "task".
        if not isinstance(node, dict):
            node = {"task": node, "arguments": []}
        elif "task" not in node:
            node = {**node, "task": node.get("tool", "unknown")}
        args = {}
        for name, value in _arg_items(node.get("arguments")):
            match = _NODE_REF.match(value) if isinstance(value, str) else None
            args[name] = {"$ref": f"s{int(match.group(1)) + 1}"} if match else value
        steps.append(Step(id=f"s{index + 1}", tool=node["task"], args=args))

    step_id_of_tool = {step.tool: step.id for step in steps}  # if a tool repeats, its last step wins
    for link in links:
        # A handful of rows (5 of 4318 in dailylifeapis) use origin/destination/dest instead
        # of source/target for the same thing - tolerate both rather than crash on them.
        source_id = step_id_of_tool.get(link.get("source", link.get("origin")))
        target_id = step_id_of_tool.get(link.get("target", link.get("destination", link.get("dest"))))
        if source_id and target_id:
            target_step = next(step for step in steps if step.id == target_id)
            if source_id not in target_step.deps():
                target_step.after.append(source_id)

    depended_on = {dep for step in steps for dep in step.deps()}
    final = [step.id for step in steps if step.id not in depended_on]
    return Plan(query=query, steps=steps, final=final)


def _gold_structure(task: Task) -> tuple[list, list, list]:
    """Gold (nodes, edges, literal args) for F1 scoring. Ref args are excluded: edges already cover them."""
    def _norm(n):
        if not isinstance(n, dict):
            return {"task": n, "arguments": []}
        return n if "task" in n else {**n, "task": n.get("tool", "unknown")}
    raw = [_norm(n) for n in task.gold["nodes"]]
    nodes = [node["task"] for node in raw]
    edges = [(link.get("source", link.get("origin")),
              link.get("target", link.get("destination", link.get("dest")))) for link in task.gold["links"]]
    args = []
    for node in raw:
        for name, value in _arg_items(node.get("arguments")):
            if not (isinstance(value, str) and _NODE_REF.match(value)):
                args.append((node["task"], name, str(value)))
    return nodes, edges, args


def _predicted_structure(result: AgentResult) -> tuple[list, list, list]:
    """Predicted (nodes, edges, literal args), from the plan if there is one, else from the raw calls."""
    plan = result.plan or result.initial_plan
    if plan is None:  # ReAct baseline: no explicit structure, so no edges
        nodes = [call.tool for call in result.calls]
        args = [(call.tool, name, str(value)) for call in result.calls for name, value in call.args.items()]
        return nodes, [], args

    tool_of_step = {step.id: step.tool for step in plan.steps}
    nodes = [step.tool for step in plan.steps]
    edges = []
    for step in plan.steps:
        for dep in step.deps():
            if dep in tool_of_step:
                edges.append((tool_of_step[dep], step.tool))
    args = []
    for step in plan.steps:
        for name, value in step.args.items():
            if not is_ref(value):
                args.append((step.tool, name, str(value)))
    return nodes, edges, args


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

        dataset = load_dataset(self.hf_path, self.subset, split=self.split)

        # Each row only lists a sample of tools; every task gets the full catalogue
        # (union over all rows) so the planner has to pick the right tools itself.
        catalog: dict[str, ToolSpec] = {}
        for row in dataset:
            for node in _loads(row["sampled_nodes"]):
                catalog.setdefault(node["task"], _spec_from_node(node))
        self.catalog = list(catalog.values())

        tasks = []
        for row in dataset:
            if self.types and row["type"] not in self.types:
                continue
            nodes, links = _loads(row["tool_nodes"]), _loads(row["tool_links"])
            tasks.append(Task(
                id=str(row["id"]), query=row["instruction"], tools=self.catalog,
                gold={"nodes": nodes, "links": links},
                gold_plan=gold_plan(row["instruction"], nodes, links),
                meta={"type": row["type"], "n_tools": row["n_tools"]},
            ))
        # Shuffle with a fixed seed so ``limit`` takes the same random subset every run.
        if self.sample_seed is not None:
            random.Random(self.sample_seed).shuffle(tasks)
        return tasks[: self.limit] if self.limit else tasks

    def registry(self, task: Task) -> ToolRegistry:
        return ToolRegistry(task.tools, {tool.name: make_mock(tool) for tool in task.tools})

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        """Structural F1s; success = exactly the right tools and the right edges."""
        gold_nodes, gold_edges, gold_args = _gold_structure(task)
        pred_nodes, pred_edges, pred_args = _predicted_structure(result)
        node = f1(pred_nodes, gold_nodes)
        edge = f1(pred_edges, gold_edges)
        arg = f1(pred_args, gold_args)
        return {"node_f1": node["f1"], "edge_f1": edge["f1"], "arg_f1": arg["f1"],
                "success": float(node["f1"] == 1.0 and edge["f1"] == 1.0)}
