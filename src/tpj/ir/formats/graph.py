"""Untyped graph format (TaskBench / GNNVerifier style): nodes + edges.

Compared with the typed IR it has no return types, no safety tags, no
conditions, and data flow is written as string placeholders ``"{{s1.field}}"``.
"""
from __future__ import annotations

import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from tpj.ir.formats.base import PlanFormat, PlanParseError
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan, Step
from tpj.llm.client import parse_json

_PLACEHOLDER = re.compile(r"^\{\{\s*(s\d+)(?:\.([\w.\-]+))?\s*\}\}$")

INSTRUCTIONS = """\
Write the plan as a JSON graph: {"nodes": [...], "edges": [...], "final": [...]}.
- node: {"id": "s1", "tool": "<tool name>", "args": {...}}. To use the output of another
  node write the string "{{s1}}" or "{{s1.field}}" as the argument value.
- edge: {"source": "s1", "target": "s2"} for every dependency.
- "final": ids of the nodes that answer the request."""


class _Node(BaseModel):
    id: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)


class _Edge(BaseModel):
    source: str
    target: str


class _Graph(BaseModel):
    nodes: list[_Node]
    edges: list[_Edge] = Field(default_factory=list)
    final: list[str] = Field(default_factory=list)


def _to_placeholders(value: Any) -> Any:
    if is_ref(value):
        path = value.get("path")
        return "{{" + value["$ref"] + (f".{path}" if path else "") + "}}"
    if isinstance(value, dict):
        return {k: _to_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_placeholders(v) for v in value]
    return value


def _from_placeholders(value: Any) -> Any:
    if isinstance(value, str):
        m = _PLACEHOLDER.match(value)
        if m:
            return {"$ref": m.group(1), "path": m.group(2)} if m.group(2) else {"$ref": m.group(1)}
        return value
    if isinstance(value, dict):
        return {k: _from_placeholders(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_from_placeholders(v) for v in value]
    return value


class GraphFormat(PlanFormat):
    name = "graph"

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        nodes = [{"id": s.id, "tool": s.tool, "args": _to_placeholders(s.args)} for s in plan.steps]
        edges = [{"source": d, "target": s.id} for s in plan.steps for d in sorted(s.deps())]
        return json.dumps({"nodes": nodes, "edges": edges, "final": plan.final}, indent=1)

    def json_schema(self) -> dict[str, Any]:
        return _Graph.model_json_schema()

    async def parse(self, text: str, query: str) -> Plan:
        try:
            g = _Graph.model_validate(parse_json(text))
        except (ValueError, ValidationError) as e:
            raise PlanParseError(str(e)) from e
        steps = []
        for n in g.nodes:
            step = Step(id=n.id, tool=n.tool, args=_from_placeholders(n.args))
            data_deps = step.deps()
            step.after = sorted({e.source for e in g.edges if e.target == n.id} - data_deps)
            steps.append(step)
        return Plan(query=query, steps=steps, final=g.final)
