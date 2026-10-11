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

# A whole-string placeholder: "{{s1}}" or "{{s1.field.sub}}". Group 1 = step id, group 2 = optional path.
_PLACEHOLDER = re.compile(r"^\{\{\s*(s\d+)(?:\.([\w.\-]+))?\s*\}\}$")

INSTRUCTIONS = """\
Write the plan as a JSON graph: {"nodes": [...], "edges": [...], "final": [...]}.
- node: {"id": "s1", "tool": "<tool name>", "args": {...}}. To use the output of another
  node write the string "{{s1}}" or "{{s1.field}}" as the argument value.
- edge: {"source": "s1", "target": "s2"} for every dependency.
- "final": ids of the nodes that answer the request."""


# Pydantic models for the graph JSON. Their names appear in json_schema() (sent for constrained
# decoding), so renaming them changes the schema.
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
    """Replace IR references ``{"$ref": "s1", "path": "x"}`` with placeholder strings ``"{{s1.x}}"`` (recursively)."""
    if is_ref(value):
        path = value.get("path")
        return "{{" + value["$ref"] + (f".{path}" if path else "") + "}}"
    if isinstance(value, dict):
        return {key: _to_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_placeholders(item) for item in value]
    return value


def _from_placeholders(value: Any) -> Any:
    """Inverse of :func:`_to_placeholders`: turn ``"{{s1.x}}"`` strings back into IR references."""
    if isinstance(value, str):
        match = _PLACEHOLDER.match(value)
        if not match:
            return value  # an ordinary string argument
        step_id, path = match.group(1), match.group(2)
        return {"$ref": step_id, "path": path} if path else {"$ref": step_id}
    if isinstance(value, dict):
        return {key: _from_placeholders(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_from_placeholders(item) for item in value]
    return value


class GraphFormat(PlanFormat):
    name = "graph"

    def instructions(self) -> str:
        return INSTRUCTIONS

    def render(self, plan: Plan) -> str:
        nodes = [{"id": step.id, "tool": step.tool, "args": _to_placeholders(step.args)} for step in plan.steps]
        # One edge per dependency (data refs, `after` and `when` refs alike); the graph has no other way to say it.
        edges = []
        for step in plan.steps:
            for dep_id in sorted(step.deps()):
                edges.append({"source": dep_id, "target": step.id})
        return json.dumps({"nodes": nodes, "edges": edges, "final": plan.final}, indent=1)

    def json_schema(self) -> dict[str, Any]:
        return _Graph.model_json_schema()

    async def parse(self, text: str, query: str) -> Plan:
        try:
            graph = _Graph.model_validate(parse_json(text))
        except (ValueError, ValidationError) as exc:  # ValueError: not JSON at all
            raise PlanParseError(str(exc)) from exc
        steps = []
        for node in graph.nodes:
            step = Step(id=node.id, tool=node.tool, args=_from_placeholders(node.args))
            # Edges already implied by a placeholder are data dependencies; any remaining incoming
            # edge only constrains order, so it goes into `after`.
            incoming = {edge.source for edge in graph.edges if edge.target == node.id}
            step.after = sorted(incoming - step.deps())
            steps.append(step)
        return Plan(query=query, steps=steps, final=graph.final)
