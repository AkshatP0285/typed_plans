"""Typed IR written as text lines (proposal Fig. 2): ``s3 = book(id=s1.first.id) if s2 -> Booking [DANGER]``."""
from __future__ import annotations

import json

from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan

REF_NOTE = 'each plan line is "sN = tool(args) -> return type [safe|DANGER]"; an argument sK or sK.field means an output of step sK; "if sK" runs the step only when sK is true; "after sK" means it waits for sK'


def _val(v):
    if is_ref(v):
        return v["$ref"] + (f".{v['path']}" if v.get("path") else "")
    if isinstance(v, list):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{json.dumps(k)}: {_val(x)}" for k, x in v.items()) + "}"
    return json.dumps(v, ensure_ascii=False)


def ir_text(plan: Plan) -> str:
    if not plan.steps:
        return "No tool needed."
    lines = []
    for s in plan.steps:
        line = f"{s.id} = {s.tool}(" + ", ".join(f"{k}={_val(v)}" for k, v in s.args.items()) + ")"
        if s.when is not None:
            ref = s.when.ref + (f".{s.when.path}" if s.when.path else "")
            line += f" if {ref}" if s.when.equals is True else f" if not {ref}" if s.when.equals is False else f" if {ref} == {_val(s.when.equals)}"
        if s.after:
            line += " after " + ", ".join(s.after)
        lines.append(line + f" -> {s.returns} [{'DANGER' if s.safety == 'danger' else 'safe'}]")
    lines.append("answer: " + ", ".join(plan.final))
    return "\n".join(lines)
