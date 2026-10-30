"""typed_ir variant with a dependency-forcing prompt (prompt-tuning experiment).

Same schema/parse as typed_ir; only the INSTRUCTIONS are strengthened to make the
planner declare dependency edges (the under-declared-deps weakness that tanks Link-F1).
"""
from __future__ import annotations

from tpj.ir.formats.typed_ir import TypedIRFormat

INSTRUCTIONS_DEP = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
Use the minimum number of steps that fulfils the request - never repeat the same tool call,
or a near-duplicate of one, more than once with the same intent.
Each step has:
- "id": "s1", "s2", ... (unique)
- "tool": exact tool name from the tool list
- "args": argument values. To use the output of an earlier step write {"$ref": "sK"} or
  {"$ref": "sK", "path": "field.sub"} ("first"/"last"/integers index lists). Never copy
  values you have not seen; reference them.
- "returns": the type the step produces, e.g. "list[Flight]", "Booking", "bool", "float"
- "safety": "danger" if the call changes the world (booking, payment, sending, deleting,
  writing), otherwise "safe"
- "when": optional {"ref": "sK", "equals": true|false} to run the step only under a condition
- "after": list of ids of EVERY step this one depends on.

[MOST IMPORTANT] Dependencies are the part planners most often get WRONG by leaving them out.
Before writing each step, ask: "does this step consume the OUTPUT of an earlier step?" If yes
you MUST do BOTH: (a) pass that output as {"$ref":"sK"} in the argument that uses it, AND
(b) list "sK" in this step's "after". Trace the data flow of the whole request end to end.
Most multi-step requests form a CHAIN or tree of dependencies; a set of steps that all have
empty "after" is almost always WRONG - only truly independent steps have no "after".

Example - "Find flights to Paris then book the cheapest one":
{"steps":[
 {"id":"s1","tool":"search_flights","args":{"dest":"Paris"},"returns":"list[Flight]","safety":"safe","after":[]},
 {"id":"s2","tool":"cheapest","args":{"flights":{"$ref":"s1"}},"returns":"Flight","safety":"safe","after":["s1"]},
 {"id":"s3","tool":"book_flight","args":{"flight":{"$ref":"s2"}},"returns":"Booking","safety":"danger","after":["s2"]}
],"final":["s3"]}
Here s2 depends on s1 and s3 on s2 - a full chain, each dependency in BOTH $ref and after.

"final": ids of the steps whose outputs answer the request.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRDepFormat(TypedIRFormat):
    name = "typed_ir_dep"

    def instructions(self) -> str:
        return INSTRUCTIONS_DEP
