"""typed_ir variant v3: dep2 + fixes from dep2's failure analysis (27B, DailyLife).

dep2 failures (69/150): missing_tool 44, extra_tool 16, link errors (nodes ok) 23.
Dropped tools were implied sub-actions ("nearest X" -> search first; "call for a meeting"
-> call AND attend meeting) that "use the minimum number of steps" told the planner to merge.
v3: decompose completely, keep request order, immediate-predecessor edges only.
"""
from __future__ import annotations

from tpj.ir.formats.typed_ir import TypedIRFormat

INSTRUCTIONS_DEP3 = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
Each step has:
- "id": "s1", "s2", ... (unique)
- "tool": exact tool name from the tool list
- "args": argument values. To use the output of an earlier step write {"$ref": "sK"} or
  {"$ref": "sK", "path": "field.sub"}. Never copy values you have not seen; reference them.
- "returns": the type the step produces
- "safety": "danger" if the call changes the world (booking, payment, sending, deleting), else "safe"
- "when": optional {"ref": "sK", "equals": true|false}
- "after": ids of the step(s) that must run IMMEDIATELY before this one.

[1. DECOMPOSE COMPLETELY] Create one step for EVERY action or activity the request mentions or
clearly implies. Never merge two actions into one step and never skip one because it seems minor
or redundant. In particular:
  - "the nearest / best / cheapest X", "find X", "look up X" -> first a search/lookup step to find X,
    then the step that uses it.
  - a meeting, call, appointment, course or event that the user will join/attend/take part in ->
    its own step with the matching join/attend tool, in addition to any step that schedules,
    books or calls for it.
  - pick the tool whose name matches the action most literally; do not substitute a loosely
    related tool.
  Do not repeat the identical call twice.

[2. ORDER] Write the steps in the order the actions appear in the request (or the order they
logically must happen).

[3. DEPENDENCIES] "after" encodes the dependency graph. Declare BOTH kinds:
  - DATA: the step uses an earlier step's OUTPUT -> pass {"$ref":"sK"} in the argument AND list sK.
  - ORDER: the request implies one action happens before another even when no data is passed ->
    list the earlier step.
  List only the IMMEDIATE predecessor(s), not every earlier step: in a chain s1 -> s2 -> s3,
  s3's "after" is ["s2"], NOT ["s1","s2"]. Use a tree (two steps sharing one predecessor) only when
  two actions genuinely branch from the same earlier action. A plan where every step has an empty
  "after" is almost always WRONG; only genuinely independent actions have no "after".

Example (hypothetical tools) - "Register me for the pottery workshop, let me join it, then take me
to the closest art supply shop":
{"steps":[
 {"id":"s1","tool":"register_workshop","args":{"name":"pottery workshop"},"returns":"Registration","safety":"danger","after":[]},
 {"id":"s2","tool":"join_workshop","args":{"registration":{"$ref":"s1"}},"returns":"Session","safety":"safe","after":["s1"]},
 {"id":"s3","tool":"find_place","args":{"query":"closest art supply shop"},"returns":"Location","safety":"safe","after":["s2"]},
 {"id":"s4","tool":"navigate_to","args":{"destination":{"$ref":"s3"}},"returns":"Trip","safety":"danger","after":["s3"]}
],"final":["s4"]}
Note: "closest shop" produced a find step; joining got its own step; each step lists only its
immediate predecessor.

"final": ids of the steps whose outputs answer the request.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRDep3Format(TypedIRFormat):
    name = "typed_ir_dep3"

    def instructions(self) -> str:
        return INSTRUCTIONS_DEP3
