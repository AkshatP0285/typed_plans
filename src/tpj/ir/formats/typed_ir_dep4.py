"""typed_ir variant v4: dep3 + fixes from dep3's failure analysis (27B, DailyLife).

dep3 remaining fails (65/150): rule over-firing (extra search/attend steps ~6) and
listed-but-independent actions left unchained (~8; gold chains every listed action in
mention order). v4 makes search/attend steps conditional on explicit wording and
defaults to mention-order chaining (a dataset convention - reported separately).
"""
from __future__ import annotations

from tpj.ir.formats.typed_ir import TypedIRFormat

INSTRUCTIONS_DEP4 = """\
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

[1. ONE STEP PER ACTION] Create one step for every action the request asks for. Never merge two
requested actions into one step and never drop one. Pick the tool whose name matches the action
most literally. Do not repeat the identical call twice. Only add a step the request does not
state outright in these two cases:
  - the request explicitly asks to FIND / LOOK UP / SEARCH FOR something ("find the nearest X",
    "look up X") -> a search step, then the step that uses its result;
  - the request explicitly says the user will JOIN / ATTEND a meeting, call or event -> an attend
    step in addition to the step that schedules or books it.
  Otherwise do NOT invent extra search, lookup or attend steps.

[2. ORDER] Write the steps in the order the actions are mentioned in the request.

[3. DEPENDENCIES] "after" encodes the dependency graph:
  - DATA: the step uses an earlier step's OUTPUT -> pass {"$ref":"sK"} in the argument AND list sK.
  - ORDER: by default, treat the actions as a sequence in the order they are mentioned: each step
    lists the step mentioned IMMEDIATELY before it in "after", even when no data is passed and the
    actions seem unrelated. Only start a new branch (several steps sharing one predecessor) when the
    request clearly says two actions follow from the same earlier action or happen in parallel.
  List only the IMMEDIATE predecessor(s): in a chain s1 -> s2 -> s3, s3's "after" is ["s2"], NOT
  ["s1","s2"]. The first step has an empty "after".

Example (hypothetical tools) - "Order me a pizza, renew my library book, and text my sister":
{"steps":[
 {"id":"s1","tool":"order_food","args":{"item":"pizza"},"returns":"Order","safety":"danger","after":[]},
 {"id":"s2","tool":"renew_book","args":{"book":"library book"},"returns":"Renewal","safety":"danger","after":["s1"]},
 {"id":"s3","tool":"send_text","args":{"to":"sister"},"returns":"Message","safety":"danger","after":["s2"]}
],"final":["s3"]}
The three actions are unrelated, yet each step follows the one mentioned before it.

"final": ids of the steps whose outputs answer the request.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRDep4Format(TypedIRFormat):
    name = "typed_ir_dep4"

    def instructions(self) -> str:
        return INSTRUCTIONS_DEP4
