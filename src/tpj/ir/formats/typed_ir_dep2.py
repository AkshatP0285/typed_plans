"""typed_ir variant v2: forces BOTH data-flow and temporal-ordering dependencies.

Diagnosis: TaskBench DailyLife deps are temporal ("do X before Y") with literal args
(no data passed). v1 only pushed data deps ($ref), so planners under-declared ordering
edges. v2 adds explicit temporal-ordering guidance + a no-data chain example.
"""
from __future__ import annotations

from tpj.ir.formats.typed_ir import TypedIRFormat

INSTRUCTIONS_DEP2 = """\
Write the plan as JSON: {"steps": [...], "final": [...]}.
Use the minimum number of steps that fulfils the request - never repeat the same tool call,
or a near-duplicate of one, more than once with the same intent.
Each step has:
- "id": "s1", "s2", ... (unique)
- "tool": exact tool name from the tool list
- "args": argument values. To use the output of an earlier step write {"$ref": "sK"} or
  {"$ref": "sK", "path": "field.sub"}. Never copy values you have not seen; reference them.
- "returns": the type the step produces
- "safety": "danger" if the call changes the world (booking, payment, sending, deleting), else "safe"
- "when": optional {"ref": "sK", "equals": true|false}
- "after": list of ids of EVERY step that must run before this one.

[MOST IMPORTANT] "after" is the plan's dependency graph and is the part planners most often get
WRONG by leaving edges out. Declare BOTH kinds of dependency in "after":
  1) DATA dependency - this step uses an earlier step's OUTPUT. Pass it as {"$ref":"sK"} in the
     argument AND list "sK" in "after".
  2) ORDER dependency - the request implies one action should happen BEFORE another, even when no
     data is passed (e.g. "book a car, then order a taxi, then enroll in a course"). List the
     earlier action's id in "after" even though its output is not used.
Read the request as a sequence of actions and chain them: if action B is described after action A
(or logically must follow it), add A's id to B's "after". Most multi-action requests form a chain
or tree; a set of steps that all have empty "after" is almost always WRONG. Only genuinely
independent actions have no "after".

Example A (data flow) - "Find flights to Paris then book the cheapest":
 s1 search_flights(dest=Paris) after:[]; s2 cheapest(flights=$ref s1) after:[s1]; s3 book_flight(flight=$ref s2) after:[s2]
Example B (temporal order, NO data passed) - "Book a car, then order a taxi, then enroll in a course":
{"steps":[
 {"id":"s1","tool":"book_car","args":{"date":"2023-08-01","location":"NYC"},"returns":"Booking","safety":"danger","after":[]},
 {"id":"s2","tool":"order_taxi","args":{"location":"Columbia University"},"returns":"Ride","safety":"danger","after":["s1"]},
 {"id":"s3","tool":"enroll_in_course","args":{"course":"Data Science"},"returns":"Enrollment","safety":"danger","after":["s2"]}
],"final":["s3"]}
Here args are all literal (no $ref) yet each step lists the previous in "after" - the ordering chain.

"final": ids of the steps whose outputs answer the request.
If no tool is needed, return {"steps": [], "final": []}."""


class TypedIRDep2Format(TypedIRFormat):
    name = "typed_ir_dep2"

    def instructions(self) -> str:
        return INSTRUCTIONS_DEP2
