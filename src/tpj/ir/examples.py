"""Canonical example IR (Figure 2 of the proposal), used in prompts and tests."""
from __future__ import annotations

from tpj.ir.schema import Plan

GOA_QUERY = (
    "Plan a Goa trip for 4 people: compare flight and train and book the cheaper one. "
    "Also find a beach hotel with free cancellation, and check the weather. If it is likely "
    "to rain, book an indoor activity, otherwise book a scooter. Then email everyone the itinerary."
)


def _r(step: str, path: str | None = None) -> dict:
    return {"$ref": step, "path": path} if path else {"$ref": step}


GOA_PLAN = Plan.model_validate(
    {
        "query": GOA_QUERY,
        "steps": [
            {"id": "s1", "tool": "search_flights", "args": {"origin": "BLR", "dest": "GOI"},
             "returns": "list[Flight]", "safety": "safe"},
            {"id": "s2", "tool": "search_trains", "args": {"origin": "BLR", "dest": "GOI"},
             "returns": "list[Train]", "safety": "safe"},
            {"id": "s3", "tool": "search_hotels", "args": {"near": "beach", "cancellation": "free"},
             "returns": "list[Hotel]", "safety": "safe"},
            {"id": "s4", "tool": "get_weather", "args": {"city": "Goa", "when": "weekend"},
             "returns": "Forecast", "safety": "safe"},
            {"id": "s5", "tool": "pick_cheapest", "args": {"options": [_r("s1"), _r("s2")]},
             "returns": "Option", "safety": "safe"},
            {"id": "s6", "tool": "greater_than", "args": {"value": _r("s4", "rain_chance"), "threshold": 0.5},
             "returns": "bool", "safety": "safe"},
            {"id": "s7", "tool": "book_transport", "args": {"option_id": _r("s5", "id"), "people": 4},
             "returns": "Booking", "safety": "danger"},
            {"id": "s8", "tool": "book_hotel", "args": {"hotel_id": _r("s3", "first.id"), "people": 4},
             "returns": "Booking", "safety": "danger"},
            {"id": "s9", "tool": "book_indoor_activity", "args": {"people": 4},
             "returns": "Booking", "safety": "danger", "when": {"ref": "s6", "equals": True}},
            {"id": "s10", "tool": "book_scooter", "args": {"people": 4},
             "returns": "Booking", "safety": "danger", "when": {"ref": "s6", "equals": False}},
            {"id": "s11", "tool": "send_email",
             "args": {"bookings": [_r("s7"), _r("s8"), _r("s9"), _r("s10")]},
             "returns": "Sent", "safety": "danger"},
        ],
        "final": ["s11"],
    }
)

# Expected waves for GOA_PLAN (Figure 3).
GOA_WAVES = [["s1", "s2", "s3", "s4"], ["s5", "s6"], ["s7", "s8", "s9", "s10"], ["s11"]]

# Separate from GOA_PLAN on purpose: tests and the toy benchmark pin exact structure
# (waves, dirty regions) against GOA_PLAN, so it stays untouched. This is the one
# actually shown to the planner as its few-shot example (see prompted.py).
#
# `after` is filled in for every step with a dependency, even ones that also carry a
# $ref - measured on the real benchmark that models under-declare deps by ~40% when
# "after" is demonstrated as a rare, ordering-only field (avg 0.42 deps/step vs graph
# format's 0.70 on identical tasks), because it reads as optional busywork buried
# alongside returns/safety/when. Demonstrating it as the complete, redundant
# dependency list (see schema.py's field description) measurably fixes that.
def _with_full_after(plan: Plan) -> Plan:
    steps = []
    for step in plan.steps:
        d = step.model_dump()
        d["after"] = sorted(step.deps())
        steps.append(d)
    return Plan.model_validate({"query": plan.query, "steps": steps, "final": plan.final})


FEWSHOT_PLAN = _with_full_after(
    Plan.model_validate(
        {
            "query": GOA_QUERY,
            "steps": [s.model_dump() for s in GOA_PLAN.steps] + [
                {"id": "s12", "tool": "block_calendar", "args": {"dates": "trip dates"},
                 "returns": "Sent", "safety": "safe", "after": ["s11"]},
            ],
            "final": ["s12"],
        }
    )
)

# Second and third few-shot examples - short, different domains, each emphasising one
# pattern GOA_PLAN/FEWSHOT_PLAN under-demonstrates on its own: conditional branching
# via "when" (example 2), and an ordering-only "after" chain paired with a $ref in a
# domain that isn't travel booking, to check the pattern generalises (example 3).
BALANCE_QUERY = "Check my account balance. If it's below 100, send me a low-balance alert; otherwise send a confirmation that all is fine."
BALANCE_PLAN = _with_full_after(Plan.model_validate(
    {
        "query": BALANCE_QUERY,
        "steps": [
            {"id": "s1", "tool": "get_balance", "args": {}, "returns": "float", "safety": "safe"},
            {"id": "s2", "tool": "less_than", "args": {"value": _r("s1"), "threshold": 100},
             "returns": "bool", "safety": "safe"},
            {"id": "s3", "tool": "send_alert", "args": {"reason": "low balance"},
             "returns": "Sent", "safety": "safe", "when": {"ref": "s2", "equals": True}},
            {"id": "s4", "tool": "send_confirmation", "args": {"message": "all is fine"},
             "returns": "Sent", "safety": "safe", "when": {"ref": "s2", "equals": False}},
        ],
        "final": ["s3", "s4"],
    }
))

DOWNLOAD_QUERY = "Download the quarterly report file, then verify its checksum, then archive it once verified."
DOWNLOAD_PLAN = _with_full_after(Plan.model_validate(
    {
        "query": DOWNLOAD_QUERY,
        "steps": [
            {"id": "s1", "tool": "download_file", "args": {"name": "quarterly_report"},
             "returns": "FileHandle", "safety": "safe"},
            # ordering-only: verify_checksum needs the file to already be downloaded, but
            # does not use s1's return value (it re-reads the file by name) - still listed
            # in "after" per the required/redundant dependency-list convention above.
            {"id": "s2", "tool": "verify_checksum", "args": {"name": "quarterly_report"},
             "returns": "bool", "safety": "safe", "after": ["s1"]},
            {"id": "s3", "tool": "archive_file", "args": {"file": _r("s1")},
             "returns": "Sent", "safety": "safe", "when": {"ref": "s2", "equals": True}},
        ],
        "final": ["s3"],
    }
))

# (query, plan) pairs in the order they're added to the prompt - prompted.py's
# `n_fewshot` picks a prefix of this list, so 1/2/3 examples is one config change.
FEWSHOT_EXAMPLES = [(GOA_QUERY, FEWSHOT_PLAN), (BALANCE_QUERY, BALANCE_PLAN), (DOWNLOAD_QUERY, DOWNLOAD_PLAN)]
