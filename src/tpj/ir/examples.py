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
