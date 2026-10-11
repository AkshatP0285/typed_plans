"""Hand-made tasks over executable trip-planning tools (Figure 2 of the proposal).

Runs without any download; used for smoke tests and debugging the loop.
"""
from __future__ import annotations

from typing import Any

from tpj.benchmarks.base import AgentResult, Benchmark, Task
from tpj.benchmarks.scoring import values_equal
from tpj.ir.examples import GOA_PLAN, GOA_QUERY
from tpj.ir.schema import Plan
from tpj.tools.registry import ToolRegistry
from tpj.tools.spec import ToolSpec

_FLIGHTS = [{"id": "F1", "mode": "flight", "price": 5200}, {"id": "F2", "mode": "flight", "price": 4100}]
_TRAINS = [{"id": "T1", "mode": "train", "price": 1900}, {"id": "T2", "mode": "train", "price": 2300}]
_HOTELS = [{"id": "H1", "name": "Sea Breeze", "near": "beach", "cancellation": "free", "price": 3500},
           {"id": "H2", "name": "Palm Stay", "near": "beach", "cancellation": "free", "price": 4200}]


# ---- Tool implementations (deterministic, no network) ----

def search_flights(origin: str, dest: str) -> list[dict]:
    return [dict(flight, origin=origin, dest=dest) for flight in _FLIGHTS]


def search_trains(origin: str, dest: str) -> list[dict]:
    return [dict(train, origin=origin, dest=dest) for train in _TRAINS]


def search_hotels(near: str, cancellation: str) -> list[dict]:
    return [hotel for hotel in _HOTELS if hotel["near"] == near and hotel["cancellation"] == cancellation]


def get_weather(city: str, when: str) -> dict:
    return {"city": city, "when": when, "rain_chance": 0.7, "temp_c": 27}


def pick_cheapest(options: list) -> dict:
    """``options`` mixes whole result lists and single options, e.g. ``[[F1, F2], [T1, T2]]`` or ``[F1, T1]``."""
    flat = []
    for group in options:
        if isinstance(group, list):
            flat.extend(group)
        else:
            flat.append(group)
    return min(flat, key=lambda option: option["price"])


def greater_than(value: float, threshold: float) -> bool:
    return float(value) > float(threshold)


def book_transport(option_id: str, people: int) -> dict:
    return {"booking_id": f"BK-{option_id}", "kind": "transport", "people": people}


def book_hotel(hotel_id: str, people: int) -> dict:
    return {"booking_id": f"BK-{hotel_id}", "kind": "hotel", "people": people}


def book_indoor_activity(people: int) -> dict:
    return {"booking_id": "BK-INDOOR", "kind": "activity", "people": people}


def book_scooter(people: int) -> dict:
    return {"booking_id": "BK-SCOOTER", "kind": "scooter", "people": people}


def send_email(bookings: list) -> dict:
    ids = [booking["booking_id"] for booking in bookings if booking]  # skip empty slots (e.g. a skipped branch)
    return {"sent": True, "booking_ids": ids}


# Tool name -> Python function, used by the ToolRegistry to actually execute calls.
IMPLS = {fn.__name__: fn for fn in [
    search_flights, search_trains, search_hotels, get_weather, pick_cheapest, greater_than,
    book_transport, book_hotel, book_indoor_activity, book_scooter, send_email,
]}


# ---- Tool specs shown to the planner (JSON-schema parameters) ----

def _obj(**props: Any) -> dict:
    """JSON-schema object with all given properties required: ``_obj(city=_S)``."""
    return {"type": "object", "properties": props, "required": list(props)}


# Shorthands for the JSON-schema types used below.
_S = {"type": "string"}
_I = {"type": "integer"}
_N = {"type": "number"}

SPECS = [
    ToolSpec(name="search_flights", description="Search flights between airport codes.",
             parameters=_obj(origin=_S, dest=_S), returns="list[Flight]"),
    ToolSpec(name="search_trains", description="Search trains between station codes.",
             parameters=_obj(origin=_S, dest=_S), returns="list[Train]"),
    ToolSpec(name="search_hotels", description="Search hotels by location and cancellation policy.",
             parameters=_obj(near=_S, cancellation=_S), returns="list[Hotel]"),
    ToolSpec(name="get_weather", description="Weather forecast with rain_chance in [0,1].",
             parameters=_obj(city=_S, when=_S), returns="Forecast"),
    ToolSpec(name="pick_cheapest", description="Pick the cheapest option (by price) from lists of options.",
             parameters=_obj(options={"type": "array"}), returns="Option"),
    ToolSpec(name="greater_than", description="Return value > threshold.",
             parameters=_obj(value=_N, threshold=_N), returns="bool"),
    ToolSpec(name="book_transport", description="Book a flight or train option by id.",
             parameters=_obj(option_id=_S, people=_I), returns="Booking", safety="danger"),
    ToolSpec(name="book_hotel", description="Book a hotel by id.",
             parameters=_obj(hotel_id=_S, people=_I), returns="Booking", safety="danger"),
    ToolSpec(name="book_indoor_activity", description="Book an indoor activity.",
             parameters=_obj(people=_I), returns="Booking", safety="danger"),
    ToolSpec(name="book_scooter", description="Book scooters.",
             parameters=_obj(people=_I), returns="Booking", safety="danger"),
    ToolSpec(name="send_email", description="Email the itinerary with the given bookings.",
             parameters=_obj(bookings={"type": "array"}), returns="Sent", safety="danger"),
]

# Gold plan for the second toy task: compare flights and trains, book the cheapest (train T1).
_CHEAPEST_PLAN = Plan.model_validate({
    "query": "Find the cheapest way to get from BLR to GOI (flight or train) and book it for 2 people.",
    "steps": [
        {"id": "s1", "tool": "search_flights", "args": {"origin": "BLR", "dest": "GOI"}, "returns": "list[Flight]"},
        {"id": "s2", "tool": "search_trains", "args": {"origin": "BLR", "dest": "GOI"}, "returns": "list[Train]"},
        {"id": "s3", "tool": "pick_cheapest", "args": {"options": [{"$ref": "s1"}, {"$ref": "s2"}]},
         "returns": "Option"},
        {"id": "s4", "tool": "book_transport", "args": {"option_id": {"$ref": "s3", "path": "id"}, "people": 2},
         "returns": "Booking", "safety": "danger"},
    ],
    "final": ["s4"],
})


class ToyBenchmark(Benchmark):
    name = "toy"

    def load(self) -> list[Task]:
        # Gold answers are the output of each plan's final step(s).
        return [
            Task("goa", GOA_QUERY, SPECS,
                 gold={"answer": {"sent": True, "booking_ids": ["BK-T1", "BK-H1", "BK-INDOOR"]}}, gold_plan=GOA_PLAN),
            Task("cheapest", _CHEAPEST_PLAN.query, SPECS,
                 gold={"answer": {"booking_id": "BK-T1", "kind": "transport", "people": 2}}, gold_plan=_CHEAPEST_PLAN),
        ]

    def registry(self, task: Task) -> ToolRegistry:
        return ToolRegistry(SPECS, IMPLS)

    def score(self, task: Task, result: AgentResult) -> dict[str, float]:
        """Success = right final answer AND no side-effecting tool the gold plan did not use."""
        answer_ok = values_equal(result.final_value(), task.gold["answer"])

        # Dangerous (side-effecting) tools the gold plan is allowed to call.
        gold_danger = set()
        if task.gold_plan:
            gold_danger = {step.tool for step in task.gold_plan.steps if step.safety == "danger"}
        # Side-effecting tools the agent actually called successfully.
        done_danger = {call.tool for call in result.successful_calls() if call.tool.startswith(("book_", "send_"))}
        unwanted = done_danger - gold_danger

        return {"success": float(answer_ok and not unwanted), "answer_match": float(answer_ok),
                "unwanted_side_effects": float(len(unwanted))}
