import asyncio

from tpj.ir.examples import GOA_PLAN
from tpj.ir.formats import GraphFormat, TypedIRFormat
from tpj.repair.patch import JudgeDecision, apply_patch


def test_typed_ir_round_trip():
    fmt = TypedIRFormat()
    plan = asyncio.run(fmt.parse(fmt.render(GOA_PLAN), GOA_PLAN.query))
    assert plan.steps == GOA_PLAN.steps


def test_graph_round_trip_keeps_dataflow():
    fmt = GraphFormat()
    plan = asyncio.run(fmt.parse(fmt.render(GOA_PLAN), GOA_PLAN.query))
    by_id = {s.id: s for s in plan.steps}
    assert by_id["s8"].args["hotel_id"] == {"$ref": "s3", "path": "first.id"}
    # conditions are not expressible in the graph format; they survive only as ordering edges
    assert by_id["s9"].when is None and "s6" in by_id["s9"].after


def test_patch_ops():
    decision = JudgeDecision.model_validate({
        "verdict": "repair",
        "ops": [
            {"op": "edit", "id": "s4", "args": {"city": "Panaji", "when": "weekend"}},
            {"op": "remove", "id": "s10"},
            {"op": "add", "step": {"id": "s12", "tool": "get_weather", "args": {"city": "Goa", "when": "today"}}},
        ],
    })
    new = apply_patch(GOA_PLAN, decision)
    ids = new.ids()
    assert "s10" not in ids and "s12" in ids
    assert new.step("s4").args["city"] == "Panaji"
    assert GOA_PLAN.step("s4").args["city"] == "Goa"  # original untouched
