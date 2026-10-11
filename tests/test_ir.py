import pytest

from tpj.ir import PlanStructureError, check_structure, dirty_region, resolve, waves
from tpj.ir.examples import GOA_PLAN, GOA_WAVES
from tpj.ir.refs import RefResolutionError, get_path


def test_goa_waves_match_figure_3():
    assert waves(GOA_PLAN) == GOA_WAVES


def test_structure_ok():
    check_structure(GOA_PLAN)


def test_dangling_ref_detected():
    bad = GOA_PLAN.model_copy(deep=True)
    bad.steps = [s for s in bad.steps if s.id != "s5"]
    with pytest.raises(PlanStructureError):
        check_structure(bad)


def test_resolve_paths():
    outputs = {"s3": [{"id": "H1"}, {"id": "H2"}], "s4": {"rain_chance": 0.7}, "s9": None}
    assert resolve({"$ref": "s3", "path": "first.id"}, outputs) == "H1"
    assert resolve({"$ref": "s3", "path": "-1.id"}, outputs) == "H2"
    assert resolve([{"$ref": "s4", "path": "rain_chance"}, {"$ref": "s9"}], outputs) == [0.7, None]
    assert get_path(5, "result") == 5  # lenient scalar unwrap
    with pytest.raises(RefResolutionError):
        resolve({"$ref": "s1"}, outputs)


def test_dirty_region_is_changed_plus_descendants():
    new = GOA_PLAN.model_copy(deep=True)
    new.steps[3].args["when"] = "today"  # s4 get_weather
    assert dirty_region(GOA_PLAN, new) == {"s4", "s6", "s9", "s10", "s11"}
