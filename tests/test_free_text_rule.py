import pytest

from tpj.ir.formats.base import PlanParseError
from tpj.ir.formats.free_text import FreeTextFormat, parse_rule


def test_rule_parser_reads_text_without_llm():
    t = ('1. Call get_weather with city = "Paris, FR", days = 3.\n'
         '2. Call send_email with to = "a@b.c", body = the result of step 1 (field summary) only if step 1 is true.\n'
         'Answer: steps 2')
    p = parse_rule(t, "q")
    assert p.steps[0].args == {"city": "Paris, FR", "days": 3}
    assert p.steps[1].args["body"] == {"$ref": "s1", "path": "summary"}
    assert p.steps[1].when.ref == "s1" and p.final == ["s2"]
    assert parse_rule(FreeTextFormat().render(p), "q").model_dump() == p.model_dump()
    assert parse_rule("No tool needed.", "q").steps == []
    q = parse_rule('1. Call a with x = 1.\n2. Call b with no arguments after step 1.', "q")
    assert q.steps[1].after == ["s1"] and parse_rule(FreeTextFormat().render(q), "q").model_dump() == q.model_dump()
    assert parse_rule("1. Call ping with no arguments.", "q").steps[0].args == {}
    assert parse_rule('1. Call Image Classification with image = "a.jpg".', "q").steps[0].tool == "Image Classification"
    with pytest.raises(PlanParseError):
        parse_rule("I would check the weather.", "q")
