from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lora"))
from error_find import FORMATS, TYPES, bfcl_items, build_cases, taskbench_items
from tpj.ir.dag import PlanStructureError, check_structure
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan

_JSON = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def check(plan: Plan, tools) -> list[str]:
    problems = []
    try:
        check_structure(plan)
    except PlanStructureError as e:
        problems.append(f"structure: {e}")
    spec = {t.name: t for t in tools}
    seen = set()
    for s in plan.steps:
        key = (s.tool, json.dumps(s.args, sort_keys=True, default=str))
        if key in seen:
            problems.append(f"{s.id}: duplicate call")
        seen.add(key)
        if s.tool not in spec:
            problems.append(f"{s.id}: unknown tool {s.tool}")
            continue
        schema = spec[s.tool].parameters or {}
        props = schema.get("properties") or {}
        if props:
            for k in set(s.args) - set(props):
                problems.append(f"{s.id}: unknown argument {k}")
        for k in schema.get("required") or []:
            if k not in s.args:
                problems.append(f"{s.id}: missing required argument {k}")
        for k, v in s.args.items():
            want = _JSON.get((props.get(k) or {}).get("type"))
            if want and not is_ref(v) and not (isinstance(v, want) and not (want is int and isinstance(v, bool))):
                if not (want == (int, float) and isinstance(v, bool)):
                    problems.append(f"{s.id}: {k} should be {props[k]['type']}")
    return problems


async def planted():
    rng = random.Random(0)
    n = 150
    items = bfcl_items(n, rng) + taskbench_items("dailylifeapis", n) + taskbench_items("huggingface", n)
    cases = build_cases(items, rng)
    out = {}
    for fname, fmt in FORMATS.items():
        by = defaultdict(lambda: Counter())
        for c in cases:
            plan = await fmt.parse(fmt.render(c["plan"]), c["task"].query)
            flagged = bool(check(plan, c["task"].tools))
            for key in (c["bench"], "ALL"):
                b = by[key]
                if c["kind"]:
                    b["broken"] += 1; b["caught"] += flagged; b[f"n_{c['kind']}"] += 1; b[f"c_{c['kind']}"] += flagged
                else:
                    b["clean"] += 1; b["false_alarm"] += flagged
        out[fname] = {k: {"caught": round(100 * v["caught"] / max(1, v["broken"]), 1),
                          "false_alarm": round(100 * v["false_alarm"] / max(1, v["clean"]), 1),
                          "by_type": {t: round(100 * v[f"c_{t}"] / v[f"n_{t}"], 1) for t in TYPES if v[f"n_{t}"]}}
                      for k, v in by.items()}
    print(json.dumps(out, indent=1))
    Path("outputs/error_find/static_planted.json").write_text(json.dumps(out, indent=1))


def real(run_dir):
    from bfcl_reviewer_metrics import _plan_calls, _score_calls, load_gold
    from tpj.benchmarks.bfcl import BFCLBenchmark
    rows = [json.loads(line) for line in open(Path(run_dir) / "results.jsonl")]
    cats = sorted({r["category"] for r in rows if r.get("category")})
    gold = load_gold(cats)
    tools = {t.id: t.tools for t in BFCLBenchmark(data_dir=os.environ["BFCL_DATA_DIR"], categories=cats).load()}
    c = Counter()
    for r in rows:
        if r["task_id"] not in gold or not r.get("initial_plan"):
            continue
        plan = Plan.model_validate({**r["initial_plan"], "query": r["initial_plan"].get("query", "")})
        right = _score_calls(_plan_calls(r["initial_plan"]), gold[r["task_id"]][0]) >= 1
        flagged = bool(check(plan, tools[r["task_id"]]))
        c["right" if right else "wrong"] += 1
        c["flag_right" if right else "flag_wrong"] += flagged
    print(f"{run_dir}: wrong plans {c['wrong']}, caught by rules {100 * c['flag_wrong'] / max(1, c['wrong']):.1f}% | "
          f"right plans {c['right']}, false alarms {100 * c['flag_right'] / max(1, c['right']):.1f}%")


def selftest():
    from tpj.ir.schema import Step
    from tpj.tools.spec import ToolSpec
    t = [ToolSpec(name="f", description="", parameters={"type": "object", "properties": {"n": {"type": "integer"}},
                                                        "required": ["n"]})]
    ok = Plan(query="q", steps=[Step(id="s1", tool="f", args={"n": 3})], final=["s1"])
    assert check(ok, t) == []
    assert check(Plan(query="q", steps=[Step(id="s1", tool="f", args={})], final=["s1"]), t)
    assert check(Plan(query="q", steps=[Step(id="s1", tool="f", args={"n": "3"})], final=["s1"]), t)
    assert check(Plan(query="q", steps=[Step(id="s1", tool="g", args={"n": 3})], final=["s1"]), t)
    assert check(Plan(query="q", steps=[Step(id="s1", tool="f", args={"n": {"$ref": "s9"}})], final=["s1"]), t)
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
    elif sys.argv[1] == "planted":
        asyncio.run(planted())
    else:
        real(sys.argv[2])
