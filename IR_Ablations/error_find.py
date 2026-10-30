from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tpj.benchmarks.bfcl import BFCLBenchmark, _safe_name
from tpj.benchmarks.taskbench import TaskBenchBenchmark
from tpj.ir.formats import FreeTextFormat, GraphFormat, TypedIRFormat
from tpj.ir.refs import is_ref
from tpj.ir.schema import Plan, Step
from tpj.llm.client import LLMClient, parse_json

FORMATS = {"typed_ir": TypedIRFormat(), "graph": GraphFormat(), "free_text": FreeTextFormat()}
TYPES = ["wrong_tool", "wrong_value", "missing_arg", "wrong_dependency", "missing_step", "extra_step"]
REVIEWERS = {"27b": ("Qwen/Qwen3.8-27B", "VLLM_27B_URL", "http://10.20.213.80:8004/v1"),
             "4b": ("Qwen/Qwen3-4B", "VLLM_4B_URL", "http://10.20.213.80:8002/v1")}

PROMPT = """\
You review a plan for a tool-calling agent before it runs. The plan contains either no error or exactly one.
Error types:
- wrong_tool: a step calls a tool that does not do what the request asks
- wrong_value: an argument value contradicts the request
- missing_arg: a step leaves out a required argument
- wrong_dependency: a step waits for or uses the result of the wrong step
- missing_step: an action the request asks for has no step
- extra_step: a step the request does not ask for

The plan is written in this format:
{rules}

Tools:
{tools}

Request: {query}

Plan:
{plan}

Reply as JSON: {{"has_error": true or false, "step": <number of the step with the error, or null>, "type": <error type, or null>}}"""

SCHEMA = {"type": "object", "properties": {
    "has_error": {"type": "boolean"}, "step": {"type": ["integer", "null"]},
    "type": {"type": ["string", "null"], "enum": TYPES + [None]}}, "required": ["has_error", "step", "type"]}


def bfcl_items(n, rng):
    bench = BFCLBenchmark(data_dir=os.environ["BFCL_DATA_DIR"], sample_seed=0)
    out = []
    for t in bench.load():
        steps = []
        for i, call in enumerate(t.gold["calls"], 1):
            (fn, allowed), = call.items()
            args = {k: next(v for v in vals if v != "") for k, vals in allowed.items() if any(v != "" for v in vals)}
            steps.append(Step(id=f"s{i}", tool=_safe_name(fn), args=args))
        if steps:
            out.append(("bfcl", t, Plan(query=t.query, steps=steps, final=[s.id for s in steps])))
    return out[:n]


def taskbench_items(subset, n):
    bench = TaskBenchBenchmark(subset=subset, sample_seed=0)
    tasks = [t for t in bench.load() if len(t.gold_plan.steps) >= 2 and any(s.deps() for s in t.gold_plan.steps)]
    return [(subset, t, t.gold_plan) for t in tasks[:n]]


def _literals(step):
    return [k for k, v in step.args.items() if not is_ref(v) and isinstance(v, (str, int, float, bool, list))]


def _renumber(plan: Plan) -> Plan:
    m = {s.id: f"s{i}" for i, s in enumerate(plan.steps, 1)}

    def fix(v):
        if is_ref(v):
            return {**v, "$ref": m[v["$ref"]]}
        if isinstance(v, dict):
            return {k: fix(x) for k, x in v.items()}
        if isinstance(v, list):
            return [fix(x) for x in v]
        return v
    for s in plan.steps:
        s.id, s.args, s.after = m[s.id], fix(s.args), [m[a] for a in s.after if a in m]
    plan.final = [m[f] for f in plan.final if f in m]
    return plan


def _other_value(v, pool, rng):
    if isinstance(v, bool):
        return not v
    if isinstance(v, (int, float)):
        return v * 2 + 7
    if isinstance(v, list):
        return v[:-1] if len(v) > 1 else None
    cands = [p for p in pool if isinstance(p, str) and p != v]
    return rng.choice(cands) if cands else None


def plant(plan: Plan, tools, kind, pool, rng):
    p = copy.deepcopy(plan)
    steps = p.steps
    if kind == "wrong_tool":
        used = {s.tool for s in steps}
        others = [t.name for t in tools if t.name not in used]
        if not others:
            return None
        i = rng.randrange(len(steps)); steps[i].tool = rng.choice(others)
        return p, i + 1
    if kind in ("wrong_value", "missing_arg"):
        cand = [(i, k) for i, s in enumerate(steps) for k in _literals(s)]
        rng.shuffle(cand)
        for i, k in cand:
            if kind == "missing_arg":
                del steps[i].args[k]
                return p, i + 1
            new = _other_value(steps[i].args[k], pool.get(k, []) or pool["*"], rng)
            if new is not None and new != steps[i].args[k]:
                steps[i].args[k] = new
                return p, i + 1
        return None
    if kind == "wrong_dependency":
        cand = [i for i, s in enumerate(steps) if s.after]
        if not cand:
            return None
        i = rng.choice(cand); s = steps[i]
        ancestors = set()
        stack = [s.id]
        while stack:
            cur = stack.pop()
            for o in steps:
                if cur in o.deps() and o.id not in ancestors:
                    ancestors.add(o.id); stack.append(o.id)
        alt = [o.id for o in steps if o.id != s.id and o.id not in s.deps() and o.id not in ancestors]
        old = rng.choice(s.after)
        s.after = [a for a in s.after if a != old] + ([rng.choice(alt)] if alt else [])
        return p, i + 1
    if kind == "missing_step":
        leaves = [s for s in steps if s.id not in {d for o in steps for d in o.deps()}]
        if len(steps) < 2 or not leaves:
            return None
        steps.remove(rng.choice(leaves))
        return _renumber(p), None
    if kind == "extra_step":
        others = [t for t in tools if t.name not in {s.tool for s in steps}]
        if not others:
            return None
        t = rng.choice(others)
        names = list((t.to_prompt().get("parameters") or {}).get("properties", {}))[:2]
        args = {name: rng.choice(pool.get(name) or pool["*"]) for name in names}
        steps.append(Step(id=f"s{len(steps) + 1}", tool=t.name, args=args))
        p.final.append(steps[-1].id)
        return p, len(steps)
    raise ValueError(kind)


def build_cases(items, rng):
    pool = defaultdict(list)
    for _, _, plan in items:
        for s in plan.steps:
            for k in _literals(s):
                pool[k].append(s.args[k]); pool["*"].append(s.args[k])
    cases = []
    for bench, task, plan in items:
        cases.append(dict(bench=bench, task=task, plan=plan, orig=plan, kind=None, step=None))
        for kind in rng.sample(TYPES, len(TYPES)):
            r = plant(plan, task.tools, kind, pool, rng)
            if r:
                cases.append(dict(bench=bench, task=task, plan=r[0], orig=plan, kind=kind, step=r[1]))
                break
    return cases


def _tools_json(task, plan, orig):
    tools = task.tools
    if len(tools) > 12:
        used = {s.tool for s in plan.steps} | {s.tool for s in orig.steps}
        tools = [t for t in tools if t.name in used] + [t for t in tools if t.name not in used][: 12]
        tools = sorted(tools, key=lambda t: t.name)
    return json.dumps([t.to_prompt() for t in tools], indent=1)


async def review(llm, case, fmt_name):
    fmt = FORMATS[fmt_name]
    prompt = PROMPT.format(rules=fmt.instructions(), tools=_tools_json(case["task"], case["plan"], case["orig"]),
                           query=case["task"].query, plan=fmt.render(case["plan"]))
    res = await llm.chat([{"role": "user", "content": prompt}], role="judge", json_schema=SCHEMA, schema_name="review")
    try:
        out = parse_json(res.text)
    except ValueError:
        out = {"has_error": False, "step": None, "type": None}
    return {"has_error": bool(out.get("has_error")), "step": out.get("step"), "type": out.get("type"),
            "prompt_tokens": res.prompt_tokens, "plan_chars": len(fmt.render(case["plan"]))}


def summarize(rows):
    by = defaultdict(list)
    for r in rows:
        by[(r["bench"], r["format"])].append(r); by[("ALL", r["format"])].append(r)
    out = {}
    for key, rs in sorted(by.items()):
        bad = [r for r in rs if r["kind"]]; clean = [r for r in rs if not r["kind"]]
        loc = [r for r in bad if r["step"] is not None]
        recall = sum(r["has_error"] for r in bad) / max(1, len(bad))
        fa = sum(r["has_error"] for r in clean) / max(1, len(clean))
        out["/".join(key)] = {
            "n_broken": len(bad), "n_clean": len(clean),
            "found": round(100 * recall, 1), "false_alarm": round(100 * fa, 1),
            "balanced_acc": round(100 * (recall + 1 - fa) / 2, 1),
            "right_step": round(100 * sum(r["has_error"] and r["pred_step"] == r["step"] for r in loc) / max(1, len(loc)), 1),
            "right_type": round(100 * sum(r["has_error"] and r["pred_type"] == r["kind"] for r in bad) / max(1, len(bad)), 1),
            "prompt_tokens": round(sum(r["prompt_tokens"] for r in rs) / len(rs)),
            "plan_chars": round(sum(r["plan_chars"] for r in rs) / len(rs)),
            "found_by_type": {k: round(100 * sum(r["has_error"] for r in bad if r["kind"] == k) /
                                       max(1, sum(r["kind"] == k for r in bad)), 1)
                              for k in TYPES if any(r["kind"] == k for r in bad)},
        }
    return out


async def main(a):
    rng = random.Random(0)
    items = bfcl_items(a.n, rng) + taskbench_items("dailylifeapis", a.n) + taskbench_items("huggingface", a.n)
    cases = build_cases(items, rng)
    model, env, default = REVIEWERS[a.reviewer]
    llm = LLMClient(model=model, base_url=os.environ.get(env, default), api_key_env=None, max_tokens=200,
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}}, max_concurrency=48)
    jobs = [(c, f) for c in cases for f in FORMATS]
    outs = await asyncio.gather(*(review(llm, c, f) for c, f in jobs))
    rows = [dict(bench=c["bench"], format=f, kind=c["kind"], step=c["step"], task=c["task"].id,
                 pred_step=o["step"], pred_type=o["type"], has_error=o["has_error"],
                 prompt_tokens=o["prompt_tokens"], plan_chars=o["plan_chars"]) for (c, f), o in zip(jobs, outs)]
    out = Path(a.out) / a.reviewer
    out.mkdir(parents=True, exist_ok=True)
    (out / "rows.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    summ = summarize(rows)
    (out / "summary.json").write_text(json.dumps(summ, indent=1))
    print(f"{'set/format':28s} found  false_alarm  bal_acc  right_step  right_type  tokens")
    for k, v in summ.items():
        print(f"{k:28s} {v['found']:5.1f}  {v['false_alarm']:11.1f}  {v['balanced_acc']:7.1f}  "
              f"{v['right_step']:10.1f}  {v['right_type']:10.1f}  {v['prompt_tokens']:6d}")


def selftest():
    rng = random.Random(1)
    from tpj.tools.spec import ToolSpec
    tools = [ToolSpec(name=n, description=n, parameters={"type": "object", "properties": {"x": {"type": "string"}}})
             for n in ("a", "b", "c", "d")]
    plan = Plan(query="q", steps=[Step(id="s1", tool="a", args={"x": "1"}), Step(id="s2", tool="b", args={"x": {"$ref": "s1"}}),
                                  Step(id="s3", tool="c", args={"x": "z"}, after=["s1"])], final=["s2", "s3"])
    pool = {"x": ["1", "z", "q"], "*": ["1", "z", "q"]}
    for kind in TYPES:
        r = plant(plan, tools, kind, pool, rng)
        assert r, kind
        broken, step = r
        assert broken.model_dump() != plan.model_dump(), kind
        for f in FORMATS.values():
            assert f.render(broken) != f.render(plan), (kind, f.name)
    broken, _ = plant(plan, tools, "missing_step", pool, random.Random(0))
    assert [s.id for s in broken.steps] == ["s1", "s2"]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reviewer", default="27b", choices=list(REVIEWERS))
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--out", default="outputs/error_find")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    selftest() if a.selftest else asyncio.run(main(a))
