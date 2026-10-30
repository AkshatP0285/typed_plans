from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src")); sys.path.insert(0, str(REPO / "lora"))

from bfcl_reviewer_metrics import _plan_calls, _score_calls, load_gold
from tpj.benchmarks.bfcl import BFCLBenchmark
from tpj.faults.injector import FaultInjector
from tpj.gate.base import INSTRUCTIONS, LABELS, OKAY, GateInput, render_state
from tpj.gate.jev import JEVGate
from tpj.ir.formats import FreeTextFormat, GraphFormat
from tpj.ir.text_view import REF_NOTE as IR_TEXT_NOTE, ir_text
from tpj.ir.schema import Plan, Step

RUNS = {"typed_ir": "outputs/paper_nonlive/base_27brev_full", "graph": "outputs/format_ablation/base_27brev_graph",
        "free_text": "outputs/format_ablation/base_27brev_free_text_rule"}
IR_REF = ('an argument {"$ref": "sN", "path": "field.sub"} means an output of step sN')
VIEWS = ["typed_ir", "typed_ir_compact", "ir_text", "graph", "free_text"]
REF_NOTE = {"typed_ir": IR_REF, "typed_ir_compact": IR_REF,
            "ir_text": IR_TEXT_NOTE,
            "graph": 'the plan is a JSON graph; an argument "{{sN}}" or "{{sN.field}}" means an output of '
                     'node sN, and edges list dependencies',
            "free_text": 'the plan is numbered plain-text lines; "the result of step N" means an output of step N'}
assert IR_REF in INSTRUCTIONS["plan"]


def render(plan: Plan, fmt: str):
    if fmt == "ir_text":
        return ir_text(plan)
    if fmt in ("typed_ir", "typed_ir_compact"):
        return json.loads(plan.to_prompt_json(compact=fmt == "typed_ir_compact"))
    if not plan.steps:
        return {"graph": GraphFormat().render(plan), "free_text": "No tool needed."}[fmt]
    return (GraphFormat() if fmt == "graph" else FreeTextFormat()).render(plan)


def body(gate: JEVGate, query: str, tools, plan: Plan, fmt: str) -> dict:
    fewshot = []
    for ex in gate.plan_fewshot:
        ex = dict(ex)
        ex["plan"] = render(Plan.model_validate({**ex["plan"], "query": ex["user_query"]}), fmt) if fmt != "typed_ir" else ex["plan"]
        fewshot.append(ex)
    case = {"user_query": query, "available_tools": [{"name": t.name, "description": t.description,
                                                      "parameters": t.parameters} for t in tools],
            "plan": render(plan, fmt)}
    return {"state": {"labeled_examples": fewshot, "case_to_classify": case}, "model": gate.model,
            "questions": {"outcome": {"type": "choice", "instructions": INSTRUCTIONS["plan"].replace(IR_REF, REF_NOTE[fmt]),
                                      "criteria": LABELS["plan"]}}}


POST_FAULTS = ["empty", "wrong_type", "schema_drift"]


def post_body(gate: JEVGate, task, plan: Plan, step: Step, output, fmt: str) -> dict:
    inp = GateInput(phase="post", query=task.query, plan=plan, tools=task.tools, step=step, args=step.args, output=output)
    if fmt == "ir_text":
        spec = next((t for t in task.tools if t.name == step.tool), None)
        parts = [f"User request: {task.query}", "Step: " + ir_text(Plan(query=task.query, steps=[step.model_copy(update={"after": [], "when": None})], final=[step.id])).splitlines()[0]]
        if spec is not None:
            parts.append(f"Tool description: {spec.description[:600]}")
        parts.append("Output: " + json.dumps(output, default=str, ensure_ascii=False)[:15000])
        state = "\n".join(parts)
    elif fmt in ("typed_ir", "typed_ir_compact"):
        state = render_state(inp, 30000)
        if fmt == "typed_ir_compact":
            state = state.replace(" (declared return type Any)", "")
    else:
        one = Plan(query=task.query, steps=[step.model_copy(update={"after": [], "when": None})], final=[step.id])
        text = render(one, fmt)
        if fmt == "free_text":
            text = text.splitlines()[0]
        spec = next((t for t in task.tools if t.name == step.tool), None)
        parts = [f"User request: {task.query}", "Step: " + text]
        if spec is not None:
            parts.append(f"Tool description: {spec.description[:600]}")
        parts.append("Output: " + json.dumps(output, default=str, ensure_ascii=False)[:15000])
        state = "\n".join(parts)
    return {"state": state, "model": gate.model, "questions": {"outcome": {
        "type": "choice", "instructions": INSTRUCTIONS["post"], "criteria": LABELS["post"]}}}


async def post_check(gate, tasks, ask) -> dict:
    inj = FaultInjector(seed=0)
    cases = []
    for line in open(REPO / RUNS["typed_ir"] / "results.jsonl"):
        r = json.loads(line)
        if r["task_id"] not in tasks or not r.get("plan") or not r.get("final"):
            continue
        task = tasks[r["task_id"]]
        plan = Plan.model_validate({**r["plan"], "query": task.query})
        for step in plan.steps:
            if step.id not in r["final"]:
                continue
            out, rng = r["final"][step.id], inj._rng(r["task_id"], step.id, "post")
            kind = rng.choice(POST_FAULTS) if rng.random() < 0.5 else None
            if kind:
                out = inj.apply_exec(kind, out, r["task_id"], step.id)
            cases.append((task, plan, step, out, kind))
    res_all = {}
    for fmt in VIEWS:
        res = await asyncio.gather(*(ask(post_body(gate, t, p, s, o, fmt), "post") for t, p, s, o, _ in cases))
        ok = [(k, f) for (*_, k), (f, _, _) in zip(cases, res) if f is not None]
        bad = [f for k, f in ok if k]; clean = [f for k, f in ok if not k]
        res_all[f"POST / JEV reads {fmt}"] = dict(
            n=len(ok), failed=len(cases) - len(ok), faulted=len(bad),
            catches=round(100 * sum(bad) / max(1, len(bad)), 1),
            false_alarm=round(100 * sum(clean) / max(1, len(clean)), 1),
            balanced_acc=round(50 * (sum(bad) / max(1, len(bad)) + 1 - sum(clean) / max(1, len(clean))), 1),
            by_fault={k: round(100 * sum(f for kk, f in ok if kk == k) / max(1, sum(kk == k for kk, _ in ok)), 1) for k in POST_FAULTS},
            cached=sum(c for _, c, _ in res), cost_usd=round(sum(x for _, _, x in res), 4))
        print(f"POST | JEV reads {fmt:9s} " + json.dumps(res_all[f"POST / JEV reads {fmt}"]), flush=True)
    return res_all


async def main():
    gate = JEVGate(model="typesafe/jev-1.13", url="https://openrouter.ai/api/alpha/decisions",
                   api_key_env="OPENROUTER_API_KEY", cache_dir=str(REPO / ".cache/jev"),
                   plan_fewshot_file=str(REPO / "data/jev_fewshot/plan_verification_fewshot.json"))
    cats = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance"]
    gold = load_gold(cats)
    tasks = {t.id: t for t in BFCLBenchmark(data_dir=os.environ["BFCL_DATA_DIR"], categories=cats).load()}
    sem = asyncio.Semaphore(24)

    async def ask(b, phase="plan"):
        async with sem:
            for attempt in range(4):
                try:
                    data, cached = await gate._fetch(b)
                    p = JEVGate._extract_probs(data, phase)
                    tot = sum(p.get(k, 0.0) for k in LABELS[phase]) or 1.0
                    return (1 - p.get(OKAY, 0.0) / tot) >= 0.5, cached, (data.get("usage") or {}).get("cost") or 0.0
                except Exception:
                    await asyncio.sleep(2 ** attempt)
            return None, False, 0.0

    out = {}
    for run_fmt, run_dir in RUNS.items():
        rows = [json.loads(line) for line in open(REPO / run_dir / "results.jsonl")]
        rows = [r for r in rows if r["task_id"] in gold and r.get("initial_plan") is not None]
        wrong = [_score_calls(_plan_calls(r["initial_plan"]), gold[r["task_id"]][0]) < 1 for r in rows]
        stored = {}
        for line in open(REPO / run_dir / "trace.jsonl"):
            e = json.loads(line)
            if e.get("type") == "gate" and e.get("phase") == "plan" and e["task_id"] not in stored:
                stored[e["task_id"]] = e["flagged"]
        for fmt in VIEWS:
            bodies = [body(gate, tasks[r["task_id"]].query, tasks[r["task_id"]].tools,
                           Plan.model_validate({**r["initial_plan"], "query": tasks[r["task_id"]].query}), fmt)
                      for r in rows]
            res = await asyncio.gather(*(ask(b) for b in bodies))
            ok = [(w, f) for w, (f, _, _) in zip(wrong, res) if f is not None]
            tp = sum(w and f for w, f in ok); fn = sum(w and not f for w, f in ok)
            fp = sum(f and not w for w, f in ok); tn = sum(not w and not f for w, f in ok)
            agree = sum(stored.get(r["task_id"]) == f for r, (f, _, _) in zip(rows, res) if f is not None)
            out[f"{run_fmt} plans / JEV reads {fmt}"] = dict(
                n=len(ok), failed=len(rows) - len(ok), wrong=tp + fn,
                catches=round(100 * tp / max(1, tp + fn), 1), false_alarm=round(100 * fp / max(1, fp + tn), 1),
                precision=round(100 * tp / max(1, tp + fp), 1),
                balanced_acc=round(50 * (tp / max(1, tp + fn) + tn / max(1, fp + tn)), 1),
                same_as_pipeline=round(100 * agree / max(1, len(ok)), 1),
                cached=sum(c for _, c, _ in res), cost_usd=round(sum(x for _, _, x in res), 4))
            print(f"{run_fmt:9s} plans | JEV reads {fmt:9s} " + json.dumps(out[f'{run_fmt} plans / JEV reads {fmt}']), flush=True)
    out.update(await post_check(gate, tasks, ask))
    (REPO / "outputs/format_ablation/jev_format_input.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    asyncio.run(main())
