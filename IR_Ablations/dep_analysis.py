from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tpj.ir.dag import PlanStructureError, waves
from tpj.ir.schema import Plan

FMTS = ["typed_ir", "graph", "free_text_rule"]
RUNS = {
    ("nestful", None): {f: f"outputs/format_ablation_nestful/{f}" for f in FMTS},
    ("taskbench", "dailylifeapis"): {f: f"outputs/dep_taskbench/dailylifeapis_{f}" for f in FMTS},
    ("taskbench", "huggingface"): {f: f"outputs/dep_taskbench/huggingface_{f}" for f in FMTS},
    ("bfcl", "parallel"): {"typed_ir": "outputs/paper_nonlive/base_27brev_full",
                           "graph": "outputs/format_ablation/base_27brev_graph",
                           "free_text_rule": "outputs/format_ablation/base_27brev_free_text_rule"},
}


def align(pred: Plan, gold: Plan) -> dict[str, str]:
    used, m = set(), {}
    for p in pred.steps:
        g = next((g for g in gold.steps if g.tool == p.tool and g.id not in used), None)
        if g:
            used.add(g.id); m[p.id] = g.id
    return m


def edges(plan: Plan) -> set[tuple[str, str]]:
    return {(d, s.id) for s in plan.steps for d in s.deps()}


def n_waves(plan: Plan) -> int | None:
    try:
        return len(waves(plan)) if plan.steps else 0
    except PlanStructureError:
        return None


def compare(pred: Plan, gold: Plan) -> dict:
    m = align(pred, gold)
    aligned = set(m.values())
    g_e = {e for e in edges(gold) if e[0] in aligned and e[1] in aligned}
    p_e = {(m[a], m[b]) for a, b in edges(pred) if a in m and b in m}
    return {"gold": len(g_e), "hit": len(g_e & p_e), "extra": len(p_e - g_e),
            "waves": n_waves(pred), "gold_waves": n_waves(gold), "same_tools": [s.tool for s in pred.steps] == [s.tool for s in gold.steps]}


def gold_plans(bench, subset):
    if bench == "nestful":
        from tpj.benchmarks.nestful import NestfulBenchmark
        return {t.id: t.gold_plan for t in NestfulBenchmark(hf_path="ibm-research/nestful", split="train",
                                                            limit=None, sample_seed=None).load()}
    if bench == "taskbench":
        from tpj.benchmarks.taskbench import TaskBenchBenchmark
        return {t.id: t.gold_plan for t in TaskBenchBenchmark(subset=subset, sample_seed=None).load()}
    return None


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    g = sum(r["gold"] for r in rows); h = sum(r["hit"] for r in rows); x = sum(r["extra"] for r in rows)
    st = [r for r in rows if r["same_tools"]]
    ok = [r for r in st if r["waves"] is not None and r["gold_waves"] is not None]
    return {
        "plans": n,
        "dep_recall": round(100 * h / max(1, g), 1),
        "dep_precision": round(100 * h / max(1, h + x), 1),
        "plans_missing_dep": round(100 * sum(r["hit"] < r["gold"] for r in rows) / max(1, n), 1),
        "plans_extra_dep": round(100 * sum(r["extra"] > 0 for r in rows) / max(1, n), 1),
        "right_tools": len(st),
        "waves_vs_gold": round(sum(r["waves"] - r["gold_waves"] for r in ok) / max(1, len(ok)), 2),
        "same_waves": round(100 * sum(r["waves"] == r["gold_waves"] for r in ok) / max(1, len(ok)), 1),
    }


def main():
    out = {}
    for (bench, subset), runs in RUNS.items():
        gold = gold_plans(bench, subset)
        for fmt, d in runs.items():
            path = REPO / d / "results.jsonl"
            if not path.exists():
                continue
            rows = []
            for line in open(path):
                r = json.loads(line)
                plan = r.get("initial_plan")
                if plan is None:
                    continue
                pred = Plan.model_validate({**plan, "query": ""})
                if bench == "bfcl":
                    if r.get("category") not in ("parallel", "parallel_multiple"):
                        continue
                    gp = Plan(query="", steps=[s.model_copy(update={"after": [], "args": {}, "when": None}) for s in pred.steps],
                              final=[])
                elif r["task_id"] in gold:
                    gp = gold[r["task_id"]]
                else:
                    continue
                rows.append(compare(pred, gp))
            key = f"{bench}{'/' + subset if subset else ''} | {fmt}"
            out[key] = summarize(rows)
            print(f"{key:36s} {json.dumps(out[key])}")
    (REPO / "outputs/format_ablation/dep_analysis.json").write_text(json.dumps(out, indent=1))


def selftest():
    from tpj.ir.schema import Step
    gold = Plan(query="", steps=[Step(id="s1", tool="a"), Step(id="s2", tool="b", args={"x": {"$ref": "s1"}}),
                                 Step(id="s3", tool="c")], final=["s2", "s3"])
    same = compare(gold, gold)
    assert same == {"gold": 1, "hit": 1, "extra": 0, "waves": 2, "gold_waves": 2, "same_tools": True}
    flat = Plan(query="", steps=[Step(id="s1", tool="a"), Step(id="s2", tool="b"), Step(id="s3", tool="c", after=["s2"])], final=[])
    c = compare(flat, gold)
    assert c["hit"] == 0 and c["extra"] == 1 and c["waves"] == 2
    s = summarize([same, c])
    assert s["dep_recall"] == 50.0 and s["plans_missing_dep"] == 50.0 and s["plans_extra_dep"] == 50.0
    print("selftest ok")


if __name__ == "__main__":
    selftest() if sys.argv[1:] == ["--selftest"] else main()
