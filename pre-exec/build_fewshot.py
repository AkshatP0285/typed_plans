"""Pick compact few-shot demonstrations from base plans that are NOT in the frozen test set.

Each example = user_query + plan + verdict (+ a one-line reason for INVALID ones), no tool schemas
(JEV's state limit is 8,000 chars and the test case itself already carries the schemas).
Output: test_set/fewshot_examples.json
"""
from __future__ import annotations

import collections
import json
import random
from pathlib import Path

import build_pre_cases as b

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "test_set" / "pre_exec_test_set.jsonl"
OUT = ROOT / "test_set" / "fewshot_examples.json"
SEED = 11
# (kind, source) per demonstration: 3 VALID + 3 INVALID, spread across sources and edit kinds
PLAN = [("gt", "toolace"), ("remove", "toolace"), ("gt", "hermes"),
        ("replace_arg", "apigen_mt"), ("gt", "apigen_mt"), ("add_unrelated", "hermes")]
MAX_Q, MAX_PLAN = 300, 650   # chars: keep demonstrations short


def reason(edit: dict, kind: str) -> str:
    if kind == "remove":
        return f"missing step: the plan never calls {edit['removed_tool']}, which the request needs"
    if kind == "replace_arg":
        return f"wrong argument: {edit['arg']} is {edit['new']!r} but the request says {edit['old']!r}"
    if kind == "add_unrelated":
        return f"unneeded step: {edit['tool']} is not required by the request"
    return ""


def main():
    rng = random.Random(SEED)
    test_ids = {json.loads(l)["base_id"] for l in open(TEST, encoding="utf-8") if l.strip()}
    rows = [r for r in b.load(str(b.DEFAULT_SRC)) if r["source_id"] not in test_ids]
    pool = b.literal_pool(rows)
    elig = collections.defaultdict(list)
    for r in rows:
        steps, tools = r["generated_ir"]["steps"], r["tools"]
        if not tools or not r["structural_check"]["passed"] or len(steps) < 2:
            continue
        tools = b.trim_tools(steps, tools, rng)
        if b.validate(steps, tools):
            continue
        if len(r["query"]) > MAX_Q or len(json.dumps(steps, ensure_ascii=False)) > MAX_PLAN:
            continue
        r["tools"] = tools
        elig[r["source"]].append(r)
    mut = {"remove": b.m_remove, "replace_arg": b.m_replace_arg, "add_unrelated": b.m_add}
    examples, used = [], set()
    for kind, src in PLAN:
        for r in rng.sample(elig[src], len(elig[src])):
            if r["source_id"] in used:
                continue
            steps, tools = r["generated_ir"]["steps"], r["tools"]
            if kind == "gt":
                plan, edit, verdict = steps, {}, "VALID"
            else:
                fn = mut[kind]
                res = fn(steps, tools, rng, pool, query=r["query"]) if kind == "replace_arg" \
                    else fn(steps, tools, rng, pool)
                if res is None or (kind == "add_unrelated" and res[1].get("add_kind") != "unrelated_tool"):
                    continue
                plan, edit = res
                verdict = "INVALID"
                if kind == "replace_arg" and b.validate(plan, tools):
                    continue
            used.add(r["source_id"])
            ex = {"user_query": r["query"], "plan": {"steps": plan}, "verdict": verdict}
            if verdict == "INVALID":
                ex["reason"] = reason(edit, kind)
            examples.append({"_base_id": r["source_id"], "_kind": kind, **ex})
            break
    assert not ({e["_base_id"] for e in examples} & test_ids), "leak: demo plan is in the test set"
    # shuffle so verdicts alternate unpredictably but deterministically
    rng.shuffle(examples)
    OUT.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    chars = len(json.dumps([{k: v for k, v in e.items() if not k.startswith("_")} for e in examples],
                           ensure_ascii=False, separators=(",", ":")))
    print(f"{len(examples)} examples, {chars} chars (budget left for the case: {8000 - chars - 300})")
    for e in examples:
        print(" ", e["_kind"], e["verdict"], e["_base_id"], "|", e["user_query"][:70])


if __name__ == "__main__":
    main()
