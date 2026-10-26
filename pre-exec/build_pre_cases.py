"""Build pre-execution (plan-level) verification cases.

Base plans are the LLM-generated "GT" IRs in merged_ir_thinking_final.jsonl.
For every sampled base plan we emit the unmodified plan (label VALID) plus
edited copies (label INVALID):

  remove        delete one step nobody depends on (plan stays structurally valid)
  replace_tool  swap one step's tool for a different available tool, args kept
  replace_arg   swap one literal argument for a value from an unrelated sample
  add           insert an unneeded step (unused tool, else a duplicate of a step)

Each case also stores `det_issues`: what a deterministic rule-based validator
finds in the plan. That is the no-LLM baseline, and it splits the edits into
"a compiler would catch this" vs "only reading the query catches this".

Output: data/pre_cases.jsonl
"""
from __future__ import annotations

import argparse
import collections
import copy
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_SRC = ROOT.parent.parent / "merged_ir_thinking_final.jsonl"
OUT = ROOT / "data" / "pre_cases.jsonl"

STATE_LIMIT = 8000          # JEV's documented serialized-state limit
BASE_BUDGET = 6500          # leave room for the edit + prompt wrapper
REF = re.compile(r"\$s(\d+)")
SOURCE_QUOTA = {"toolace": 40, "apigen_mt": 35, "hermes": 25}


# ------------------------------------------------------------------ helpers
def is_ref(v) -> bool:
    return isinstance(v, str) and bool(REF.search(v))


def refs_in(v) -> set[str]:
    if isinstance(v, str):
        return {f"s{m}" for m in REF.findall(v)}
    if isinstance(v, dict):
        return set().union(*[refs_in(x) for x in v.values()]) if v else set()
    if isinstance(v, list):
        return set().union(*[refs_in(x) for x in v]) if v else set()
    return set()


def rewrite_refs(v, mapping):
    if isinstance(v, str):
        return REF.sub(lambda m: mapping.get(f"s{m.group(1)}", m.group(0)).replace("s", "$s", 1)
                       if f"s{m.group(1)}" in mapping else m.group(0), v)
    if isinstance(v, dict):
        return {k: rewrite_refs(x, mapping) for k, x in v.items()}
    if isinstance(v, list):
        return [rewrite_refs(x, mapping) for x in v]
    return v


def renumber(steps):
    """Give steps contiguous ids s1..sN (in list order) and remap refs/depends_on,
    so an edited plan carries no id gap that would give the edit away."""
    mapping = {s["id"]: f"s{i + 1}" for i, s in enumerate(steps)}
    out = []
    for s in steps:
        t = copy.deepcopy(s)
        t["id"] = mapping[s["id"]]
        t["depends_on"] = [mapping[d] for d in s["depends_on"] if d in mapping]
        t["args"] = rewrite_refs(s["args"], mapping)
        out.append(t)
    return out


def tool_map(tools):
    return {t["name"]: t for t in tools}


def schema(tool):
    p = tool.get("parameters") or {}
    return p.get("properties") or {}, p.get("required") or []


def validate(steps, tools):
    """Deterministic rule-based checks. Returns a list of issue strings."""
    issues, tm, seen = [], tool_map(tools), set()
    for s in steps:
        if s["id"] in seen:
            issues.append(f"duplicate_id:{s['id']}")
        t = tm.get(s["tool"])
        if t is None:
            issues.append(f"unknown_tool:{s['id']}")
        else:
            props, req = schema(t)
            for r in req:
                if r not in s["args"]:
                    issues.append(f"missing_required:{s['id']}.{r}")
            if props:
                for a in s["args"]:
                    if a not in props:
                        issues.append(f"unknown_arg:{s['id']}.{a}")
        for d in s["depends_on"]:
            if d not in seen:
                issues.append(f"bad_dependency:{s['id']}->{d}")
        for r in refs_in(s["args"]):
            if r not in seen:
                issues.append(f"dangling_ref:{s['id']}->{r}")
        seen.add(s["id"])
    return issues


def trim_tools(steps, tools, rng, n_distractors=3):
    """Keep every tool the plan uses plus a few unused ones (original order). Needed to fit
    JEV's 8k state limit (apigen_mt ships its whole tool suite); distractors keep
    replace_tool / add edits meaningful."""
    used = {s["tool"] for s in steps}
    unused = [t["name"] for t in tools if t["name"] not in used]
    keep = used | set(rng.sample(unused, min(n_distractors, len(unused))))
    return [t for t in tools if t["name"] in keep]


def state_size(query, tools, steps):
    return len(json.dumps({"q": query, "t": tools, "p": {"steps": steps}},
                          ensure_ascii=False, separators=(",", ":")))


# ----------------------------------------------------------------- loading
def load(path):
    rows = []
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        ir = r.get("generated_ir")
        if not (isinstance(ir, dict) and isinstance(ir.get("steps"), list)):
            continue
        rows.append(r)
    return rows


def literal_pool(rows):
    """param name -> {type name -> set of literal values} across the corpus."""
    pool = collections.defaultdict(lambda: collections.defaultdict(set))
    for r in rows:
        for s in r["generated_ir"]["steps"]:
            for k, v in s["args"].items():
                if isinstance(v, (str, int, float)) and not isinstance(v, bool) and not is_ref(v):
                    if isinstance(v, str) and len(v) > 60:
                        continue
                    pool[k][type(v).__name__].add(v)
    return pool


# --------------------------------------------------------------- mutations
def m_remove(steps, tools, rng, pool):
    referenced = set()
    for s in steps:
        referenced |= set(s["depends_on"]) | refs_in(s["args"])
    cands = [i for i, s in enumerate(steps) if s["id"] not in referenced]
    if not cands or len(steps) < 2:
        return None
    i = rng.choice(cands)
    new = renumber(steps[:i] + steps[i + 1:])
    return new, {"kind": "remove", "removed_tool": steps[i]["tool"], "position": i}


def m_replace_tool(steps, tools, rng, pool):
    i = rng.randrange(len(steps))
    used = {s["tool"] for s in steps}
    others = [t["name"] for t in tools if t["name"] != steps[i]["tool"]]
    if not others:
        return None
    unused = [n for n in others if n not in used]
    new_tool = rng.choice(unused or others)
    new = copy.deepcopy(steps)
    new[i]["tool"] = new_tool
    return renumber(new), {"kind": "replace_tool", "from": steps[i]["tool"], "to": new_tool, "position": i}


def m_replace_arg(steps, tools, rng, pool, query=""):
    # only edit values the query itself pins down, so the edit is verifiably wrong
    q = query.lower()
    slots = [(i, k) for i, s in enumerate(steps) for k, v in s["args"].items()
             if isinstance(v, (str, int, float)) and not isinstance(v, bool) and not is_ref(v)
             and str(v).lower() in q]
    rng.shuffle(slots)
    for i, k in slots:
        old = steps[i]["args"][k]
        alts = [x for x in pool.get(k, {}).get(type(old).__name__, ()) if x != old]
        if alts:
            new_val = rng.choice(sorted(alts, key=str))
        elif isinstance(old, (int, float)):
            new_val = old * 2 + 1
        else:
            continue
        new = copy.deepcopy(steps)
        new[i]["args"][k] = new_val
        return renumber(new), {"kind": "replace_arg", "tool": steps[i]["tool"], "arg": k,
                               "old": old, "new": new_val, "position": i}
    return None


def default_for(pname, pinfo, pool, rng):
    typ = (pinfo or {}).get("type", "string")
    pyt = {"integer": "int", "int": "int", "number": "float", "float": "float"}.get(typ, "str")
    alts = sorted(pool.get(pname, {}).get(pyt, ()), key=str)
    if alts:
        return rng.choice(alts)
    return {"int": 1, "float": 1.0}.get(pyt, "example")


def m_add(steps, tools, rng, pool):
    used = {s["tool"] for s in steps}
    # "think" is a harmless no-op scratchpad tool in apigen_mt, not a real fault when added
    unused = [t for t in tools if t["name"] not in used and t["name"] != "think"]
    new = copy.deepcopy(steps)
    if unused:
        t = rng.choice(unused)
        props, req = schema(t)
        args = {p: default_for(p, props.get(p), pool, rng) for p in req}
        step = {"id": "NEW", "tool": t["name"], "args": args, "depends_on": [], "danger": False}
        pos = rng.randrange(len(new) + 1)
        new.insert(pos, step)
        return renumber(new), {"kind": "add", "add_kind": "unrelated_tool", "tool": t["name"], "position": pos}
    i = rng.randrange(len(new))
    dup = copy.deepcopy(new[i])
    dup["id"] = "NEW"
    new.insert(i + 1, dup)
    return renumber(new), {"kind": "add", "add_kind": "duplicate", "tool": dup["tool"], "position": i + 1}


MUTATIONS = [m_remove, m_replace_tool, m_replace_arg, m_add]


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = load(args.src)
    pool = literal_pool(rows)
    elig, dropped = [], collections.Counter()
    for r in rows:
        steps = r["generated_ir"]["steps"]
        tools = trim_tools(steps, r["tools"], rng) if r["tools"] else []
        r["tools"] = tools
        if not r["tools"] or not r["structural_check"]["passed"] or len(steps) < 2:
            dropped["no_tools/failed_struct/<2_steps"] += 1
        elif validate(steps, tools):
            dropped["validator_flags_GT"] += 1       # GT must be deterministically clean
        elif state_size(r["query"], tools, steps) > BASE_BUDGET:
            dropped["too_large_for_8k_state"] += 1
        else:
            elig.append(r)
    print(f"eligible base plans: {len(elig)}/{len(rows)}; dropped: {dict(dropped)}")

    by_src = collections.defaultdict(list)
    for r in elig:
        by_src[r["source"]].append(r)
    scale = args.n / sum(SOURCE_QUOTA.values())
    base = []
    for s, q in SOURCE_QUOTA.items():
        base += rng.sample(by_src[s], min(round(q * scale), len(by_src[s])))
    rng.shuffle(base)

    cases, skipped = [], collections.Counter()
    for r in base:
        steps, tools = r["generated_ir"]["steps"], r["tools"]
        common = dict(base_id=r["source_id"], source=r["source"], query=r["query"], tools=tools,
                      n_steps=len(steps))
        cases.append(dict(common, case_id=f"{r['source_id']}__gt", kind="gt", label="VALID",
                          plan={"steps": steps}, det_issues=[], edit={}))
        for m in MUTATIONS:
            res = m(steps, tools, rng, pool, query=r["query"]) if m is m_replace_arg else m(steps, tools, rng, pool)
            if res is None:
                skipped[m.__name__] += 1
                continue
            new, edit = res
            if state_size(r["query"], tools, new) > STATE_LIMIT - 400:
                skipped[m.__name__ + "_size"] += 1
                continue
            key = edit["kind"] + (f"_{edit['add_kind']}" if "add_kind" in edit else "")
            cases.append(dict(common, case_id=f"{r['source_id']}__{key}", kind=key, label="INVALID",
                              plan={"steps": new}, det_issues=validate(new, tools), edit=edit))

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text("\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n",
                              encoding="utf-8")
    print(f"wrote {len(cases)} cases from {len(base)} base plans -> {args.out}")
    print("by kind:", dict(collections.Counter(c["kind"] for c in cases)))
    print("base by source:", dict(collections.Counter(r["source"] for r in base)))
    print("base by #steps:", dict(sorted(collections.Counter(min(len(r['generated_ir']['steps']), 6) for r in base).items())))
    print("skipped:", dict(skipped))
    inv = [c for c in cases if c["label"] == "INVALID"]
    flagged = collections.Counter(c["kind"] for c in inv if c["det_issues"])
    tot = collections.Counter(c["kind"] for c in inv)
    print("rule-based validator flags:", {k: f"{flagged[k]}/{tot[k]}" for k in tot})


if __name__ == "__main__":
    main()
