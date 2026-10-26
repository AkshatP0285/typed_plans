"""Build few-shot demonstrations v2: correct / partially wrong / wrong plans.

Fixes the problems seen in v1: noisy "missing step" label, a questionable VALID demo, `think`
steps, placeholder ids, and no wrong-throughout example. Every demonstration is drawn from base
plans that are NOT in the frozen test set (asserted) and satisfies:

  correct          verdict VALID    plan passes the schema validator, no `think` step, no placeholder
                                    ids, every literal argument value appears in the user query
  partially_wrong  verdict INVALID  a correct plan with ONE defect, and the reason says which steps
                                    are still right:
                                      - replace_arg : a query-grounded value replaced by another
                                      - remove      : a step whose own arguments the query spells
                                                      out is deleted (so it is clearly required)
  wrong            verdict INVALID  nothing useful is left:
                                      - unrelated   : the plan answers a different request
                                      - all_tools   : every step calls the wrong tool

Output: test_set/fewshot_examples_v2.json
"""
from __future__ import annotations

import collections
import copy
import json
import random
from pathlib import Path

import build_pre_cases as b

ROOT = Path(__file__).resolve().parent
TEST = ROOT / "test_set" / "pre_exec_test_set.jsonl"
OUT = ROOT / "test_set" / "fewshot_examples_v2.json"
SEED = 21
MAX_Q, MAX_PLAN = 220, 520          # chars: demonstrations must stay short (8,000-char state limit)
PLACEHOLDERS = ("current", "user_id_", "example", "placeholder", "unknown", "pending_")


def literals(step):
    return [v for v in step["args"].values()
            if isinstance(v, (str, int, float)) and not isinstance(v, bool) and not b.is_ref(v)]


def grounded(step, query):
    lits = literals(step)
    return bool(lits) and all(str(v).lower() in query.lower() for v in lits)


def clean_correct(r):
    steps, q = r["generated_ir"]["steps"], r["query"]
    if any(s["tool"] == "think" for s in steps):
        return False
    if any(any(p in str(v).lower() for p in PLACEHOLDERS) for s in steps for v in literals(s)):
        return False
    # every literal argument is spelled out in the query (no invented values)
    return all(all(str(v).lower() in q.lower() for v in literals(s)) for s in steps)


def short(r):
    return len(r["query"]) <= MAX_Q and len(json.dumps(r["generated_ir"]["steps"], ensure_ascii=False)) <= MAX_PLAN


def stems(text):
    import re
    words = re.findall(r"[A-Za-z]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", text))
    return {w.lower()[:4] for w in words if len(w) >= 3}


def clearly_unrelated(tool_name, query):
    """The replacement tool's name shares no 4-letter word stem with the query, so calling it is
    unambiguously wrong (guards against e.g. MatchStatistics for a 'stats' request)."""
    return not (stems(tool_name) & stems(query))


def names(steps, ids):
    return ", ".join(f"{s['id']} ({s['tool']})" for s in steps if s["id"] in ids)


def main():
    rng = random.Random(SEED)
    test_ids = {json.loads(l)["base_id"] for l in open(TEST, encoding="utf-8") if l.strip()}
    rows = [r for r in b.load(str(b.DEFAULT_SRC)) if r["source_id"] not in test_ids]
    pool = b.literal_pool(rows)
    good = []
    for r in rows:
        steps = r["generated_ir"]["steps"]
        if not r["tools"] or not r["structural_check"]["passed"] or not (1 <= len(steps) <= 3):
            continue
        r["tools"] = b.trim_tools(steps, r["tools"], rng)
        if b.validate(steps, r["tools"]) or not clean_correct(r) or not short(r):
            continue
        good.append(r)
    multi = [r for r in good if len(r["generated_ir"]["steps"]) >= 2]
    by_src = collections.defaultdict(list)
    for r in good:
        by_src[r["source"]].append(r)
    print(f"clean short base plans: {len(good)} ({len(multi)} multi-step); by source {({k: len(v) for k, v in by_src.items()})}")

    used, examples = set(), []

    def take(cands, pred=lambda r: True):
        for r in rng.sample(cands, len(cands)):
            if r["source_id"] not in used and pred(r):
                used.add(r["source_id"])
                return r
        raise SystemExit("not enough eligible plans for a demonstration slot")

    def add(kind, quality, verdict, query, steps, reason, base):
        examples.append({"_base_id": base, "_kind": kind, "assessment": quality, "user_query": query,
                         "plan": {"steps": steps}, "verdict": verdict, "reason": reason})

    # --- correct (2): one with independent calls, one dependent chain / different source
    r = take(by_src["toolace"])
    add("correct", "correct", "VALID", r["query"], r["generated_ir"]["steps"],
        "all steps are right: tools and argument values match the request and nothing is missing or extra", r["source_id"])
    r = take(by_src["hermes"] + by_src["apigen_mt"], lambda r: r["source_id"] not in used)
    add("correct", "correct", "VALID", r["query"], r["generated_ir"]["steps"],
        "all steps are right: tools and argument values match the request and nothing is missing or extra", r["source_id"])

    # --- partially wrong (2): replace_arg and a clearly required removed step
    for _ in range(50):
        r = take(multi)
        steps, tools = r["generated_ir"]["steps"], r["tools"]
        res = b.m_replace_arg(steps, tools, rng, pool, query=r["query"])
        if res is None or b.validate(res[0], tools):
            used.discard(r["source_id"]); continue
        new, e = res
        ok = [s["id"] for i, s in enumerate(new) if i != e["position"]]
        add("partial_replace_arg", "partially_wrong", "INVALID", r["query"], new,
            f"partially wrong: {names(new, ok)} are right, but {new[e['position']]['id']} "
            f"({e['tool']}) uses {e['arg']}={e['new']!r} while the request says {e['old']!r}", r["source_id"])
        break
    for _ in range(100):
        r = take(multi)
        steps, tools = r["generated_ir"]["steps"], r["tools"]
        ref = set()
        for s in steps:
            ref |= set(s["depends_on"]) | b.refs_in(s["args"])
        cand = [i for i, s in enumerate(steps) if s["id"] not in ref and grounded(s, r["query"])]
        if not cand:
            used.discard(r["source_id"]); continue
        i = rng.choice(cand)
        new = b.renumber(steps[:i] + steps[i + 1:])
        gone = steps[i]
        add("partial_remove", "partially_wrong", "INVALID", r["query"], new,
            f"partially wrong: the remaining steps are right, but a needed step is missing: the request "
            f"asks for {gone['tool']} with {json.dumps(gone['args'], ensure_ascii=False)} and the plan never does it", r["source_id"])
        break

    # --- wrong (2): unrelated plan, and every tool replaced
    a, c = take(good), take(good, lambda r: r["source"] != "x")
    add("wrong_unrelated", "wrong", "INVALID", a["query"], c["generated_ir"]["steps"],
        f"wrong: the plan answers a different request; it calls {', '.join(s['tool'] for s in c['generated_ir']['steps'])}, "
        f"none of which the request asks for", a["source_id"])
    used.add(c["source_id"])
    for _ in range(100):
        r = take([x for x in multi if len(x["tools"]) >= 3])
        steps, tools = r["generated_ir"]["steps"], r["tools"]
        new, ok = copy.deepcopy(steps), True
        for s in new:
            other = [t["name"] for t in tools if t["name"] != s["tool"]
                     and clearly_unrelated(t["name"], r["query"])]
            if not other:
                ok = False; break
            s["tool"] = rng.choice(other)
        if not ok:
            used.discard(r["source_id"]); continue
        add("wrong_all_tools", "wrong", "INVALID", r["query"], b.renumber(new),
            "wrong: every step calls the wrong tool for the arguments it is given, so no part of the plan "
            "does what the request asks", r["source_id"])
        break

    assert not ({e["_base_id"] for e in examples} & test_ids), "leak: a demo plan is in the test set"
    rng.shuffle(examples)
    OUT.write_text(json.dumps(examples, ensure_ascii=False, indent=1), encoding="utf-8")
    chars = len(json.dumps([{k: v for k, v in e.items() if not k.startswith("_")} for e in examples],
                           ensure_ascii=False, separators=(",", ":")))
    print(f"\n{len(examples)} demonstrations, {chars} chars -> {OUT.name}")
    for e in examples:
        print(f"\n[{e['assessment']:15s} {e['verdict']:7s} {e['_kind']}] {e['_base_id']}\n  Q: {e['user_query'][:160]}")
        for s in e["plan"]["steps"]:
            print("    ", s["id"], s["tool"], json.dumps(s["args"], ensure_ascii=False)[:120], s["depends_on"])
        print("  reason:", e["reason"][:230])


if __name__ == "__main__":
    main()
