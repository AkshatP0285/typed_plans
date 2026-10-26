"""Runs one experiment: iterate cases, call the backend, append one JSON row per case.
Resumable: cases with a successful row in the output file are skipped; failures are retried."""
from __future__ import annotations

import dataclasses
import json
import threading
from concurrent.futures import ThreadPoolExecutor

from .backends import BACKENDS
from .config import Experiment
from .tasks import TASKS


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def rep_path(out, rep: int):
    """Repeat 1 is the normal results file; repeat K>1 is <stem>.repK.jsonl next to it."""
    return out if rep <= 1 else out.with_name(f"{out.stem}.rep{rep}.jsonl")


def rep_experiment(exp: Experiment, rep: int) -> Experiment:
    return exp if rep <= 1 else dataclasses.replace(exp, out=rep_path(exp.out, rep))


def build_state_fn(exp: Experiment, task):
    """Plain task state, or (state.wrap / state.fewshot) a wrapper that carries labelled examples.
    Demonstrations come from a JSON file; keys starting with `_` are bookkeeping and are stripped."""
    opts = exp.state_opts
    if not (opts.get("wrap") or opts.get("fewshot")):
        return task.make_state
    examples = []
    if opts.get("fewshot"):
        raw = json.load(open(exp.data.parents[1] / opts["fewshot"], encoding="utf-8"))
        examples = [{k: v for k, v in e.items() if not k.startswith("_")} for e in raw]

    def make(case):
        base = task.make_state(case)
        return {"task": base["task"], "labeled_examples": examples,
                "case_to_classify": {k: v for k, v in base.items() if k != "task"}}
    return make


def run_experiment(exp: Experiment, limit: int = 0, dry_run: bool = False,
                   only_ids: set[str] | None = None) -> None:
    task = TASKS[exp.task]
    cases = read_jsonl(exp.data)
    if only_ids is not None:
        cases = [c for c in cases if c["case_id"] in only_ids]
    if limit:
        cases = cases[:limit]
    done = set()
    if exp.out.exists():
        done = {r["case_id"] for r in read_jsonl(exp.out) if not r.get("error")}
    todo = [c for c in cases if c["case_id"] not in done]
    make_state = build_state_fn(exp, task)
    max_chars = exp.state_opts.get("max_chars")
    if max_chars:  # e.g. JEV's 8,000-char state limit: skip (not error) cases that no longer fit
        fit = [c for c in todo if len(json.dumps(make_state(c), ensure_ascii=False,
                                                 separators=(",", ":"))) <= max_chars]
        if len(fit) < len(todo):
            print(f"  skipping {len(todo) - len(fit)} cases whose state exceeds {max_chars} chars")
        todo = fit
    print(f"[{exp.name}] backend={exp.backend} prompt={exp.prompt.name} params={exp.params}\n"
          f"  {len(done)} already scored, {len(todo)} to run -> {exp.out}")
    if dry_run or not todo:
        if dry_run and todo:
            print("  sample state:", json.dumps(make_state(todo[0]), ensure_ascii=False)[:600], "...")
        return

    backend = BACKENDS[exp.backend](**exp.params)
    backend.setup()
    exp.out.parent.mkdir(parents=True, exist_ok=True)
    lock, n = threading.Lock(), [0]

    def one(case):
        row = {"experiment": exp.name, "case_id": case["case_id"], "label": case[task.truth_field],
               "prompt": exp.prompt.name, "det_flagged": bool(case.get("det_issues"))}
        row.update({k: case.get(k) for k in task.meta_fields})
        try:
            p = backend.classify(make_state(case), exp.prompt.labels, exp.prompt.instructions,
                                 f"{exp.name}-{case['case_id']}")
            row.update(prediction=p.label, probabilities=p.probabilities, confidence=p.confidence,
                       input_tokens=p.input_tokens, latency_seconds=p.latency_seconds, error="",
                       cost_usd=p.cost_usd)
        except Exception as exc:  # keep going; failed cases are retried on the next run
            row.update(prediction="", probabilities={}, confidence=None, input_tokens=None,
                       latency_seconds=None, error=repr(exc))
        with lock:
            n[0] += 1
            with exp.out.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"  [{n[0]}/{len(todo)}] {case['case_id']}: true={row['label']} "
                  f"pred={row['prediction'] or 'ERR ' + row['error'][:70]}")

    if backend.max_workers > 1:
        with ThreadPoolExecutor(backend.max_workers) as ex:
            list(ex.map(one, todo))
    else:
        for c in todo:
            one(c)
