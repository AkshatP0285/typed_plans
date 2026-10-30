"""Metrics from trace events (proposal Section 11).

Per task (:func:`task_metrics`): tokens and API calls per role, gate decisions
vs injected faults, interventions, cache reuse. Aggregated over tasks
(:func:`aggregate`): success rate, averages, gate P/R/F1, detection rate and
false intervention rate.

Gate quality is measured against injected faults: a "positive" is a faulty
plan/output, and the gate "predicts positive" when it flags it.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from tpj.faults.injector import RAISING

Confusion = dict[str, int]  # {"tp": .., "fp": .., "fn": .., "tn": ..}


def _empty_confusion() -> Confusion:
    return {"tp": 0, "fp": 0, "fn": 0, "tn": 0}


def _confusion_cell(faulty: bool, flagged: bool) -> str:
    """Which confusion-matrix cell a gate decision falls into."""
    if flagged:
        return "tp" if faulty else "fp"
    return "fn" if faulty else "tn"


def _events(events: list[dict[str, Any]], type_: str) -> list[dict[str, Any]]:
    return [e for e in events if e["type"] == type_]


# ---------------------------------------------------------------- per task
def task_metrics(events: list[dict[str, Any]], score: dict[str, float]) -> dict[str, Any]:
    """Summarise one task's trace events into a flat metrics row."""
    tokens, api_calls, latency = _cost_by_role(events)
    node_results = _events(events, "node_result")
    interventions = _events(events, "intervention")
    return {
        "score": score,
        "tokens": tokens,
        "api_calls": api_calls,
        "latency": latency,
        "task_latency": _task_latency(events),
        "plan": _plan_gate_row(events),
        "node_gate": _node_gate_confusion(node_results),
        "tool_executions": len(node_results),
        "re_executions": sum(1 for e in node_results if e.get("attempt", 1) > 1),
        "cache_hits": len(_events(events, "cache_hit")),
        "interventions": len(interventions),
        "false_alarm_verdicts": sum(1 for e in interventions if e["verdict"] == "false_alarm"),
        "exec_faults": sum(1 for e in _events(events, "fault") if e["level"] == "exec"),
        "crashed": bool(_events(events, "crash")),
    }


def _cost_by_role(events: list[dict[str, Any]]) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
    """Tokens, API calls and total latency-seconds per role (planner, gate, judge, ...)."""
    tokens: dict[str, int] = defaultdict(int)
    calls: dict[str, int] = defaultdict(int)
    latency: dict[str, float] = defaultdict(float)
    for e in _events(events, "llm_call"):
        tokens[e["role"]] += e["prompt_tokens"] + e["completion_tokens"]
        latency[e["role"]] += e.get("latency") or 0.0
        if e["role"] != "gate":  # gate calls are counted from "gate" events below
            calls[e["role"]] += 1
    for e in _events(events, "gate_api"):  # JEV reports its own token usage
        tokens["gate"] += (e.get("input_tokens") or 0) + (e.get("output_tokens") or 0)
        latency["gate"] += e.get("latency") or 0.0
    # Every gate decision counts as one call, whichever gate (local model, API, LLM) made it.
    gate_decisions = len(_events(events, "gate"))
    if gate_decisions:
        calls["gate"] += gate_decisions
    return dict(tokens), dict(calls), dict(latency)


def _task_latency(events: list[dict[str, Any]]) -> float:
    """Wall-clock time from the first to the last event - how long the whole task took."""
    times = [e["t"] for e in events]
    return max(times) - min(times) if times else 0.0


def _plan_gate_row(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Was an injected plan fault flagged before execution? Only the first plan check counts."""
    plan_faults = [e for e in _events(events, "fault") if e["level"] == "plan"]
    plan_gates = [e for e in _events(events, "gate") if e["phase"] == "plan"]
    first_plan_gate = plan_gates[0] if plan_gates else None
    return {
        "faulty": bool(plan_faults),
        "flagged": bool(first_plan_gate and first_plan_gate["flagged"]),
        "gated": first_plan_gate is not None,
        "edits": len(_events(events, "plan_edit")),
    }


def _node_gate_confusion(node_results: list[dict[str, Any]]) -> Confusion:
    """Gate decisions on executed steps that produced an output.

    Raising faults (exception, timeout) never reach the gate, so they are not
    counted as faulty here.
    """
    confusion = _empty_confusion()
    for e in node_results:
        if not e.get("gated"):
            continue
        faulty = e.get("fault") is not None and e["fault"] not in RAISING
        confusion[_confusion_cell(faulty, bool(e.get("flagged")))] += 1
    return confusion


# ---------------------------------------------------------------- aggregate
def _prf(c: Confusion) -> dict[str, float]:
    """Precision, recall, F1 and accuracy from a confusion matrix."""
    tp, fp, fn, tn = c["tp"], c["fp"], c["fn"], c["tn"]
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + fn + tn
    return {"precision": precision, "recall": recall, "f1": f1,
            "accuracy": (tp + tn) / total if total else 0.0, "support": total}


def _mean_per_key(dicts: list[dict[str, float]], keys: list[str], n: int) -> dict[str, float]:
    """Average of each key over all tasks, counting a missing key as 0."""
    return {k: sum(d.get(k, 0) for d in dicts) / n for k in keys}


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Combine the per-task rows from :func:`task_metrics` into one metrics dict."""
    n = len(rows) or 1  # avoid dividing by zero on an empty run
    out: dict[str, Any] = {"n_tasks": len(rows)}

    # --- task success and cost
    score_keys = sorted({k for r in rows for k in r["score"]})
    out["scores"] = _mean_per_key([r["score"] for r in rows], score_keys, n)
    out["success_rate"] = out["scores"].get("success", 0.0)

    roles = sorted({k for r in rows for k in r["tokens"]} | {k for r in rows for k in r["api_calls"]})
    out["avg_tokens"] = _mean_per_key([r["tokens"] for r in rows], roles, n)
    out["avg_tokens"]["total"] = sum(out["avg_tokens"].values())
    out["avg_api_calls"] = _mean_per_key([r["api_calls"] for r in rows], roles, n)

    # --- gate quality against injected faults
    node = _empty_confusion()
    for r in rows:
        for cell in node:
            node[cell] += r["node_gate"][cell]
    plan = _empty_confusion()
    for r in rows:
        if r["plan"]["gated"]:
            plan[_confusion_cell(r["plan"]["faulty"], r["plan"]["flagged"])] += 1
    out["gate_node"] = _prf(node)
    out["gate_plan"] = _prf(plan)

    injected_plan_faults = sum(1 for r in rows if r["plan"]["faulty"])
    out["unsafe_action_detection_rate"] = plan["tp"] / injected_plan_faults if injected_plan_faults else None
    # Of everything the gate saw that was actually fine, how much did it flag anyway?
    valid_seen = node["fp"] + node["tn"] + plan["fp"] + plan["tn"]
    out["false_intervention_rate"] = (node["fp"] + plan["fp"]) / valid_seen if valid_seen else None

    # --- execution efficiency
    cache_hits = sum(r["cache_hits"] for r in rows)
    executions = sum(r["tool_executions"] for r in rows)
    out["avg_tool_executions"] = executions / n
    out["avg_re_executions"] = sum(r["re_executions"] for r in rows) / n
    out["cache_reuse_ratio"] = cache_hits / (cache_hits + executions) if cache_hits + executions else 0.0
    out["avg_interventions"] = sum(r["interventions"] for r in rows) / n

    # --- repair: success rate among tasks where the judge intervened at least once
    repaired = [r for r in rows if r["interventions"]]
    out["repair_success_rate"] = (sum(r["score"].get("success", 0.0) for r in repaired) / len(repaired)
                                  if repaired else None)
    out["crash_rate"] = sum(1 for r in rows if r["crashed"]) / n
    return out
