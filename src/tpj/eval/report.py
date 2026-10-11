"""Collect ``metrics.json`` from (multi)run directories into one table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

COLUMNS = [
    ("success_rate", "SR"),
    ("avg_tokens.total", "tokens"),
    ("avg_api_calls.judge", "judge_calls"),
    ("avg_api_calls.gate", "gate_calls"),
    ("gate_node.f1", "gate_node_f1"),
    ("gate_plan.f1", "gate_plan_f1"),
    ("unsafe_action_detection_rate", "UADR"),
    ("false_intervention_rate", "FIR"),
    ("cache_reuse_ratio", "cache_reuse"),
    ("avg_re_executions", "re_exec"),
    ("repair_success_rate", "repair_SR"),
]


def _get(d: dict[str, Any], dotted: str) -> Any:
    for part in dotted.split("."):
        if not isinstance(d, dict) or part not in d:
            return None
        d = d[part]
    return d


def collect(root: str | Path) -> list[dict[str, Any]]:
    rows = []
    for mpath in sorted(Path(root).rglob("metrics.json")):
        run_dir = mpath.parent
        overrides_path = run_dir / ".hydra" / "overrides.yaml"
        overrides = ""
        if overrides_path.exists():
            import yaml

            overrides = " ".join(yaml.safe_load(overrides_path.read_text()) or [])
        metrics = json.loads(mpath.read_text())
        row = {"run": str(run_dir.relative_to(root)) if run_dir != Path(root) else ".", "overrides": overrides}
        for key, name in COLUMNS:
            row[name] = _get(metrics, key)
        rows.append(row)
    return rows


def to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no runs found)"
    cols = list(rows[0])
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else ("" if v is None else str(v))  # noqa: E731
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(lines)
