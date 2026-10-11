"""Collect ``metrics.json`` from (multi)run directories into one table."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# (dotted path into metrics.json, column name in the table)
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


def _get(data: dict[str, Any], dotted: str) -> Any:
    """``_get(m, "gate_node.f1") == m["gate_node"]["f1"]``, or ``None`` if any key is missing."""
    value: Any = data
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def _read_overrides(run_dir: Path) -> str:
    """The Hydra command-line overrides of a run (e.g. ``gate=jev seed=1``), or "" if none were saved."""
    overrides_path = run_dir / ".hydra" / "overrides.yaml"
    if not overrides_path.exists():
        return ""
    import yaml

    return " ".join(yaml.safe_load(overrides_path.read_text()) or [])


def collect(root: str | Path) -> list[dict[str, Any]]:
    """One row per ``metrics.json`` found under ``root``: run dir, overrides, then the :data:`COLUMNS`."""
    rows = []
    for metrics_path in sorted(Path(root).rglob("metrics.json")):
        run_dir = metrics_path.parent
        overrides = _read_overrides(run_dir)
        metrics = json.loads(metrics_path.read_text())
        run_name = str(run_dir.relative_to(root)) if run_dir != Path(root) else "."
        row = {"run": run_name, "overrides": overrides}
        for dotted_key, column in COLUMNS:
            row[column] = _get(metrics, dotted_key)
        rows.append(row)
    return rows


def _format_cell(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.3f}"
    return "" if value is None else str(value)


def to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "(no runs found)"
    columns = list(rows[0])
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for row in rows:
        lines.append("| " + " | ".join(_format_cell(row[column]) for column in columns) + " |")
    return "\n".join(lines)
