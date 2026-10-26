"""Tasks define how a test case becomes the `state` sent to a decision model and which
field holds the ground truth. Add a task here and select it with `task = "..."` in the config."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Task:
    name: str
    make_state: Callable[[dict], dict]
    truth_field: str              # case field holding the ground-truth label
    positive: str                 # label treated as the "problem found" class in reports
    meta_fields: tuple[str, ...]  # case fields copied into every result row


TASKS: dict[str, Task] = {}


def _pre_exec_state(case: dict) -> dict:
    return {
        "task": "pre_execution_plan_verification",
        "user_query": case["query"],
        "available_tools": case["tools"],
        "plan": case["plan"],
    }


TASKS["pre_exec_plan"] = Task(
    name="pre_exec_plan", make_state=_pre_exec_state, truth_field="label", positive="INVALID",
    meta_fields=("base_id", "kind", "source", "n_steps"),
)
