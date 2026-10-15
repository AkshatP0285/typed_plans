"""Run zero-shot JEV classification on post-execution cases."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from jev_client import JevClient, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "post_exec_cases.jsonl"
OUT_DIR = ROOT / "results"

LABELS = {
    "OKAY": (
        "Execution completed successfully AND the returned output satisfies "
        "the expected output schema/requirements provided in the case."
    ),
    "ERROR": (
        "The tool call failed or the tool output explicitly reports an execution "
        "failure, exception, timeout, or unsuccessful operation. Use this for "
        "execution failure, not merely a successful result with the wrong format."
    ),
    "NEEDS_FORMAT_CHANGE": (
        "Execution appears to have completed, but the returned output is missing "
        "required fields, has incompatible types, or violates the expected output "
        "format/schema. Use this when the main issue is output representation."
    ),
}

INSTRUCTIONS = (
    "Classify this POST-EXECUTION tool outcome. Use the user request, tool call, "
    "execution status/output, and expected output requirements. Return exactly "
    "one label from the supplied criteria. If the tool explicitly failed, choose "
    "ERROR. If execution succeeded but its output violates the expected schema, "
    "choose NEEDS_FORMAT_CHANGE. Choose OKAY only when execution succeeded and "
    "the output meets the requirements. Do not infer success merely because output exists."
)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def make_state(case: dict[str, Any]) -> dict[str, Any]:
    # Keep original ToolACE fields available, and add the minimum runtime evidence.
    return {
        "task": "post_execution_tool_outcome_classification",
        "source": case.get("source"),
        "source_id": case.get("source_id"),
        "user_query": case.get("query", ""),
        "available_tools": case.get("tools", []),
        "generated_ir": case.get("generated_ir", {"steps": []}),
        "node": {
            "id": case.get("node_id"),
            "job": case.get("node_job", ""),
            "inputs": case.get("inputs", {}),
            "execution_status": case.get("execution_status", ""),
            "actual_output": case.get("actual_output"),
            "expected_output_requirements": case.get("expected_output_requirements", ""),
        },
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap for a small pilot run.")
    args = parser.parse_args()

    cases = read_jsonl(Path(args.data))
    if args.limit is not None:
        cases = cases[:args.limit]

    client = JevClient(model=args.model)
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []

    for i, case in enumerate(cases, start=1):
        state = make_state(case)
        try:
            result = client.classify(
                state=state,
                criteria=LABELS,
                instructions=INSTRUCTIONS,
                experiment_id=f"zero_shot-{case['case_id']}",
            )
            row = {
                "experiment": "zero_shot",
                "id": case["case_id"],
                "ground_truth": case["ground_truth"],
                "prediction": result.label,
                "probabilities": json.dumps(result.probabilities, ensure_ascii=False),
                "confidence": result.confidence,
                "model": result.model,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "latency_seconds": result.latency_seconds,
                "state_chars": len(json.dumps(state, ensure_ascii=False, separators=(",", ":"))),
                "error": "",
                "raw_response": json.dumps(result.raw_response, ensure_ascii=False),
                "synthetic_execution": case.get("synthetic_execution", True),
            }
            print(f"[zero_shot] {i}/{len(cases)} {case['case_id']}: "
                  f"true={case['ground_truth']} pred={result.label}")
        except Exception as exc:
            row = {
                "experiment": "zero_shot",
                "id": case["case_id"],
                "ground_truth": case["ground_truth"],
                "prediction": "",
                "probabilities": "",
                "confidence": None,
                "model": args.model,
                "input_tokens": None,
                "output_tokens": None,
                "latency_seconds": None,
                "state_chars": len(json.dumps(state, ensure_ascii=False, separators=(",", ":"))),
                "error": repr(exc),
                "raw_response": "",
                "synthetic_execution": case.get("synthetic_execution", True),
            }
            print(f"[zero_shot] {case['case_id']} FAILED: {exc}")
        all_rows.append(row)
        time.sleep(0.15)

    out_path = OUT_DIR / "predictions.csv"
    pd.DataFrame(all_rows).to_csv(out_path, index=False, encoding="utf-8")
    print(f"\\nSaved results to: {out_path}")
    print("Next run: python evaluate_results.py")

if __name__ == "__main__":
    main()
