"""Run zero-shot and example-enriched (few-shot-style) runtime classification."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from jev_client import JevClient, DEFAULT_MODEL

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dummy.jsonl"
DEMO_PATH = ROOT / "data" / "few_shot_examples.json"
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

def make_state(case: dict[str, Any], mode: str, demonstrations: list[dict[str, Any]]) -> dict[str, Any]:
    state = {
        "task": "post_execution_tool_outcome_classification",
        "case": {
            "id": case["id"],
            "user_query": case["user_query"],
            "tool_call": case["tool_call"],
            "execution_status": case["execution_status"],
            "tool_output": case["tool_output"],
            "expected_output_requirements": case["expected_output_requirements"],
        },
    }
    if mode == "few_shot":
        # Demonstrations are kept compact to respect the API's state/context limit.
        state["labelled_examples"] = demonstrations
    return state

def compact_demo(d: dict[str, Any]) -> dict[str, Any]:
    return {
        "case": {
            "tool_call": d["tool_call"],
            "execution_status": d["execution_status"],
            "tool_output": d["tool_output"],
            "expected_output_requirements": d["expected_output_requirements"],
        },
        "correct_label": d["ground_truth"],
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA_PATH))
    parser.add_argument("--demos", default=str(DEMO_PATH))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--modes", nargs="+", choices=["zero_shot", "few_shot"],
                        default=["zero_shot", "few_shot"])
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional cap for a small pilot run.")
    args = parser.parse_args()

    cases = read_jsonl(Path(args.data))
    if args.limit:
        cases = cases[:args.limit]
    demos_raw = json.loads(Path(args.demos).read_text(encoding="utf-8"))
    demonstrations = [compact_demo(d) for d in demos_raw]

    client = JevClient(model=args.model)
    OUT_DIR.mkdir(exist_ok=True)
    all_rows = []

    for mode in args.modes:
        for i, case in enumerate(cases, start=1):
            state = make_state(case, mode, demonstrations)
            try:
                result = client.classify(
                    state=state,
                    criteria=LABELS,
                    instructions=INSTRUCTIONS,
                    experiment_id=f"{mode}-{case['id']}",
                )
                row = {
                    "experiment": mode,
                    "id": case["id"],
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
                }
                print(f"[{mode}] {i}/{len(cases)} {case['id']}: "
                      f"true={case['ground_truth']} pred={result.label}")
            except Exception as exc:
                row = {
                    "experiment": mode,
                    "id": case["id"],
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
                }
                print(f"[{mode}] {case['id']} FAILED: {exc}")
            all_rows.append(row)
            # Stay comfortably below the documented 120 requests/minute limit.
            time.sleep(0.15)

    df = pd.DataFrame(all_rows)
    out_path = OUT_DIR / "predictions.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"\nSaved all results to: {out_path}")
    print("Next run: python evaluate_results.py")

if __name__ == "__main__":
    main()
