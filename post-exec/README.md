# JEV Runtime Outcome Classifier — Pilot

This prototype evaluates **post-execution runtime classification only**. It does not implement pre-execution IR verification, repair, caching, or agent integration.

## Labels

- `OKAY`: execution succeeded and output meets the expected requirements.
- `ERROR`: execution failed / exception / timeout / unsuccessful operation.
- `NEEDS_FORMAT_CHANGE`: execution appears successful but output violates the required format/schema.

These definitions are initial annotation rules. Align them with the group's rubric before evaluating real data.

## API

- Endpoint: `https://jevmodel.org/v1/systemone`
- Pinned model: `jev-1.13.0`
- Input: `state` object plus a `choice` question with `criteria`.
- Output: `answers.outcome.choice`, `probabilities`, `confidence`, `model`, and `usage`.

The API documentation limits serialized `state` to 8,000 characters. This client checks that limit before sending. Actual upstream context constraints may be smaller.

## Setup (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:JEVMODEL_API_KEY = Read-Host "Enter JEV API key"
```

If your current terminal already has `TYPESAFE_API_KEY`, the client accepts that as a fallback.

## Run

Run a small pilot first (3 examples per mode, 6 API calls):

```powershell
python run_experiments.py --limit 3
```

Run all dummy examples in both modes (24 API calls):

```powershell
python run_experiments.py
python evaluate_results.py
```

Only zero-shot:

```powershell
python run_experiments.py --modes zero_shot
```

Results are written to `results/predictions.csv`. This output folder is gitignored because results can include data and API usage details. Save research artifacts intentionally and avoid committing private/user data.

## Experiment design

- `zero_shot`: sends the case plus fixed label criteria/instructions.
- `few_shot`: sends the same case plus three labelled demonstrations inside `state`.
- Both modes use the same held-out dummy cases and pinned model.
- Demonstration examples are separate from evaluation cases.
- The few-shot variant is an **example-enriched state experiment**, not a special JEV API few-shot parameter. The documented API exposes `state`, `questions`, and typed question fields.
- Compare accuracy, per-class precision/recall/F1, confusion matrix, latency, input tokens, and state character count.
- Synthetic data is only a smoke test. Use your team's labelled data for meaningful conclusions. Split demonstrations from evaluation data to avoid leakage.

## Important limitations

1. This is a prototype for post-execution classification only.
2. The dummy cases are hand-authored and may not reflect the distribution or ambiguity of real execution outcomes.
3. JEV output is a model judgment, not a safety guarantee. Do not let this prototype autonomously execute or block real actions.
4. Before production integration, add schema validation for the real IR/tool outputs, structured logging, policy for low confidence, and tests against real API responses.
5. Keep API keys out of source control.
