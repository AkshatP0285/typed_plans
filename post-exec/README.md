# JEV Post-Execution Classifier — Zero-Shot Pilot

## 1. Overview

This pilot evaluates JEV as a **zero-shot post-execution classifier**. For each case, JEV receives one node's task, inputs, execution status, actual output, and expected output requirements. It classifies the execution into one of three categories:

- `OKAY` — the execution output satisfies the expected requirements.
- `ERROR` — the execution failed or produced an execution error.
- `NEEDS_FORMAT_CHANGE` — the execution returned a result, but its format or fields do not satisfy the expected requirements.

The experiment uses no few-shot examples. 

## 2. Dataset and Original Fields

The input file is `data/post_exec_cases.jsonl`. It contains six synthetic pilot cases derived from ToolACE records.

The following fields are retained from the original ToolACE dataset:

| Original field | Purpose |
|---|---|
| `source` | Identifies the data source, such as `toolace`. |
| `source_id` | Identifies the original ToolACE example. |
| `query` | The user's original request. |
| `tools` | The tools available in the original example, including their descriptions and parameter requirements. |
| `route` | The original routing decision. |
| `generated_ir` | The original intermediate representation, including its steps, tool names, arguments, dependencies, and other node properties. |
| `structural_check` | The original structural validation result and any recorded errors. |

The `generation_meta` field is retained, but its contents have been modified to document the synthetic execution assumptions. Therefore, it should not be treated as unchanged original metadata.

## 3. Changes Made to the Data

The original ToolACE records primarily describe user requests, available tools, and generated intermediate representations. They do not provide the actual post-execution outputs needed for this experiment.

We therefore added the following fields:

| Added field | Purpose |
|---|---|
| `case_id` | A unique identifier for each experimental case. |
| `node_id` | Identifies the node evaluated by JEV. |
| `node_job` | A concise description of the selected node's task. |
| `inputs` | The inputs supplied to the node, extracted from the original arguments where possible or explicitly constructed for synthetic fixtures. |
| `execution_status` | A simulated execution status, such as `success` or `failed`. |
| `actual_output` | A simulated result or error returned by the hypothetical execution. |
| `expected_output_requirements` | A description of the expected result, including required fields, types, or value constraints. |
| `ground_truth` | The manually assigned reference label used to evaluate JEV's prediction. |
| `synthetic_execution` | Marks the case as synthetic rather than a record of a real tool execution. |

The original fields are not all sufficient to reconstruct a real execution. In particular, `node_job`, `inputs`, expected output requirements, execution outcomes, and reference labels may require manual construction. These additions are documented as experimental fixtures, not as original ToolACE annotations.

## 4. Pilot Tests

The six cases are intended to exercise different classification outcomes:

1. **Successful-looking output (`OKAY`)** — a simulated prediction containing a student ID and a numeric dropout probability between 0 and 1.
2. **Execution failure (`ERROR`)** — a simulated timeout from the prediction service.
3. **Incorrect output type (`NEEDS_FORMAT_CHANGE`)** — the dropout probability is returned as the string `"low"` rather than a numeric probability.
4. **Missing expected field (`NEEDS_FORMAT_CHANGE`)** — the output contains a confidence value but omits the required `dropout_probability` field.
5. **Unavailable tool (`ERROR`)** — a synthetic screenshot operation fails because the original tool list does not contain a screenshot tool.
6. **Additional synthetic classification case** — a hypothetical node and outcome are added to an original record that does not contain an executable IR step.

The first four cases are derived from the same original ToolACE example (`toolace_6681`), while the screenshot case is based on `toolace_1758`. The sixth case is based on another original record with no executable steps.

**Important limitation:** These cases test the pipeline and provide illustrative classification scenarios; they do not constitute six independent real tool executions. The first four cases reuse the same original request and IR, and the last two require synthetic nodes or outcomes.

The original `toolace_6681` IR supplies `student_id` but omits the required `course_id`. Consequently, its simulated successful output must not be interpreted as evidence that the original tool call was executable as written. This mismatch should be acknowledged when interpreting the `OKAY` case.

## 5. Running the Experiment

From the project directory, set the JEV API key in PowerShell:

```powershell
$env:JEVMODEL_API_KEY="your_actual_api_key"
```

Run the zero-shot predictions:

```powershell
python run_experiments.py
```

Evaluate the predictions against the manually assigned reference labels:

```powershell
python evaluate_results.py
```

The first command calls JEV and saves its predictions. The second calculates evaluation metrics from those predictions.

## 6. Interpretation and Limitations

All execution outcomes and reference labels in this pilot are synthetic. They are intended to check whether the JEV request, response parsing, prediction saving, and evaluation pipeline work as expected.

Any accuracy, precision, recall, F1 score, or confusion matrix obtained from these six cases describes performance only on this small synthetic pilot. It must not be presented as a measure of JEV's performance on real ToolACE executions.

A proper post-execution evaluation would require genuine tool execution outputs, clearly defined expected output requirements, and independently verified reference labels. The current pilot is a preliminary pipeline test rather than a definitive benchmark.