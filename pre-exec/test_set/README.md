# Pre-execution test set (frozen)

File: `pre_exec_test_set.jsonl` — 466 cases from 100 base plans. sha256: `6e6f7f588341f72ceb6995f02cdfb63fc5f29ad039f95b6950afb29dfa881b70`

Built by `../build_pre_cases.py --seed 7 --n 100` from `merged_ir_thinking_final.jsonl`
(schema-clean multi-step plans only; tool lists trimmed to used tools + up to 3 distractors).
Do not regenerate between experiments — reuse this file so settings are comparable.

## Fields
| field | meaning |
|---|---|
| case_id | unique id: `<base_id>__<kind>` |
| base_id | the original plan this case came from (all variants of one plan share it) |
| source | toolace / apigen_mt / hermes |
| kind | gt, remove, replace_tool, replace_arg, add_unrelated_tool, add_duplicate |
| label | ground truth: VALID (gt) or INVALID (all edits) |
| n_steps | step count of the plan in this case |
| query, tools, plan | the model input (plan = {"steps": [...]}) |
| det_issues | issues found by the rule-based validator (empty = it would pass the plan) |
| edit | what was changed (empty for gt) |

## Counts
- by kind: {'gt': 100, 'remove': 100, 'replace_arg': 76, 'add_duplicate': 42, 'replace_tool': 90, 'add_unrelated_tool': 58}
- base plans by source: {'toolace': 40, 'apigen_mt': 35, 'hermes': 25}

## Notes
- Labels: only `gt` is VALID. The GT plans are Qwen-generated and schema-checked only, so some may have semantic flaws.
- `replace_tool` is the easy tier (the rule-based validator catches ~98%); the other edit kinds are semantic-only.
- Split results by `base_id` when sampling subsets, so a plan and its edits stay together.
- Results and run log: `../results/` (`RUN_LOG.md`).

## Using it with different settings
Add or edit an experiment in `../experiments.toml` (it points at this file by default), then
`python ../run.py run --only <name>` and `python ../run.py report`. Each experiment writes its own
`results/<name>.jsonl` and resumes per file.
