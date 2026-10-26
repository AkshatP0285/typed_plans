# Pre-execution verification: run log

Task: given a user query, tool schemas and a plan (IR), say VALID or INVALID before anything runs.
Test set: `test_set/pre_exec_test_set.jsonl`, 466 cases from 100 base plans (seed 7): 100 unedited plans
(label VALID) + 366 edited plans (label INVALID: remove 100, replace_arg 76, replace_tool 90,
add_unrelated 58, add_duplicate 42). Positive class = INVALID.

Experiments kept (see `experiments.toml`): `rules_baseline`, `jev_or_zeroshot`, `jev_or_fewshot4_v1`,
`jev_or_fewshot_v2`, `laya_english_2048`, `laya_typed_2048` (partial). Scores below are counts:
edits caught (of 366), false alarms = valid plans wrongly said INVALID (of 100), total correct (of 466)
= edited said INVALID + valid said VALID.

## Data notes
- Source IRs (`merged_ir_thinking_final.jsonl`, Qwen-generated "GT"): only about 660 of ~4,460 multi-step plans with tools pass a basic
  schema check (the rest have wrong argument names or missing required arguments); the dataset's own `structural_check` does not catch this.
  Base plans were sampled only from the schema-clean ones. They are NOT checked semantically, so some "valid" plans are arguably flawed.
- Edits: step ids are renumbered after an edit (no gap), tool lists are trimmed to the used tools + up to 3 distractors, `replace_arg` only touches
  values the query states, `think` is never added. The rule-based validator catches 88 of the 90 `replace_tool` edits and none of the others.
- No dependency-only edits (re-wiring `$sN` references) are in the test set, so results say nothing about dependency reasoning specifically.

## Rules baseline (no model)
Deterministic schema/reference validator (unknown tool, missing/unknown args, dangling refs). Caught 88/366, 0/100 false alarms, 188/466 correct (40.3%).

## Laya (local, CPU)
Laya English, `convaiinnovations/laya`, laya 0.4.8, max_len=2048, head_max_len=192, same prompt and labels as JEV, 8.4 s/case.
- All 466 scored, 0 errors. AT CHANCE: caught 5/366, 3/100 false alarms, 102/466 correct (21.9%), AUROC 0.502.
  Predicts VALID for 458/466. Mean P(INVALID) is 0.306 on valid plans vs 0.307 on edited plans.
- Laya printed a warning that one calibration temperature of the checkpoint is invalid and was clamped, so its probabilities are uncalibrated.
- The model card says the base checkpoints are near chance zero-shot (0.362 against a 0.461 majority baseline on its own benchmark) and that accuracy comes from fine-tuning.
- Token measurements (Laya tokenizer, 466 cases), mean / max tokens: query 116 / 578, tool schemas 497 / 1,553, plan 177 / 503, wrapper keys 30 / 50, state total 820 / 1,941.
  Default English budget (512, ~339 left for the state) would truncate 438/466 plans; at max_len=2048 only 4/466 lose about 32 tokens.
  Each option description is hard-capped at 48 tokens (ours are 50-51: last 2-3 tokens cut).
Laya typed-decisions (same settings, head_max_len 256 = checkpoint default): PARTIAL, 108 of 466 cases scored (the run was stopped on request), 8.8 s/case.
  On those 108 (85 edited, 23 valid): caught 2/85, 1/23 false alarms, AUROC 0.514; mean P(INVALID) 0.353 valid vs 0.356 edited. Also at chance.

## JEV via OpenRouter (all 466 cases, model typesafe/jev-1.13)
Backend `jev_openrouter`: OpenRouter Decisions API, POST https://openrouter.ai/api/alpha/decisions, key from OPENROUTER_API_KEY (an `sk-or-v1-...` key; never stored).
State limit ~32,000 bytes, so nothing was skipped for size. 0.57 s/case. Cost about $0.03 (zero-shot), $0.046 (v1), $0.053 (v2) per full run.

Few-shot demonstrations are put in the state as `labeled_examples` with the test case under `case_to_classify`; all demonstration plans come from base plans NOT in the test set (asserted).
- v1 (`test_set/fewshot_examples.json`, build_fewshot.py): 4 examples, 2 VALID + 2 INVALID (one wrong argument, one removed step); reasons on the INVALID ones only.
- v2 (`test_set/fewshot_examples_v2.json`, build_fewshot_v2.py): 6 examples: 2 correct (VALID), 2 partially wrong (one wrong argument; one clearly required step missing; reasons say which steps are still right),
  2 wrong (a plan for an unrelated request; every step calls an unrelated tool). Stricter selection: no `think` steps, no placeholder ids, every literal argument appears in the query,
  wrong-tool demos use tools whose name shares no word stem with the query. The prompt explains the correct / partially wrong / wrong assessments.

Single run (rep 1):
| experiment | edits caught /366 | false alarms /100 | total correct /466 | precision | F1 | AUROC |
|---|---|---|---|---|---|---|
| rules_baseline | 88 | 0 | 188 (40.3%) | 1.000 | 0.388 | 0.620 |
| laya_english_2048 | 5 | 3 | 102 (21.9%) | 0.625 | 0.027 | 0.502 |
| jev_or_zeroshot | 359 | 32 | 427 (91.6%) | 0.918 | 0.948 | 0.971 |
| jev_or_fewshot_v2 | 357 | 25 | 432 (92.7%) | 0.935 | 0.955 | 0.980 |
| jev_or_fewshot4_v1 | 360 | 23 | 437 (93.8%) | 0.940 | 0.961 | 0.979 |

Repeatability: a determinism check (40 cases x 3 identical calls) gave identical P(INVALID) for 28/40, max spread 0.06, no verdict flips, so JEV is nearly deterministic.
Three repeats of the full set (`--rep 2`, `--rep 3`), mean +- sample std:
| experiment | edits caught /366 | false alarms /100 | total correct /466 | F1 | AUROC | cases whose verdict changed |
|---|---|---|---|---|---|---|
| jev_or_zeroshot | 358.3 +- 0.6 | 31.3 +- 0.6 | 427.0 +- 0.0 | 0.948 +- 0.000 | 0.971 +- 0.001 | 4 |
| jev_or_fewshot4_v1 | 360.3 +- 0.6 | 23.3 +- 1.5 | 437.0 +- 1.0 | 0.961 +- 0.001 | 0.979 +- 0.000 | 10 |
| jev_or_fewshot_v2 | 357.7 +- 0.6 | 23.7 +- 1.2 | 434.0 +- 1.7 | 0.957 +- 0.002 | 0.980 +- 0.000 | 8 |
Per-repeat (caught | false alarms | correct): zero-shot 359|32|427, 358|31|427, 358|31|427; v1 360|23|437, 361|25|436, 360|22|438; v2 357|25|432, 358|23|435, 358|23|435.

95% cluster bootstrap over the 100 base plans (2000 resamples, repeat 1), difference vs zero-shot:
- v1: false alarms -0.088 [-0.170, -0.010] (excludes 0); edits caught +0.002 [-0.011, 0.016]
- v2: false alarms -0.069 [-0.140, 0.000] (touches 0);   edits caught -0.006 [-0.019, 0.008]
Paired exact McNemar (two-sided) on false alarms: zero vs v1 p=0.049, zero vs v2 p=0.092, v1 vs v2 p=0.688; on edits caught all p>=0.375. Three comparisons were made, so p=0.049 is borderline.

Reading:
- Few-shot lowers false alarms by about 7-9 valid plans out of 100 with no change in edits caught; the drop is much larger than run-to-run noise (std 0.6-1.5). The richer v2 set did not beat v1.
- Repeats measure API nondeterminism only; they do not capture how the result depends on which demonstrations were chosen.
- 18 valid plans are flagged in all three settings (11 apigen_mt, 6 toolace, 1 hermes) and few-shot does not move them. A manual read of four of them (apigen_mt_1058, apigen_mt_3290, apigen_mt_871, hermes_282)
  found at least three defensible flags (invented user ids, guessed output fields, a cancel step with no confirmation), so part of the "false alarm" rate is label noise in the GT plans. A manual audit of the valid plans is still open.
- Weakest edit type is `remove` (93/100 caught zero-shot, 92/100 v2); every other edit type is at or near 100%.
- Probabilities are saturated: about 58% of P(INVALID) values are exactly 0 or 1, so AUROC behaves close to a binary metric.

## Check: is the rules baseline's 0 false alarms evidence that all IR errors are rule-based?
- No. 0/100 false alarms is guaranteed by construction: build_pre_cases.py drops every base plan the rule validator flags, so all 100 GT plans pass it.
- In the RAW Qwen-generated IRs (9,174 plans with >=1 step and tools) only 2,591 (28.2%) pass the rule validator; 6,583 (71.8%) fail it
  (missing_required in 6,091, unknown_arg in 5,887, unknown_tool in 9). So in the raw data the most common IR error IS rule-detectable (schema violations); the test set excludes them on purpose.
- Edits: the rules flag 88/90 replace_tool edits and 0/276 of the others (remove 0/100, replace_arg 0/76, add 0/100) by design; JEV catches 98% overall, i.e. it detects edits no schema rule can see.
- Among the 100 schema-clean GT plans, crude non-schema heuristics (placeholder-looking values, `think` steps, string arguments not stated in the query, `$sN.field` references to guessed output fields) mark 53/100 as candidate flaws.
  JEV zero-shot flags 26/53 (49%) of those vs 6/47 (13%) of the rest (Fisher exact p=0.0001). Of the 6 JEV-flagged plans with no heuristic hit, a manual read judged about 3 genuine false alarms
  (toolace_2409, toolace_8140, toolace_7378), the others had subtle issues the heuristics skip (nested guessed `$s1.id` ref in toolace_565, invented coordinates in toolace_1408, conditional step in toolace_696).
- Caveats: heuristics are crude (a `think` step or a value not repeated in the query is not necessarily wrong, nested arguments were not checked) and this is not a full manual audit.
- Implication: results on this test set measure what rules cannot catch; on raw planner output rules alone would already flag ~72% of plans. Not tested: JEV on the rule-flagged raw plans.

## Meaning-only view (edits a rule checker cannot catch) — no new model calls
- `python run.py report --meaning-only` drops the 88 rule-detectable edits (the tool swaps the validator flags): 378 cases = 278 meaning-type edits + 100 valid plans. Saved to results/comparison_meaning_only.csv.
- Counts (edits caught /278 | false alarms /100 | total correct /378):
  rules_baseline 0 | 0 | 100 (26.5%);  laya_english_2048 4 | 3 | 101 (26.7%);  jev_or_zeroshot 271 | 32 | 339 (89.7%);
  jev_or_fewshot4_v1 272 | 23 | 349 (92.3%);  jev_or_fewshot_v2 269 | 25 | 344 (91.0%). JEV AUROC on this subset 0.964 / 0.975 / 0.975.
- Capacity of the dataset for a larger meaning-only test: 2,591 form-clean plans (660 multi-step, 1,931 single-step), all within the 8k state limit after trimming tools.
  One edit of each kind per plan would give 4,863 meaning-type edited cases (remove 660, replace_arg 1,612, add_unrelated_tool 2,012, add_duplicate 579; remove needs >=2 steps)
  plus 2,591 valid plans = 7,454 cases. The raw data has no labels for meaning errors; these cases are generated by editing form-clean plans.
