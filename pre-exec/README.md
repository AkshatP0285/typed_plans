# Pre-execution plan verification: decision-model comparison

Tests whether a decision model (JEV, Laya, ...) can tell a valid plan (IR) from an edited one
(removed / replaced / added step) *before* anything runs. Everything is driven by
[experiments.toml](experiments.toml).

## Quick start

```powershell
python run.py list                              # experiments, backends, how many cases each has scored
python run.py run --only rules_baseline         # free, no model
$env:OPENROUTER_API_KEY="..."                   # JEV via OpenRouter (sk-or-v1-...); never put keys in the config
python run.py run --only jev_or_zeroshot,jev_or_fewshot4_v1,jev_or_fewshot_v2
.venv-laya\Scripts\python run.py run --only laya_english_2048   # Laya needs the venv (torch + laya)
python run.py report                            # paired comparison table
python run.py report --meaning-only             # drop edits a rule checker can catch (valid plans + meaning-type edits only)
python run.py report --only jev_or_zeroshot,jev_or_fewshot_v2 --detail jev_or_fewshot_v2
```

Uncertainty: `python run.py run --only a,b --rep 2` (and 3) repeat a run into `<name>.repK.jsonl`; `python run.py report --only a,b --repeats 3 --bootstrap 2000` prints mean +- std over repeats and 95% bootstrap CIs (resampled by base plan) incl. differences vs the first experiment.

`run` is resumable (scored cases are skipped, failures retried), `--limit N` smoke-tests, `--dry-run`
previews without loading a model. Results land in `results/<experiment>.jsonl`; `report` writes
`results/comparison.csv`.

## Choosing what to test
Edit `experiments.toml`:
- **enable / disable** an experiment with `enabled = true|false` (`--only name` runs it regardless).
- **add a model variant** by copying a `[[experiments]]` block and changing `params`
  (e.g. Laya `ckpt`, `max_len`; JEV `model`).
- **try prompt wording** by adding a `[prompts.<name>]` block and setting `prompt = "<name>"`.
- **different data** with `data = "path.jsonl"` per experiment or in `[defaults]`.

## Adding a new decision model
In [harness/backends.py](harness/backends.py) subclass `Backend`, implement `classify(state, labels,
instructions, experiment_id) -> Prediction` (label + probabilities), decorate with `@register("name")`,
then use `backend = "name"` in the config. Nothing else changes.

## Adding a new task (e.g. post-execution)
Add a `Task` in [harness/tasks.py](harness/tasks.py) (how to build the state from a case, which field
is the truth). Reports are currently written for a binary positive/negative task.

## Layout
| path | role |
|---|---|
| `experiments.toml` | experiments, prompts, defaults |
| `run.py` | CLI: list / run / report |
| `harness/` | backends, tasks, config loader, runner, report |
| `test_set/` | frozen 466-case test set (+ README, checksum) |
| `build_pre_cases.py` | generates the test set from the IR dataset (do not rerun between experiments) |
| `results/` | per-experiment predictions, `RUN_LOG.md`, comparison tables |
| `build_fewshot.py`, `build_fewshot_v2.py` | pick the few-shot demonstrations (from plans outside the test set) |

## Notes
- Compare models on the same cases: `report` is paired by default (cases every selected experiment scored).
- Laya English truncates inputs at 512 tokens by default; our plans average ~1.4k, hence `max_len = 2048`.
- Laya's probabilities ship over-confident (see its model card), so AUROC is more trustworthy than the argmax numbers.
