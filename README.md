# Typed Plans, Lightweight Judges

Execution-aware verification for tool-calling agents (LMA group project, IIIT Hyderabad).

An LLM writes the plan as a **typed intermediate representation (IR)**. The plan is checked by a **lightweight classifier** (Laya or JEV) before execution and executed in **dependency-aware waves**. Each wave's outcomes are classified again. Only flagged steps go to an **LLM judge**, which patches the affected part of the IR. Execution then resumes, and verified outputs are reused from a **cache**.

```
query ─▶ planner ─▶ IR ─▶ gate(plan) ──flag──▶ LLM IR editor ─┐
                           │ ok                    ▲          │
                           ▼                       └──────────┘
                    wave execution ─▶ gate(outcomes) ──flag──▶ LLM judge ─▶ local IR patch
                           ▲               │ ok                               │
                           │               ▼                                  │
                           └──── next wave ◀── cache ◀── re-run dirty steps ◀─┘
```

New to the code? Start with [docs/CODE_WALKTHROUGH.md](docs/CODE_WALKTHROUGH.md). It gives a reading order, explains how Hydra builds the objects and how tracing works, and follows one task through the code.

## Layout

```
configs/                  Hydra config groups (one per pipeline box, see below)
src/tpj/
  ir/                     IR schema, $ref resolution, waves / dirty regions, plan formats (RQ1)
  llm/client.py           one OpenAI-compatible client for vLLM and hosted APIs, disk-cached
  planner/                prompted / fine-tuned planner, gold (oracle) planner
  gate/                   Laya (local HF), JEV (HTTP), LLM classifier, constant, oracle
  judge/                  LLM IR editor (pre-execution) and LLM judge (post-execution)
  repair/                 patch ops (edit/remove/replace/add) + local / full-replan / none
  executor/               execution state, wave vs sequential scheduling
  cache/                  verified-output cache keyed by tool + resolved args
  faults/                 plan- and execution-level fault injection (ground-truth labels)
  agent/                  run.py = Figure-1 state machine; VerifiedAgent; ReAct baseline
  benchmarks/             toy, NESTFUL, TaskBench, BFCL, tau2 adapters
  eval/                   trace events, metrics, report table
  finetune/sft.py         QLoRA SFT of the IR planner
scripts/                  run_agent.py, run_tau2.py, finetune.py, report.py
tests/                    IR, formats, patches, end-to-end loop (no models needed)
```

## Setup

Python ≥ 3.10, GPU machine.

```bash
pip install -e ".[gpu,dev]"            # core + Laya/QLoRA deps
pip install vllm                         # model serving
pip install git+https://github.com/sierra-research/tau2-bench   # only for tau2
pytest                                   # runs without any model
```

Serve the open models (OpenAI-compatible):

```bash
vllm serve Qwen/Qwen3-8B  --port 8000 --enable-lora --lora-modules ir_planner=checkpoints/ir_planner_qwen3_8b
vllm serve Qwen/Qwen3-32B --port 8001
export VLLM_SMALL_URL=http://localhost:8000/v1 VLLM_LARGE_URL=http://localhost:8001/v1
```

API keys, only for the components you use: `OPENAI_API_KEY`, `TYPESAFE_API_KEY` (JEV), `DASHSCOPE_API_KEY` (Qwen Max).

Benchmark data:
- NESTFUL and TaskBench load from Hugging Face automatically. For executable NESTFUL tools, clone `IBM/NESTFUL` and set `NESTFUL_FUNCTIONS_DIR=<clone>/data_v2/executable_functions`. Without it, mocks plus sequence metrics are used.
- BFCL: clone `ShishirPatil/gorilla` and set `BFCL_DATA_DIR=<clone>/berkeley-function-call-leaderboard/bfcl_eval/data`.
- tau2: installed as a package (see above).

## Running

```bash
# smoke test: oracle plan, no models
python scripts/run_agent.py benchmark=toy planner=gold gate=always_pass

# full system on NESTFUL with light fault injection
python scripts/run_agent.py benchmark=nestful gate=laya faults=light

# swap anything from the CLI
python scripts/run_agent.py benchmark=taskbench gate=jev model@models.judge=gpt4o_mini limit=50

# baselines
python scripts/run_agent.py +experiment=baseline_direct benchmark=nestful   # no verification
python scripts/run_agent.py agent=react benchmark=nestful                   # ReAct

# experiment sweeps (Hydra multirun)
python scripts/run_agent.py -m +experiment=rq1_format benchmark=taskbench
python scripts/run_agent.py -m +experiment=rq2_gating benchmark=nestful
python scripts/run_agent.py -m +experiment=ablations  benchmark=nestful
python scripts/run_agent.py -m +experiment=threshold_sweep gate=laya
python scripts/report.py multirun/<date>/<time>

# tau2-bench
python scripts/run_tau2.py benchmark=tau2 benchmark.domain=airline benchmark.num_tasks=20

# fine-tune the planner once the synthetic IR dataset exists
python scripts/finetune.py finetune.train_file=data/ir/train.jsonl finetune.eval_file=data/ir/dev.jsonl
python scripts/run_agent.py planner=finetuned benchmark=nestful
```

Each run writes `results.jsonl` (per-task plan, calls, score), `trace.jsonl` (every event) and `metrics.json` to its Hydra output directory. Responses from the LLMs and JEV are cached on disk (`.cache/`), so re-running an unchanged config makes no new calls.

## Config groups

| Group | Options | Pipeline box |
|---|---|---|
| `model@models.<role>` | `qwen3_8b_vllm`, `qwen3_32b_vllm`, `planner_lora`, `qwen_max_api`, `gpt4o_mini`, `gpt56_luna`, `gpt6_luna_pro` | model for each role: `planner`, `planner_ft`, `editor`, `judge`, `gate_llm`, `converter`, `react`, `responder` |
| `plan_format` | `typed_ir`, `graph`, `free_text` | plan representation (RQ1) |
| `planner` | `prompted`, `finetuned`, `gold` | LLM planner |
| `gate` | `laya`, `jev`, `llm_classifier`, `always_pass`, `always_flag`, `oracle` | pre-/post-execution classifier |
| `ir_editor` | `llm`, `none` | LLM IR editing |
| `judge` | `llm`, `none` | LLM-as-a-judge |
| `repair` | `local`, `full_replan`, `none` | IR repair |
| `scheduler` | `waves`, `sequential` | wave-wise execution |
| `cache` | `node`, `none` | cache of successful outputs |
| `faults` | `none`, `light`, `heavy`, `silent` | controlled failures |
| `benchmark` | `toy`, `nestful`, `taskbench`, `bfcl`, `tau2` | evaluation |
| `agent` | `verified`, `react` | system vs baseline |
| `+experiment` | `baseline_direct`, `baselines`, `rq1_format`, `rq2_gating`, `ablations`, `threshold_sweep` | presets / sweeps |

The gate flags a plan or an outcome when `1 - P(OKAY) >= gate.threshold`. `gate.check_plan` and `gate.check_post` turn the two checkpoints on and off independently.

## The IR

```json
{"steps": [
  {"id": "s1", "tool": "search_flights", "args": {"origin": "BLR", "dest": "GOI"}, "returns": "list[Flight]", "safety": "safe"},
  {"id": "s5", "tool": "pick_cheapest", "args": {"options": [{"$ref": "s1"}, {"$ref": "s2"}]}, "returns": "Option", "safety": "safe"},
  {"id": "s7", "tool": "book_transport", "args": {"option_id": {"$ref": "s5", "path": "id"}}, "returns": "Booking", "safety": "danger"},
  {"id": "s9", "tool": "book_indoor_activity", "args": {"people": 4}, "returns": "Booking", "safety": "danger",
   "when": {"ref": "s6", "equals": true}}
 ],
 "final": ["s11"]}
```

Dependencies exist only through `$ref` (and `when`), and waves are derived from them. The full Figure 2 example is in [src/tpj/ir/examples.py](src/tpj/ir/examples.py). A reference to a step that was skipped by its condition resolves to `null`.

The synthetic IR dataset for fine-tuning should be JSONL lines of `{"query": ..., "tools": [ToolSpec], "plan": {"steps": [...], "final": [...]}}`. `finetune/sft.py` builds prompts with the same `PromptedPlanner.build_messages` used at inference.

## Metrics (from `trace.jsonl`)

- **Success Rate**: benchmark-specific `success` (NESTFUL answer/sequence match, TaskBench node+edge match, BFCL AST match, tau2 reward from tau2's own output)
- **Avg tokens / API calls**: split by role (planner, gate, ir_editor, judge, converter, react, responder)
- **Gate P/R/F1/accuracy**: against injected faults, separately for plan level and node level. Raised exceptions are not counted because they never reach the gate.
- **Unsafe Action Detection Rate**: injected plan faults flagged before execution ÷ injected plan faults
- **False Intervention Rate**: valid plans and nodes that were flagged ÷ all valid plans and nodes the gate saw
- **Cache reuse, re-executions, interventions, repair success rate**

## Planned extensions (not implemented yet)

These are deliberately left out of the code for now:

1. **Rule-based IR validator (free, deterministic pre-check).** Runs before the gate and catches structural problems at zero model cost:
   - every `$ref` points to an existing earlier step, and the graph is acyclic
   - the tool exists and the args match its JSON schema (required params, types)
   - a referenced step's `returns` type is compatible with the parameter type
   - a step whose tool is side-effecting must not be tagged `safe`
   - `when` references a `bool` step, and no step is dead (unused and not final)

   Issues would go to the IR editor together with the gate verdict. Splitting "structural (free)" from "semantic (gate)" would also give one more ablation. Today only the minimum needed to execute is checked (unknown refs, cycles, duplicate ids, in `ir/dag.py::check_structure`).
2. **DANGER pre-step check.** Run the gate on every `safety: danger` step right before it executes, with resolved arguments, and block or route it to the judge before the side effect happens. This would make an action-level Unsafe Action Detection Rate measurable. The current rate is measured at plan level. The `safety` tag is already part of the IR and the traces (`node_result.safety`), so this needs a hook in `PlanRun.next_wave` plus a `faults` type that injects unsafe danger calls.

## Notes / things to verify

- **Laya**: the HF card `convaiinnovations/layab` is gated. `gate/laya.yaml` supports a causal-LM option scorer (`choice_lm`) and a classification head (`seq_cls`), so pick the one that matches the card.
- **JEV**: uses `POST https://api.typesafe.ai/v1/systemone` with one `choice` question. The response parsing accepts both `answers.<id>.probabilities` and inline option probabilities.
- **tau2**: the adapter follows tau2's agent README (`HalfDuplexAgent`, `generate_next_message`, `registry.register_agent_factory`). Re-check message class names against the installed version. tau2 tools are real, so fault injection is off there.
- **BFCL**: single-turn categories only. Multi-turn needs BFCL's stateful executable backends.
- **Model ids** for `gpt56_luna`, `gpt6_luna_pro` and `qwen_max_api` come from the proposal. Confirm the exact API ids.
