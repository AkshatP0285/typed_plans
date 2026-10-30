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

## Setup

Python >= 3.10 (3.12 for tau2), GPU machine.

```bash
pip install -e ".[gpu,dev]"
pip install vllm                                                   # model serving
pip install git+https://github.com/sierra-research/tau2-bench     # only for tau2
pytest                                                             # runs without any model
```

Serve the planner (4B) and the editor / judge (27B), both OpenAI-compatible:

```bash
vllm serve Qwen/Qwen3-4B --port 8002
vllm serve <Qwen 27B model> --port 8004 --tensor-parallel-size 2 --max-model-len 16384 --reasoning-parser qwen3
export VLLM_4B_URL=http://localhost:8002/v1 VLLM_27B_URL=http://localhost:8004/v1
export OPENROUTER_API_KEY=...        # JEV; set in the shell only, never commit it
```

Benchmark data:
- TaskBench loads from Hugging Face automatically. NESTFUL needs `NESTFUL_FUNCTIONS_DIR=<IBM/NESTFUL clone>/data_v2/executable_functions` for executable tools.
- BFCL: clone `ShishirPatil/gorilla` and set `BFCL_DATA_DIR=<clone>/berkeley-function-call-leaderboard/bfcl_eval/data`.
- tau2: set `TAU2_DATA_DIR=<tau2-bench clone>/data`.

## Running

```bash
# smoke test: oracle plan, no models
python scripts/run_agent.py benchmark=toy planner=gold gate=always_pass

# common overrides used in the experiments
M="model@models.planner=qwen3_4b_vllm model@models.editor=qwen3_27b_vllm model@models.judge=qwen3_27b_vllm model@models.converter=qwen3_4b_vllm"

# BFCL Non-Live, typed IR (swap plan_format=graph | free_text_rule)
python scripts/run_agent.py benchmark=bfcl "benchmark.categories=[simple_python,multiple,parallel,parallel_multiple,irrelevance]" \
  benchmark.limit=1235 planner=prompted planner.few_shot=false plan_format=typed_ir gate=jev ir_editor=llm judge=llm $M

# JEV sees the IR as text; post-checks only on danger steps
python scripts/run_agent.py benchmark=bfcl ... gate=jev gate.plan_view=ir_text gate.post_steps=danger

# TaskBench with retrieved examples (subset: dailylifeapis | huggingface | multimedia)
python scripts/run_agent.py benchmark=taskbench benchmark.subset=huggingface benchmark.limit=500 \
  planner=retrieval plan_format=typed_ir_dep5r gate=jev ir_editor=llm judge=llm $M

# ablations
python IR_Ablations/error_find.py              # planted-error detection per format
python IR_Ablations/jev_format_input.py        # JEV input view on stored plans
python IR_Ablations/dep_analysis.py            # dependency recall / precision / waves
python IR_Ablations/static_check.py

# sweeps and report
python scripts/run_agent.py -m +experiment=rq1_format benchmark=taskbench
python scripts/report.py multirun/<date>/<time>
```

Each run writes `results.jsonl` (per-task plan, calls, score), `trace.jsonl` (every event) and `metrics.json` to its Hydra output directory. Responses from the LLMs and JEV are cached on disk (`.cache/`), so re-running an unchanged config makes no new calls.

## The IR

```json
{"steps": [
  {"id": "s1", "tool": "search_flights", "args": {"origin": "BLR", "dest": "GOI"}, "after": [], "returns": "list[Flight]", "safety": "safe"},
  {"id": "s5", "tool": "pick_cheapest", "args": {"options": [{"$ref": "s1"}, {"$ref": "s2"}]}, "after": ["s1", "s2"], "returns": "Option", "safety": "safe"},
  {"id": "s7", "tool": "book_transport", "args": {"option_id": {"$ref": "s5", "path": "id"}}, "after": ["s5"], "returns": "Booking", "safety": "danger"},
  {"id": "s9", "tool": "book_indoor_activity", "args": {"people": 4}, "after": ["s6"], "returns": "Booking", "safety": "danger",
   "when": {"ref": "s6", "equals": true}}
 ],
 "final": ["s11"]}
```
