# Implementation Plan: Typed Plans, Lightweight Judges

Implementation plan for the pipeline in `proposal_template__2_.pdf` (Section 7.1, Figure 1).

**Decisions so far:** JSON IR validated with Pydantic · open models served by vLLM (OpenAI-compatible) · Laya is the default gate and JEV plugs in · all four benchmarks (NESTFUL, TaskBench, BFCL, τ²-Bench) sit behind one adapter interface · Hydra handles configuration and sweeps.

> The course brief (§6) says not to start implementation until the TA approves the idea. Phase 0 below is setup only and is safe to begin.

> **Status (2026-10-08): code scaffold written.** See README.md. Differences from this plan:
> - The rule-based IR validator (§3.1) and the DANGER pre-step gate check (§3.4) are **not implemented**. They are listed under "Planned extensions" in README.md. Only the minimal executability check (unknown refs, cycles, duplicate ids) remains.
> - Tool retrieval / RAG (§10.1) is dropped.
> - Synthetic IR data generation (§6, steps 1–3) will be built separately. `finetune/sft.py` already consumes its JSONL format.
> - Hydra groups are named `scheduler` (not `executor`) and `model@models.<role>` (not `llm`), and there is an extra `faults=silent`.

---

## 1. Design principles

1. **Each box in Figure 1 is a swappable module.** Every stage has an abstract base class plus one or more implementations. Hydra picks the implementation through `_target_`, so an ablation is a command-line override, not a code change.
2. **There is one IR.** Planner, compiler, validator, gate, judge, repair, cache and fine-tuning data all read and write the same Pydantic `Plan` object.
3. **Every run produces a trace.** Each LLM call, gate decision, tool call, fault injection and repair is logged as a JSONL event. All metrics in §11 of the proposal are computed from traces after the run, so no metric needs a re-run.
4. **The budget is tight ($30 API, ~100 GPU h).** Every LLM response is cached on disk, keyed by (model, prompt, params). Re-running an evaluation with unchanged prompts costs nothing.
5. **Benchmarks are adapters.** Each benchmark is converted to a common `Task` with a `ToolSpec` list and its own scorer. The agent never sees benchmark-specific code.

---

## 2. Repository layout

```
typed-plans/
├── pyproject.toml                  # uv / pip; python 3.11
├── configs/                        # Hydra config tree (see §4)
├── src/tpj/
│   ├── ir/
│   │   ├── schema.py               # Step, Plan, TypeRef, Safety, Condition (Pydantic)
│   │   ├── validate.py             # static checks → list[IRIssue]
│   │   ├── dag.py                  # dependency graph, wave levels, dirty-region calc
│   │   └── formats/                # RQ1: text / graph / typed-IR plan formats + parsers
│   ├── tools/
│   │   ├── spec.py                 # ToolSpec: name, params schema, return type, safety
│   │   ├── registry.py             # name → callable, per task
│   │   └── mocks/                  # deterministic mock tools per benchmark
│   ├── llm/
│   │   ├── client.py               # OpenAI-compatible client (vLLM + APIs), retries
│   │   ├── cache.py                # diskcache response cache
│   │   └── usage.py                # token / call accounting per role
│   ├── planner/                    # query + tools → Plan
│   │   ├── base.py
│   │   ├── prompted.py             # few-shot / JSON-schema-constrained
│   │   └── prompts/
│   ├── compiler/
│   │   └── compile.py              # Plan → ExecutablePlan (ref resolution, conditions)
│   ├── gate/                       # lightweight classifier: OKAY / ERROR / NEEDS_FORMAT_CHANGE
│   │   ├── base.py                 # Gate.classify(GateInput) → GateOutput(label, probs)
│   │   ├── laya.py                 # local HF model
│   │   ├── jev.py                  # TypeSafe API
│   │   ├── llm_classifier.py       # prompted LLM fallback
│   │   └── constant.py             # always_pass / always_flag (ablations)
│   ├── judge/
│   │   └── llm_judge.py            # (IR, flagged nodes, outputs) → Diagnosis + IRPatch
│   ├── repair/
│   │   ├── patch.py                # IRPatch ops: edit / remove / replace / add
│   │   ├── local.py                # apply patch, revalidate, compute dirty region
│   │   └── full_replan.py          # baseline: replan everything
│   ├── executor/
│   │   ├── waves.py                # asyncio.gather per wave
│   │   ├── sequential.py           # ablation: one node at a time
│   │   └── state.py                # ExecutionState: node status, outputs, history
│   ├── cache/
│   │   └── node_cache.py           # key = hash(tool, resolved args, upstream keys)
│   ├── faults/
│   │   ├── plan_faults.py          # drop step, bad ref, wrong tool, missing DANGER tag…
│   │   └── exec_faults.py          # exception, timeout, empty, wrong type, wrong value…
│   ├── agent/
│   │   ├── loop.py                 # the Figure 1 orchestrator
│   │   └── baselines/
│   │       ├── react.py            # ReAct, no plan
│   │       └── direct.py           # same planner + tools, no verification/recovery
│   ├── benchmarks/
│   │   ├── base.py                 # Task, BenchmarkAdapter(load, tools, score)
│   │   ├── nestful.py
│   │   ├── taskbench.py
│   │   ├── bfcl.py
│   │   ├── tau2.py
│   │   └── toy.py                  # Goa-trip style hand-made tasks for unit tests
│   ├── eval/
│   │   ├── trace.py                # event schema + JSONL writer
│   │   ├── metrics.py              # SR, tokens, calls, gate P/R/F1, UADR, FIR, reuse
│   │   └── report.py               # aggregate multirun dirs → tables / plots
│   ├── data/
│   │   ├── build_ir_dataset.py     # ToolACE + When2Call → IR via Qwen teacher
│   │   ├── filter.py               # keep only IRs that validate (+ execute on mocks)
│   │   └── annotate.py             # stratified sample + rubric sheet for humans
│   └── finetune/
│       └── sft_lora.py             # TRL + PEFT (QLoRA) on the IR dataset
├── scripts/                        # thin @hydra.main entry points
│   ├── run_agent.py                # evaluate a pipeline on a benchmark
│   ├── build_data.py
│   ├── finetune.py
│   ├── eval_gate.py                # gate classification eval in isolation
│   └── report.py
├── tests/                          # pytest: IR validation, waves, patch, cache, faults
└── outputs/                        # Hydra run dirs (gitignored)
```

---

## 3. Core components

### 3.1 IR (`ir/schema.py`)

```python
class Step(BaseModel):
    id: str                          # "s5"
    tool: str                        # "pick_cheapest"
    args: dict[str, Arg]             # literal | {"$ref": "s1"} | {"$ref": "s3", "path": "first.id"} | list
    returns: TypeRef                 # "list[Flight]", "Booking", "bool"
    safety: Literal["safe", "danger"]
    when: Condition | None = None    # {"ref": "s6", "equals": true}
    rationale: str | None = None     # optional, for analysis only

class Plan(BaseModel):
    query: str
    steps: list[Step]
    final: list[str]                 # step ids whose outputs answer the query
```

Dependencies come only from `$ref`s, as in Figure 2. The plan never states execution order.

**Static validator (`ir/validate.py`).** This is the deterministic, free layer of pre-execution verification, and it runs before the gate:
- every `$ref` points to an earlier, existing step, and the graph is acyclic
- the tool exists and the args match its JSON schema (required params, types)
- the referenced step's `returns` type is compatible with the param type
- a step whose tool's spec says `danger` must not be tagged `safe` (catches mislabels)
- `when` references a `bool` step, and no dead steps (outputs never used and not in `final`)

It returns `list[IRIssue]`, which goes to the LLM IR editor. Splitting "structural check (free)" from "semantic check (gate)" also gives a clean ablation.

**`ir/dag.py`**
- `waves(plan)`: topological levels (Kahn's algorithm). This reproduces Figure 3.
- `dirty_region(old, new)`: changed steps plus all their descendants. Everything else keeps its cache entry.

### 3.2 Planner
`Planner.plan(query, tools, examples) -> Plan`. The prompted planner uses vLLM's `guided_json` with `Plan.model_json_schema()`, so output always parses and failures are semantic, not syntactic. The fine-tuned planner is the same class with a different `llm` config pointing at the LoRA adapter served by vLLM (`--enable-lora`).

### 3.3 Compiler
Binds each step to a registry callable and resolves `$ref` paths at run time from `ExecutionState`. Steps whose `when` is false are marked `skipped`.

### 3.4 Gate (Laya / JEV)
```python
class GateInput:  query, step, tool_spec, resolved_args, output | None, phase: "pre"|"post"
class GateOutput: label in {OKAY, ERROR, NEEDS_FORMAT_CHANGE}, probs, latency
```
- **Pre-execution:** run on the whole plan (plan-level check), and **also on each `danger` step just before it runs**. This per-step check is what makes "Unsafe Action Detection Rate (before execution)" measurable.
- **Post-execution:** run on each node's output after its wave.
- The decision policy lives in config: `route_to_judge if p(ERROR)+p(NEEDS_FORMAT_CHANGE) > threshold`. The threshold is swept to produce cost-vs-reliability curves.
- `NEEDS_FORMAT_CHANGE` can first try a cheap deterministic coercion (e.g. str→int, unwrap a list) before calling the judge. This can be turned off.

### 3.5 Judge and local repair
The judge receives the IR, the flagged nodes, their upstream outputs and the tool specs, and returns:
```python
class IRPatch: ops: list[Edit | Remove | Replace | Add], diagnosis: str, verdict: "repair"|"false_alarm"|"abort"
```
`repair/local.py` applies the patch, re-runs the static validator, computes the dirty region, and hands it back to the executor. `false_alarm` verdicts are logged; they feed the False Intervention Rate. There is a max-repair budget per task, set in config.

### 3.6 Executor and cache
Wave loop: `for wave in waves(plan): await gather(run(n) for n in wave if not cached(n))`. The node cache key is `hash(tool, resolved_args, cache_keys_of_parents)`, so a repaired upstream node automatically invalidates descendants and nothing else.

### 3.7 Orchestrator (`agent/loop.py`), i.e. Figure 1
```
plan = planner.plan(task)
loop (≤ max_plan_edits):
    issues = validator(plan) + gate.pre(plan)
    if none: break
    plan = ir_editor.fix(plan, issues)
state = ExecutionState()
for wave in waves(plan):            # recomputed after each repair
    for n in wave if n.safety == danger: gate.pre_step(n) → block / judge
    execute wave (cache-aware, faults injected here)
    flags = [n for n in wave if gate.post(n) not OKAY]
    if flags:
        patch = judge(plan, flags, state)
        plan, dirty = repair(plan, patch)
        state.invalidate(dirty)        # resume; valid nodes are kept
answer = collect(plan.final, state)
```
Every arrow in this loop is a module chosen by config, so each ablation in §12 of the proposal is a single override.

### 3.8 Fault injection (`faults/`)
The faults provide the ground-truth labels the gate is evaluated against, which also covers the brief's "known ground truth" requirement.

| Level | Fault types | Label |
|---|---|---|
| Plan | drop step, wrong `$ref`, wrong tool, swapped args, `danger` tagged `safe`, unnecessary danger step | per plan + per step |
| Exec | exception, timeout, empty result, wrong type, schema drift (renamed field), plausible-but-wrong value, unsafe side effect | per node |

Each fault has an injection rate and a seed in config (`faults=none|light|heavy`). Every injected fault is written to the trace with an id, so P/R/F1 and UADR are exact.

---

## 4. Hydra config tree

```
configs/
├── config.yaml                 # defaults list + seed, output dir, run name
├── llm/          qwen_7b_vllm.yaml · qwen_32b_vllm.yaml · gpt4o_mini.yaml · planner_lora.yaml
├── planner/      prompted.yaml · finetuned.yaml
├── plan_format/  typed_ir.yaml · graph.yaml · free_text.yaml          # RQ1
├── validator/    static.yaml · none.yaml
├── gate/         laya.yaml · jev.yaml · llm_classifier.yaml · always_pass.yaml · always_flag.yaml
├── judge/        llm_judge.yaml · none.yaml
├── repair/       local.yaml · full_replan.yaml · none.yaml
├── executor/     waves.yaml · sequential.yaml
├── cache/        node.yaml · none.yaml
├── faults/       none.yaml · light.yaml · heavy.yaml
├── benchmark/    nestful.yaml · taskbench.yaml · bfcl.yaml · tau2.yaml · toy.yaml
├── agent/        full.yaml · direct.yaml · react.yaml                 # baselines
├── experiment/   rq1_format.yaml · rq2_gating.yaml · ablations.yaml · threshold_sweep.yaml
├── data_gen/     toolace_when2call.yaml
└── finetune/     qlora_qwen7b.yaml
```

`config.yaml`:
```yaml
defaults:
  - llm@planner_llm: qwen_7b_vllm
  - llm@judge_llm: qwen_32b_vllm
  - planner: prompted
  - plan_format: typed_ir
  - validator: static
  - gate: laya
  - judge: llm_judge
  - repair: local
  - executor: waves
  - cache: node
  - faults: light
  - benchmark: nestful
  - agent: full
  - _self_

seed: 0
limit: null          # cap #tasks for quick runs
max_plan_edits: 2
max_repairs: 3
```

`gate/laya.yaml`:
```yaml
_target_: tpj.gate.laya.LayaGate
model_id: convaiinnovations/layab
device: cuda
threshold: 0.5
check_danger_pre: true
check_post: true
```

Example runs:
```bash
# full system on NESTFUL
python scripts/run_agent.py benchmark=nestful

# RQ2 gating ablation, all benchmarks (multirun)
python scripts/run_agent.py -m gate=always_pass,always_flag,laya,jev \
    benchmark=nestful,taskbench,bfcl,tau2

# component ablations
python scripts/run_agent.py -m +experiment=ablations   # sweeps validator/cache/repair/executor

# RQ1: plan formats, prompted vs fine-tuned
python scripts/run_agent.py -m plan_format=free_text,graph,typed_ir planner=prompted,finetuned
```
Each run directory holds `.hydra/config.yaml`, `trace.jsonl` and `metrics.json`. `scripts/report.py` reads a multirun directory and produces the tables.

---

## 5. Benchmarks

All four use the same adapter interface: `load(split, limit) -> list[Task]`, `tools(task) -> list[ToolSpec] + callables`, `score(task, trace) -> dict`.

| Benchmark | What it tests here | Integration notes | Native metric | Order |
|---|---|---|---|---|
| **NESTFUL** | nested calls with dependencies; the core IR + wave case | Gold sequences already use variable references, so they map almost 1:1 to IR. Executable subset can run for real; the rest uses mocks. | full-sequence accuracy, win rate | 1st (MVP) |
| **TaskBench** | plan structure quality (RQ1), GNNVerifier comparison | Tool graphs come without execution, so outputs are mocked from tool return types. | node F1, edge F1, args F1 | 2nd |
| **BFCL** | single/parallel/multi-turn call accuracy, sanity check | Use exec + multi-turn categories. Plans are shallow, so expect small gains; that is a useful negative result. | AST / exec accuracy | 3rd |
| **τ²-Bench** | stateful, conversational, real side effects (DANGER steps) | Plug in as a custom agent. Plan per user turn with real domain tools. This is the heaviest integration. | pass^k | 4th |

All benchmarks are configured from the start, and they are integrated in this order so the pipeline is proven on the easiest fit first. Use small fixed subsets (e.g. 150–300 tasks each, stratified by plan depth) to stay within budget, as the brief recommends.

---

## 6. Data and fine-tuning

1. **Teacher generation** (`data/build_ir_dataset.py`): for each ToolACE / When2Call sample, prompt the teacher (Qwen large) with the query, tool list, an example IR and the IR rules, using `guided_json`. When2Call "no tool needed" cases become empty plans or clarification plans.
2. **Automatic filter:** keep samples that pass the static validator, that execute on mocks without error, and (where gold calls exist) whose tool set matches. Log the rejection rate per reason.
3. **Human evaluation:** take a stratified sample (by depth, #tools, source) of about 200 items, 50 per team member, with ~20% overlap for inter-annotator agreement (Cohen's κ). Rubric: correctness, completeness, dependency structure, executability, each scored 1–3.
4. **QLoRA SFT** (`finetune/sft_lora.py`): Qwen ~7B on TRL `SFTTrainer` with PEFT, 4-bit. Track IR validity rate on a held-out set. Serve the result through vLLM `--lora-modules`.
5. **Gate data (optional):** the fault-injected traces give labelled (input, output, label) triples. If zero-shot Laya is weak, a small LoRA on it is an easy extra fine-tuning result.

---

## 7. Metrics (all computed from traces)

| Metric | Computed as |
|---|---|
| Success Rate | benchmark scorer |
| Avg tokens / API calls | `usage.py`, split by role: planner, gate, judge, ir_editor |
| Gate acc / P / R / F1 | gate label vs injected-fault label per node (pre and post separately) |
| Unsafe Action Detection Rate | injected unsafe `danger` actions blocked before execution ÷ all injected unsafe actions |
| False Intervention Rate | valid nodes routed to judge, modified or blocked ÷ all valid nodes |
| Computation preserved | cache-reused nodes ÷ nodes that would have re-run under full replan |
| Repair success | repaired tasks that end successfully ÷ tasks with ≥1 repair |
| IR human score | rubric mean, plus κ |
| Judge validity | judge verdicts vs human labels on a sample (required by the brief §7) |

---

## 8. Experiments mapped to research questions

- **RQ1 (does the IR format matter):** `plan_format ∈ {free_text, graph, typed_ir}` × `planner ∈ {prompted, finetuned}`, with no faults and gate off, on TaskBench + NESTFUL. Measure plan validity, node/edge F1 and SR.
- **RQ2 (selective gating):** `gate ∈ {always_pass (no review), always_flag (judge every node), laya, jev}`, with faults=light/heavy. Measure SR, judge calls, tokens, UADR and FIR. Add a threshold sweep for a cost–reliability curve.
- **Ablations (§12):** remove each of validator, pre-gate, post-gate, judge, local repair (→ full replan), cache and waves (→ sequential).
- **Baselines:** `agent=react`, `agent=direct`. GNNVerifier, Reinforced Agent and ToolTree use their reported numbers on overlapping benchmarks (TaskBench for GNNVerifier), and are reproduced only if code is released.

---

## 9. Phases

| Phase | Output | Exit criterion |
|---|---|---|
| **0. Scaffold** (week 1) | repo, Hydra tree, LLM client + disk cache, trace writer, toy benchmark (Goa trip) | `run_agent.py benchmark=toy agent=direct` produces a trace |
| **1. Core IR** (wk 1–2) | schema, validator, waves, compiler, executor, node cache, unit tests | Figure 2 IR yields Figure 3's waves in a test |
| **2. MVP loop** (wk 2–3) | prompted planner, LLM-classifier gate stub, judge, local repair, fault injector, NESTFUL adapter | full loop runs on 50 NESTFUL tasks with metrics.json |
| **3. Real gate** (wk 3) | Laya gate, JEV gate, `eval_gate.py` | gate P/R/F1 table on injected faults |
| **4. More benchmarks** (wk 3–5) | TaskBench, BFCL, τ²-Bench adapters | each produces a scored run with `agent=direct` and `agent=full` |
| **5. Data + FT** (wk 2–5, parallel) | IR dataset, human annotation, QLoRA planner | fine-tuned validity ≥ prompted 7B baseline |
| **6. Experiments** (wk 5–6) | RQ1, RQ2, ablations, sweeps | all tables in `report.py` |
| **7. Analysis + write-up** (wk 7) | failure cases, qualitative examples, paper | — |

**Suggested split for 4 people** (all of this can be reassigned):
- A: IR + validator + compiler + executor + cache
- B: gate (Laya/JEV) + judge + repair + fault injection
- C: benchmark adapters + metrics/report
- D: data generation + human-eval tooling + QLoRA fine-tuning

The interfaces in §3 should be frozen at the end of Phase 0 so the four tracks can proceed in parallel.

---

## 10. Gaps in the proposal

1. **RAG is not covered.** Section 8 says five capabilities, but the table lists four. A cheap and well-motivated fix is **tool retrieval**: embed tool docs and retrieve the top-k tools for the planner, which matters for ToolACE/TaskBench-sized tool lists. It fits as a `retriever/` module with `retriever=none|bm25|dense` and is measured by tool recall@k and SR.
2. **Laya's input format is unverified.** Confirm what its model card expects (query + options + context) before building `GateInput` formatting. If it is not a good fit, `llm_classifier` is the fallback.
3. **Budget.** $30 covers JEV plus a few GPT-4o-mini runs at most. The default judge, planner and teacher should be local Qwen on vLLM, with API models only for a small comparison slice.
4. **"Graph" format for RQ1.** Define it concretely as nodes and edges without types or safety tags (TaskBench style). The comparison then isolates exactly what "typed" adds.

---

## 11. Open questions

- What is the submission deadline? It fixes the week numbers above.
- Which GPU is available (type/VRAM)? It decides 7B vs 14B planner and whether a 32B judge fits next to it.
- Do you have JEV API access and pricing, or is it Laya-only for now?
- Should the RAG / tool-retrieval addition (§10.1) go into the proposal revision?
