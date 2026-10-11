# Code walkthrough

A guide for reading this codebase for the first time. The [README](../README.md) says *what* the system does and how to run it. This file explains *how the code is put together*, including the few mechanisms that are not obvious from reading one file at a time.

## Reading order

If you only have an hour, read these in order:

1. [ir/schema.py](../src/tpj/ir/schema.py) and [ir/examples.py](../src/tpj/ir/examples.py): what a plan looks like (`Plan`, `Step`, `$ref`).
2. [agent/verified.py](../src/tpj/agent/verified.py) `VerifiedAgent.run`: the top-level loop, about 15 lines.
3. [agent/run.py](../src/tpj/agent/run.py) `PlanRun`: everything that happens to one task. This is the heart of the project.
4. [gate/base.py](../src/tpj/gate/base.py), [judge/llm_judge.py](../src/tpj/judge/llm_judge.py), [repair/strategies.py](../src/tpj/repair/strategies.py): the three things `PlanRun` calls when it verifies and repairs.
5. [runner.py](../src/tpj/runner.py) and [eval/metrics.py](../src/tpj/eval/metrics.py): how a whole benchmark is run and scored.

Everything else is a pluggable variant of one of these pieces: other gates, other benchmarks, other plan formats.

## Four mechanisms that are easy to miss

### 1. Objects are built from YAML (Hydra `_target_`)

No file calls `VerifiedAgent(...)` directly. [scripts/run_agent.py](../scripts/run_agent.py) loads [configs/config.yaml](../configs/config.yaml), and [runner.py](../src/tpj/runner.py) calls `hydra.utils.instantiate(cfg.agent)`. Hydra reads `_target_` and calls that class with the remaining keys as keyword arguments:

```yaml
# configs/agent/verified.yaml
_target_: tpj.agent.VerifiedAgent
planner: ${planner}        # <- itself built from configs/planner/<choice>.yaml
gate: ${gate}
max_repairs: 3
```

```yaml
# configs/planner/prompted.yaml
_target_: tpj.planner.PromptedPlanner
llm: ${models.planner}     # <- an LLMClient built from configs/model/qwen3_8b_vllm.yaml
format: ${plan_format}
```

So `VerifiedAgent.__init__` receives a fully built planner, gate, judge, and so on. Consequences:

- **To find which class runs**, look at the YAML for the chosen option, not at the Python imports. `gate=jev` on the command line means `configs/gate/jev.yaml`, which means `tpj.gate.jev.JEVGate`.
- **Constructor parameter names are part of the config.** Renaming `max_repairs` in Python breaks every YAML that sets it.
- An experiment such as `+experiment=rq2_gating` is just a list of config choices to sweep over.

### 2. `emit(...)` writes to a tracer you never pass around

Almost every component calls `emit("some_event", key=value)` from [eval/trace.py](../src/tpj/eval/trace.py). There is no tracer argument because the current tracer lives in a [context variable](https://docs.python.org/3/library/contextvars.html). [runner.py](../src/tpj/runner.py) opens `with tracing(task.id) as tracer:` around each task, and every `emit` inside that task, including inside concurrently running `asyncio` sub-tasks, lands in `tracer.events`. Outside a `tracing` block, `emit` does nothing.

**All metrics are computed from these events** ([eval/metrics.py](../src/tpj/eval/metrics.py)), not from return values. So event names and fields such as `"node_result"`, `flagged` and `fault` are effectively an API. Changing them changes the metrics.

### 3. `async` / `await` and `asyncio.gather`

LLM calls, gate calls and tool calls are I/O, so everything is `async`. Two patterns matter:

- `await asyncio.gather(*[f(x) for x in items])` runs all `f(x)` concurrently and returns their results in order. It is used to run a wave's tool calls together, to gate a wave's outputs together, and to run many tasks together.
- `asyncio.Semaphore(n)` limits how many run at once (`task_concurrency`, `max_concurrency`).

### 4. Ablations are "null" components, not `if` statements

To switch a feature off, the config swaps in a component that does nothing: `NullJudge`, `NullEditor`, `NoCache`, `NoRepair`, `ConstantGate(label=OKAY)`, `SequentialScheduler`. Each has the same methods as the real one, so `PlanRun` has no special cases. Classes like `Gate`, `Planner`, `RepairStrategy` and `Benchmark` are abstract base classes (`ABC`) that define those shared methods.

## Following one task through the code

```
runner.evaluate
└─ for each task (concurrently), inside tracing(task.id):
   └─ VerifiedAgent.run(task, registry)
      ├─ PlanRun.prepare()                                       BEFORE EXECUTION
      │   ├─ planner.plan(task)              -> Plan
      │   ├─ faults.corrupt_plan(...)        maybe inject a plan fault (ground truth for the gate)
      │   └─ up to max_plan_edits rounds:
      │        check_structure(plan)          dangling $ref / cycle?
      │        gate.classify(phase="plan")    flagged?
      │        ir_editor.revise(...)          if either says so, rewrite the plan
      │
      ├─ while not run.done:                                     EXECUTION LOOP
      │   ├─ run.next_wave()                 ready steps -> list[PendingCall]
      │   │     settles steps that need no tool call: upstream failed,
      │   │     `when` false (skipped), bad $ref, or cache hit
      │   ├─ VerifiedAgent._execute(...)     run each call (maybe with an injected fault)
      │   └─ run.observe(wave, outcomes)
      │         ├─ _record_outcomes          store outputs in ExecutionState
      │         ├─ _gate_outputs             gate.classify(phase="post") per successful call
      │         ├─ _collect_flags            tool errors + gate flags -> list[Flag]
      │         ├─ _judge_and_repair         judge.review -> repair.apply -> new plan,
      │         │                            "dirty" steps reset to pending
      │         └─ _cache_outputs            cache outputs that passed
      │
      └─ run.result()                        -> AgentResult (completed / failed)
   bench.score(task, result)                 benchmark-specific success
   task_metrics(tracer.events, score)        one metrics row
aggregate(rows) -> metrics.json
```

### Why repairs are cheap: dirty regions and the cache

When the judge patches step `s5`, only `s5` and the steps that depend on it (its *descendants*, see `ir/dag.py::dirty_region`) are reset to `pending`. Everything else keeps its output. Steps that do re-run look in the cache first. The cache key is *tool name + fully resolved arguments* ([cache/node_cache.py](../src/tpj/cache/node_cache.py)), so a downstream step whose inputs did not change gets a cache hit instead of a second tool call.

## Glossary

| Term | Meaning |
|---|---|
| **IR / plan** | The typed plan: a list of `Step`s plus `final` (the step ids whose outputs answer the query). |
| **step / node** | The same thing. "Node" is used when thinking of the plan as a dependency graph, e.g. `NodeState`. |
| **`$ref`** | An argument value `{"$ref": "s1", "path": "first.id"}` meaning "the output of step s1, then `.first.id`". This is the *only* way to create a dependency ([ir/refs.py](../src/tpj/ir/refs.py)). |
| **wave** | A set of steps whose dependencies are all finished, executed together. |
| **gate** | Cheap classifier (Laya, JEV, ...) that says OKAY / ERROR / NEEDS_FORMAT_CHANGE. It is called "plan" phase before execution and "post" phase after a step runs. |
| **flagged** | The gate's `1 - P(OKAY) >= threshold`, or the tool raised an error. Only flagged things reach an LLM. |
| **IR editor** | LLM that rewrites the whole plan before execution when the gate flags it. |
| **judge** | LLM that, after execution, looks at flagged steps and returns a `JudgeDecision` (repair / retry / false_alarm / abort, plus patch ops). |
| **patch op** | `edit`, `replace`, `remove` or `add` on one step ([repair/patch.py](../src/tpj/repair/patch.py)). |
| **dirty** | Steps that must re-run after a repair. |
| **fault** | A deliberately injected error ([faults/injector.py](../src/tpj/faults/injector.py)). Because we know where faults are, we can measure gate precision and recall. |
| **trace event** | A dict recorded by `emit(...)`. One run's events go to `trace.jsonl`. |

## How to add …

- **a gate**: subclass `Gate` in `src/tpj/gate/`, implement `async _classify(self, inp) -> {label: score}`, and add `configs/gate/<name>.yaml` with `_target_` pointing to it. `Gate.classify` handles normalising, thresholding and tracing.
- **a benchmark**: subclass `Benchmark` (`load`, `registry`, `score`) in `src/tpj/benchmarks/`, then add `configs/benchmark/<name>.yaml`. [benchmarks/toy.py](../src/tpj/benchmarks/toy.py) is the smallest example.
- **a metric**: if the information is not in the trace yet, `emit` it where it happens. Then read it in `task_metrics` and summarise it in `aggregate`.

## Tests

`pytest` runs without any model. [tests/test_loop.py](../tests/test_loop.py) is the best executable documentation of the whole loop: it builds a `VerifiedAgent` by hand (no Hydra) with a gold planner, an oracle gate and scripted judges, and runs the toy benchmark.
