"""Prompted planner with retrieved in-context demonstrations (TaskBench).

For each request, the k most similar requests (TF-IDF cosine) are pulled from a demo pool
of the same TaskBench subset and shown with their gold plans before the real request.
The pool EXCLUDES every task that any eval slice can sample (first ``exclude_n`` tasks of
each seed in ``exclude_seeds``), so no eval case or its answer appears in a prompt.
Demos teach the dataset's conventions (tool choice, decomposition, ordering) that hand-written
rules cannot.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from tpj.planner.prompted import PromptedPlanner

_STOP = set("a an the and or to of for in on at by with from me my i you your it is are be can could "
            "would will please also then need want help this that some about as so".split())


def _toks(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 1]


class _Tfidf:
    """Minimal TF-IDF + cosine (stdlib only)."""

    def __init__(self, docs: list[str]):
        tok = [_toks(d) for d in docs]
        df = Counter(w for t in tok for w in set(t))
        n = len(docs)
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}
        self.vecs = [self._vec(t) for t in tok]

    def _vec(self, toks):
        tf = Counter(toks)
        v = {w: c * self.idf.get(w, 0.0) for w, c in tf.items()}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    def top(self, query: str, k: int) -> list[int]:
        q = self._vec(_toks(query))
        sims = [sum(q.get(w, 0.0) * x for w, x in d.items()) for d in self.vecs]
        return sorted(range(len(sims)), key=sims.__getitem__, reverse=True)[:k]


class RetrievalPlanner(PromptedPlanner):
    def __init__(self, llm, format, subset: str = "dailylifeapis", k: int = 3,
                 exclude_seeds: tuple = (0, 1), exclude_n: int = 500,
                 constrained: bool = True, max_parse_retries: int = 1):
        super().__init__(llm, format, few_shot=False, constrained=constrained,
                         max_parse_retries=max_parse_retries)
        from tpj.benchmarks.taskbench import TaskBenchBenchmark

        held = set()
        for seed in exclude_seeds:
            held |= {t.id for t in TaskBenchBenchmark(subset=subset, limit=exclude_n, sample_seed=seed).load()}
        all_tasks = TaskBenchBenchmark(subset=subset, limit=None, sample_seed=None).load()
        self.pool = [t for t in all_tasks if t.id not in held and t.gold_plan is not None]
        self.k = k
        self._index = _Tfidf([t.query for t in self.pool])

    def _demos(self, query: str):
        return [self.pool[i] for i in self._index.top(query, self.k)]

    def build_messages(self, query, tools, feedback=None):
        msgs = super().build_messages(query, tools, feedback)  # [system, ..., final user]
        demos = []
        for d in self._demos(query):
            demos.append({"role": "user", "content": f"Request: {d.query}\n\n(tools: same list as in the final request)"})
            demos.append({"role": "assistant", "content": self.format.render(d.gold_plan)})
        return [msgs[0]] + demos + msgs[1:]
