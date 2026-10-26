"""Scores experiment outputs and prints a side-by-side comparison.

Positive class = the task's "problem found" label (INVALID for plan verification).
By default models are compared on the cases ALL selected experiments scored (paired), so a
model that was only partly run is not compared against one that saw the full set."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .config import Experiment
from .tasks import TASKS


def load_preds(path: Path) -> pd.DataFrame:
    """Successful rows only; if a case was retried, the last successful row wins."""
    if not Path(path).exists():
        return pd.DataFrame()
    rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df = df[(df["error"].fillna("") == "") & (df["prediction"].fillna("") != "")]
    return df.drop_duplicates("case_id", keep="last").reset_index(drop=True)


def _auroc(y, s):
    try:
        from sklearn.metrics import roc_auc_score
        if y.nunique() < 2 or s.nunique() < 2:
            return float("nan")
        return float(roc_auc_score(y, s))
    except Exception:
        return float("nan")


def meaning_only(df: pd.DataFrame, positive: str) -> pd.DataFrame:
    """Drop edited plans the rule-based validator can catch (det_flagged), keeping the valid plans and
    the edits only a reader of the query can notice (remove, replace_arg, add, rule-invisible tool swaps)."""
    return df[(df["label"] != positive) | ~df["det_flagged"].astype(bool)].reset_index(drop=True)


def metrics(df: pd.DataFrame, positive: str) -> dict:
    neg_df = df[df["label"] != positive]
    pos_df = df[df["label"] == positive]
    pred_pos = df["prediction"] == positive
    tp = int(pred_pos[pos_df.index].sum())
    fp = int(pred_pos[neg_df.index].sum())
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / len(pos_df) if len(pos_df) else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if tp else float("nan")
    s = df["probabilities"].apply(lambda p: (p or {}).get(positive))
    y = (df["label"] == positive).astype(int)
    ok = s.notna()
    sem = (df["label"] != positive) | ~df["det_flagged"]      # GT + edits a rule validator misses
    tn = len(neg_df) - fp
    return {
        "n": len(df), "n_edited": len(pos_df), "n_valid": len(neg_df),
        "caught": tp, "missed": len(pos_df) - tp, "false_alarms": fp, "correct_valid": tn,
        "correct_total": tp + tn,           # edited said wrong + valid said right
        "detect": rec, "false_alarm": fp / len(neg_df) if len(neg_df) else float("nan"),
        "precision": prec, "f1": f1,
        "auroc": _auroc(y[ok], s[ok]),
        "auroc_semantic": _auroc(y[ok & sem], s[ok & sem]),
        "latency_s": float(df["latency_seconds"].mean()),
    }


def compare(exps: list[Experiment], paired: bool = True, out_dir: Path | None = None,
            meaning: bool = False) -> pd.DataFrame:
    positive0 = TASKS[exps[0].task].positive
    data = {e.name: load_preds(e.out) for e in exps}
    data = {k: (meaning_only(v, positive0) if meaning and not v.empty else v) for k, v in data.items()}
    data = {k: v for k, v in data.items() if not v.empty}
    if not data:
        raise SystemExit("no scored results yet - run some experiments first")
    if paired and len(data) > 1:
        common = set.intersection(*[set(v["case_id"]) for v in data.values()])
        data = {k: v[v["case_id"].isin(common)].reset_index(drop=True) for k, v in data.items()}
        print(f"paired comparison on {len(common)} cases scored by every selected experiment\n")
    positive = TASKS[exps[0].task].positive
    rows = []
    for e in exps:
        if e.name not in data:
            continue
        m = metrics(data[e.name], positive)
        rows.append({"experiment": e.name, "backend": e.backend, "prompt": e.prompt.name, **m})
    tab = pd.DataFrame(rows)
    show = pd.DataFrame({
        "experiment": tab["experiment"], "prompt": tab["prompt"], "n": tab["n"],
        "edits_caught": [f"{r.caught}/{r.n_edited} ({r.caught / r.n_edited:.1%})" for r in tab.itertuples()],
        "false_alarms": [f"{r.false_alarms}/{r.n_valid} ({r.false_alarms / r.n_valid:.1%})"
                         for r in tab.itertuples()],
        "total_correct": [f"{r.correct_total}/{r.n} ({r.correct_total / r.n:.1%})" for r in tab.itertuples()],
    })
    for c in ("precision", "f1", "auroc", "auroc_semantic"):
        show[c] = tab[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:.3f}")
    show["latency_s"] = tab["latency_s"].map(lambda v: "n/a" if pd.isna(v) else f"{v:.2f}")
    print(show.to_string(index=False))
    print(f"\n(positive class = {positive}. edits_caught = edited plans correctly said INVALID; "
          f"false_alarms = valid plans wrongly said INVALID; total_correct = edited said INVALID + valid "
          f"said VALID, out of all cases; auroc_semantic ignores edits a rule-based validator catches)")
    if out_dir:
        out_dir.mkdir(exist_ok=True)
        suffix = "_meaning_only" if meaning else ""
        tab.to_csv(out_dir / f"comparison{suffix}.csv", index=False)
        (out_dir / f"comparison{suffix}.md").write_text(show.to_markdown(index=False)
                                                 if _has_tabulate() else show.to_string(index=False),
                                                 encoding="utf-8")
        print(f"saved {out_dir / ('comparison' + suffix + '.csv')}")
    return tab


def _has_tabulate() -> bool:
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


def detail(exp: Experiment, restrict_to: set[str] | None = None) -> None:
    """Per edit-type detection for one experiment, next to the rule-based validator."""
    df = load_preds(exp.out)
    if restrict_to is not None:
        df = df[df["case_id"].isin(restrict_to)]
    positive = TASKS[exp.task].positive
    bad = df[df["label"] == positive]
    rows = []
    for kind, g in bad.groupby("kind"):
        hit = int((g["prediction"] == positive).sum())
        rows.append({"edit": kind, "n": len(g), "detected": f"{hit}/{len(g)} ({100 * hit / len(g):.0f}%)",
                     "rule-based flags": f"{int(g['det_flagged'].sum())}/{len(g)}"})
    print(f"\n== {exp.name}: detection by edit type ==")
    print(pd.DataFrame(rows).to_string(index=False))
    for src, g in df.groupby("source"):
        v = g[g["label"] != positive]
        b = g[g["label"] == positive]
        print(f"  {src:10s} detect {int((b['prediction'] == positive).sum())}/{len(b)}   "
              f"false alarm {int((v['prediction'] == positive).sum())}/{len(v)}")


# --------------------------------------------------------------------------- repeats / bootstrap
def _counts(df: pd.DataFrame, positive: str) -> dict:
    pos, pred = df["label"] == positive, df["prediction"] == positive
    return {"tp": int((pos & pred).sum()), "n_pos": int(pos.sum()),
            "fp": int((~pos & pred).sum()), "n_neg": int((~pos).sum())}


def _rates(tp, n_pos, fp, n_neg):
    f1 = 2 * tp / (2 * tp + fp + (n_pos - tp)) if tp else float("nan")
    return {"detect": tp / n_pos, "false_alarm": fp / n_neg, "f1": f1}


def repeat_summary(exps: list[Experiment], repeats: int, bootstrap: int, out_dir: Path,
                   meaning: bool = False) -> None:
    """mean +- std over repeats (sample std, ddof=1) and 95% cluster-bootstrap CIs over base plans.

    Repeats only capture API nondeterminism. The bootstrap captures which plans happened to be in
    the test set; both are reported because they answer different questions."""
    import numpy as np
    from .runner import rep_path
    positive = TASKS[exps[0].task].positive
    rows, first = [], {}
    for e in exps:
        reps = [load_preds(rep_path(e.out, k)) for k in range(1, repeats + 1)]
        reps = [meaning_only(r, positive) if meaning and not r.empty else r for r in reps]
        reps = [r for r in reps if not r.empty]
        if not reps:
            continue
        common = set.intersection(*[set(r["case_id"]) for r in reps])
        reps = [r[r["case_id"].isin(common)].sort_values("case_id").reset_index(drop=True) for r in reps]
        per = [{**metrics(r, positive)} for r in reps]
        row = {"experiment": e.name, "repeats": len(reps), "n": len(common)}
        sd = lambda v: v.std(ddof=1) if len(v) > 1 else float("nan")
        for label, key, tot in (("edits_caught", "caught", per[0]["n_edited"]),
                                ("false_alarms", "false_alarms", per[0]["n_valid"]),
                                ("total_correct", "correct_total", per[0]["n"])):
            v = np.array([p[key] for p in per], float)
            row[label] = f"{v.mean():.1f} +- {sd(v):.1f} / {tot}"
        for m in ("precision", "f1", "auroc"):
            v = np.array([p[m] for p in per], float)
            row[m] = f"{v.mean():.3f} +- {sd(v):.3f}"
        preds = np.array([(r["prediction"] == positive).to_numpy() for r in reps])
        row["cases_with_changing_verdict"] = int((preds.min(0) != preds.max(0)).sum())
        rows.append(row)
        if bootstrap:
            df = reps[0]                       # repeat 1
            agg = {}
            for base, g in df.groupby("base_id"):
                agg[base] = _counts(g, positive)
            first.setdefault("bases", sorted(agg))
            first.setdefault("data", {})[e.name] = agg
    print(pd.DataFrame(rows).to_string(index=False))
    print("\n(mean +- sample std over repeats of the SAME inputs: API nondeterminism only)")
    if bootstrap and first:
        rng = np.random.default_rng(0)
        bases = [b for b in first["bases"] if all(b in a for a in first["data"].values())]
        idx = rng.integers(0, len(bases), size=(bootstrap, len(bases)))
        keys = ("tp", "n_pos", "fp", "n_neg")
        arr = {n: np.array([[a[b][k] for k in keys] for b in bases]) for n, a in first["data"].items()}

        def boot(n):
            s = arr[n][idx].sum(1)             # (B, 4)
            return {"detect": s[:, 0] / s[:, 1], "false_alarm": s[:, 2] / s[:, 3],
                    "f1": 2 * s[:, 0] / np.maximum(2 * s[:, 0] + s[:, 2] + (s[:, 1] - s[:, 0]), 1)}
        B = {n: boot(n) for n in arr}
        ci = lambda x: f"[{np.percentile(x, 2.5):.3f}, {np.percentile(x, 97.5):.3f}]"
        print(f"\n95% bootstrap CIs over {len(bases)} base plans ({bootstrap} resamples, repeat 1)")
        base_name = exps[0].name
        out = []
        for n in B:
            r = {"experiment": n, **{m: ci(B[n][m]) for m in ("detect", "false_alarm", "f1")}}
            if n != base_name and base_name in B:
                d = B[n]["false_alarm"] - B[base_name]["false_alarm"]
                r[f"false_alarm - {base_name}"] = f"{np.mean(d):+.3f} {ci(d)}"
                d = B[n]["detect"] - B[base_name]["detect"]
                r[f"detect - {base_name}"] = f"{np.mean(d):+.3f} {ci(d)}"
            out.append(r)
        print(pd.DataFrame(out).to_string(index=False))
        print("(a difference whose interval excludes 0 is a real change on this test set)")
