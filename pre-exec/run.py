"""CLI for the decision-model comparison harness. Edit experiments.toml, then:

  python run.py list                          show experiments, backends, status
  python run.py run                           run every enabled experiment (resumable)
  python run.py run --only jev_or_zeroshot    run specific ones (works even if enabled=false)
  python run.py run --limit 10 --dry-run      smoke test / preview without calling any model
  python run.py run --like jev_or_zeroshot    only the cases that experiment scored (paired probe)
  python run.py report                        paired comparison table of enabled experiments
  python run.py report --only a,b --detail a  per-edit-type breakdown for one experiment
  python run.py report --meaning-only         only valid plans + edits a rule checker cannot catch
"""
from __future__ import annotations

import argparse
from pathlib import Path

from harness.backends import BACKENDS
from harness.config import load_config, select
from harness.report import compare, detail, load_preds, repeat_summary
from harness.runner import rep_experiment, run_experiment

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(HERE / "experiments.toml"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    r = sub.add_parser("run")
    r.add_argument("--only")
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--like", help="only run cases already scored by this experiment (paired probe)")
    r.add_argument("--rep", type=int, default=1,
                   help="repeat number; 1 writes the normal results file, K>1 writes <name>.repK.jsonl")
    p = sub.add_parser("report")
    p.add_argument("--only")
    p.add_argument("--all-cases", action="store_true", help="do not restrict to the shared case set")
    p.add_argument("--detail", help="experiment name for a per-edit-type breakdown")
    p.add_argument("--meaning-only", action="store_true",
                   help="drop edits a rule-based validator can catch; keep valid plans + meaning-type edits")
    p.add_argument("--repeats", type=int, default=0,
                   help="N>1: summarise N repeats per experiment as mean +- std (rep files from run --rep)")
    p.add_argument("--bootstrap", type=int, default=0,
                   help="B>0: add 95%% bootstrap CIs over test cases (resampled by base plan)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.cmd == "list":
        print(f"backends: {sorted(BACKENDS)}\n")
        for e in cfg.experiments:
            n = len(load_preds(e.out))
            print(f"{'on ' if e.enabled else 'off'} {e.name:24s} {e.backend:6s} prompt={e.prompt.name:10s} "
                  f"scored={n:4d}  {e.params}")
    elif args.cmd == "run":
        only_ids = None
        if args.like:
            ref = next((e for e in cfg.experiments if e.name == args.like), None)
            if ref is None:
                raise SystemExit(f"--like: unknown experiment {args.like!r}")
            only_ids = set(load_preds(ref.out)["case_id"])
        for e in select(cfg, args.only):
            run_experiment(rep_experiment(e, args.rep), limit=args.limit, dry_run=args.dry_run,
                           only_ids=only_ids)
    elif args.cmd == "report":
        exps = select(cfg, args.only)
        if args.repeats > 1 or args.bootstrap:
            repeat_summary(exps, max(args.repeats, 1), args.bootstrap, cfg.root / "results",
                           meaning=args.meaning_only)
            return
        compare(exps, paired=not args.all_cases, out_dir=cfg.root / "results", meaning=args.meaning_only)
        if args.detail:
            d = next(e for e in exps if e.name == args.detail)
            common = None
            if not args.all_cases and len(exps) > 1:
                sets = [set(load_preds(e.out)["case_id"]) for e in exps if not load_preds(e.out).empty]
                common = set.intersection(*sets)
            detail(d, common)


if __name__ == "__main__":
    main()
