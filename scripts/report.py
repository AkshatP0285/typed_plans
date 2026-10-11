"""Summarise runs:  python scripts/report.py multirun/2026-10-08/12-00-00"""
from __future__ import annotations

import sys

from tpj.eval.report import collect, to_markdown

if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "outputs"
    print(to_markdown(collect(root)))
