#!/usr/bin/env python
"""Generate data/workflows_1000.pkl (Phase 2) from the command line.

Usage (from the repository root):
    python scripts/generate_workflows.py                 # defaults: 1000 workflows, seed 42
    python scripts/generate_workflows.py --n 200 --seed 7
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.workflow_generator import build_and_save  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data", help="folder containing pai_task_table.csv (optional)")
    p.add_argument("--n", type=int, default=1000, help="number of workflows")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="workflows_1000.pkl", help="output file name inside --data-dir")
    a = p.parse_args()
    path, source, wfs = build_and_save(a.data_dir, a.n, a.seed, a.out)
    n_tasks = sum(w["dag"].number_of_nodes() for w in wfs)
    print(f"Saved {len(wfs)} workflows ({n_tasks} tasks) to {path}")
    print(f"Task-size source: {source}")
    if source != "alibaba_pai_trace":
        print("NOTE: pai_task_table.csv was not found in --data-dir; the log-normal fallback was used.\n"
              "      Place the Alibaba PAI trace in data/ (see data/README.md) to use real task sizes.")


if __name__ == "__main__":
    main()
