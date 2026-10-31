#!/usr/bin/env python
"""Recompute the paper's headline tables from the saved CSVs in results/.

No training and no PyTorch needed -- this only reads results/*.csv.

    python scripts/summarize_results.py

Prints (1) the four-method baseline comparison (paper Table 3) and (2) the
three-condition thermal-awareness ablation with the severe-hotspot metric and
the Welch t-test / Cohen's d / bootstrap CI (paper Table 4 and Section 5.2).
"""
import os

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
SEVERE_C = 40.0  # evaluation-only threshold; the training threshold is 32 C


def bootstrap_ci_diff(a, b, n_boot=10_000, seed=42):
    rng = np.random.default_rng(seed)
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = rng.choice(a, len(a), replace=True).mean() - rng.choice(b, len(b), replace=True).mean()
    return np.percentile(d, [2.5, 97.5])


def main():
    print("=== Baseline comparison (per-task metrics, 150 evaluation episodes) ===")
    print(pd.read_csv(os.path.join(RESULTS, "phase4_summary_table.csv")).round(3).to_string(index=False))

    names = {"A (no thermal awareness)": "thermal_ablation0_floor.csv",
             "B (penalized, blind)": "thermal_ablationA_reward_only.csv",
             "C (penalized, observable)": "thermal_ablationB_full_fix.csv"}
    dfs = {k: pd.read_csv(os.path.join(RESULTS, v)) for k, v in names.items()}

    rows = []
    for k, d in dfs.items():
        rows.append({"condition": k,
                     "mean_hotspot_penalty": d.mean_hotspot_penalty.mean(),
                     "severe_rate_first20_%": 100 * (d.max_rack_temp.head(20) > SEVERE_C).mean(),
                     "severe_rate_last20_%": 100 * (d.max_rack_temp.tail(20) > SEVERE_C).mean(),
                     "max_rack_temp_last20_C": d.max_rack_temp.tail(20).mean()})
    print("\n=== Thermal-awareness ablation ===")
    print(pd.DataFrame(rows).round(2).to_string(index=False))

    c = dfs["C (penalized, observable)"].max_rack_temp.tail(20).values
    print("\n=== Condition C vs. others (last 20 episodes, max rack temperature) ===")
    for k in ("A (no thermal awareness)", "B (penalized, blind)"):
        o = dfs[k].max_rack_temp.tail(20).values
        t, p = stats.ttest_ind(c, o, equal_var=False)
        diff = c.mean() - o.mean()
        d = diff / np.sqrt((c.std(ddof=1) ** 2 + o.std(ddof=1) ** 2) / 2)
        lo, hi = bootstrap_ci_diff(c, o)
        print(f"C vs {k:26s} diff={diff:6.2f} C  Welch t={t:7.2f}  p={p:.2e}  d={d:6.2f}  95% CI [{lo:.2f}, {hi:.2f}]")


if __name__ == "__main__":
    main()
