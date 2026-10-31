# results/

Training histories (CSV), summary tables (CSV) and trained checkpoints (PyTorch `.pt`) from the runs reported in the article.
File-to-experiment mapping below is taken from the notebook and the file names; Phase 5 files are exploratory.

## Baseline comparison (Phase 3 + 4)
| File | Content |
|---|---|
| `baseline_fixed_alpha.pt`, `baseline_fixed_alpha_history.csv` | Proposed hierarchical agent (fixed alpha = 0.5), 150 training episodes |
| `common_actor.pt`, `common_actor_history.csv` | Common-Actor baseline (one shared critic) |
| `random_history.csv`, `greenopt_history.csv`, `proposed_eval_history.csv` | Per-episode evaluation of Random, Green-Opt, Proposed |
| `phase4_summary_table.csv` | **Article Table 3** (per-task energy, execution time, reward, wait time) |

## Thermal-awareness ablation (Phase 4.5)
| File | Article condition |
|---|---|
| `thermal_ablation0_floor.csv` | **A** -- no thermal awareness (no penalty, blind state) |
| `thermal_ablationA_reward_only.csv` | **B** -- penalty in reward, blind state |
| `thermal_ablationB_full_fix.csv`, `thermal_ablationB_checkpoint.pt` | **C** -- penalty + observable rack temperature |
| `thermal_aware_baseline.pt`, `thermal_baseline_history.csv`, `*_thermal_history.csv`, `proposed_thermal_eval_history.csv`, `phase4_5_thermal_summary_table.csv` | Thermal-tracked side-by-side of Random / Green-Opt / Proposed (temperature measured, thermal mechanism of the ablation not enabled in Random/Green-Opt) |

Naming note: the notebook calls the ablations "0 / A / B"; the article calls them "A / B / C".
`python scripts/summarize_results.py` recomputes Table 4 and the significance tests from these CSVs.

## Exploratory: dynamic alpha (Phase 5, not part of the article)
`dynamic_alpha_5a*.pt|csv`, `phase5a_*.csv`, `fixed_alpha_eval_with_deadlines.csv`, `dynamic_alpha_eval_history.csv` (5a: deadline-urgency-based alpha, several variants/seeds) and `learned_alpha_5b_BEST.pt`, `5b_train_history.csv`, `phase5b_final_comparison.csv` (5b: alpha as a learned continuous actor output). See the main README for the honest summary.
