# Renewable- and Thermal-Aware Hierarchical MARL Workflow Scheduler

Intelligent cloud resource management for **energy optimization in geo-distributed data centers from a renewable-energy perspective**.
This repository reproduces and extends the hierarchical multi-agent actor-critic scheduler and adds
(1) real site-specific solar data, (2) workloads grounded in a real cluster trace, and
(3) a rack-level **thermal-awareness** extension with a controlled ablation.

*Author:* Mohammad Sepehr Ansari -- *Supervisor:* Dr. Mohammad Reza Rasouli (Master's research project)

---

## 1. The problem in one paragraph

Cloud workflows are DAGs of dependent tasks. A provider with several data centers powered partly by intermittent solar energy must decide, for every task, **which data center** runs it (where is green energy available right now?) and **which server** inside it (balance speed, energy, queueing) -- while avoiding rack **hotspots** that raise cooling cost. Centralized, fixed-rule schedulers scale and adapt poorly, so the scheduler is learned with **multi-agent reinforcement learning (MARL)**: one *global* agent picks the data center, one *local* agent per data center picks the server, trained with centralized critics that only see a *local neighborhood* of information.

## 2. What was done (contributions)

| # | Contribution | Where |
|---|---|---|
| 1 | Reproduction of the hierarchical global/local actor-critic scheduler and its evaluation against **Random**, **Green-Opt** (greedy green-energy-first heuristic) and **Common-Actor** (shared-critic MARL) baselines | `src/scheduler_env.py`, `src/agents.py`, `src/baselines.py` |
| 2 | **Data realism:** hourly solar generation from PVGIS for 10 real cities (instead of one trace for all sites) and 1000 workflows with task sizes resampled from the Alibaba Cluster Trace GPU-v2020 | `src/solar_data.py`, `src/workflow_generator.py` |
| 3 | **Thermal extension:** lumped-capacitance rack heat model, softplus hotspot penalty in the reward, optional rack-temperature observability in the agents' state; 3-condition ablation | `src/thermal_model.py`, `src/thermal_scheduler_env.py` |
| 4 | *Exploratory (not in the article):* dynamic / learned trade-off coefficient alpha (Phase 5a/5b) | `src/dynamic_alpha.py`, `src/dynamic_alpha_5b.py` |

## 3. Key results

Federation: 10 data centers x 25 servers; 150 episodes x 20 workflows; per-task averages. (`python scripts/summarize_results.py` recomputes everything below from `results/`.)

**Baseline comparison**

| Method | Energy / task (J) | Exec. time / task (s) | Mean global reward | Wait time (s) |
|---|---:|---:|---:|---:|
| Random | 2809.5 | 4.71 | -0.941 | 0.036 |
| Green-Opt | 1819.7 | 3.72 | -0.548 | 0.000 |
| Common-Actor | 699.8 | 1.45 | -0.418 | 0.336 |
| **Proposed (hierarchical MARL)** | **364.8** | **0.73** | **-0.395** | 0.135 |

The proposed scheduler uses 87.0% / 79.9% / 47.9% less energy per task than Random / Green-Opt / Common-Actor.

**Thermal-awareness ablation** (severe hotspot = max rack temperature > 40 C)

| Condition | Penalty in reward | Rack temp in state | Severe-hotspot rate, first 20 -> last 20 episodes | Mean max rack temp, last 20 (C) |
|---|:--:|:--:|:--:|---:|
| A -- no thermal awareness | no | no | 100% -> 100% | 44.34 |
| B -- penalized, blind | yes | no | 95% -> 85% | 44.21 |
| **C -- penalized, observable** | yes | **yes** | 100% -> **0%** | **33.62** |

Condition C vs A: -10.73 C (Welch t = -16.07, p = 2.0e-13, Cohen's d = -5.08); C vs B: -10.59 C (t = -10.08, p = 2.9e-9, d = -3.19).
**Takeaway:** a thermal penalty only helps when the agent can *observe* the quantity being penalized.

**Honest caveats**
- The original paper's *faster-convergence* claim for the local-neighborhood critic was **not** reproduced: in this simulator Common-Actor's training reward rises at least as fast. The *final-policy-quality* advantage (table above) is reproduced.
- Mean hotspot penalty over the whole run is a misleading metric here (the 32 C threshold sits close to normal operating temperature), which is why the severe-hotspot rate is reported.
- Everything is simulated (no physical hardware); absolute temperatures/energies are for comparing methods, not predicting a real data center.
- Phase 5 (dynamic alpha) is exploratory: 5a did not beat fixed alpha (441.6 vs 365.2 J/task, deadline violations 18.9% vs 15.8%); 5b lowered deadline violations (12.9% vs 16.1%) at similar energy but is not part of the article's claims.

## 4. Repository layout

```
.
├── thesis_pipeline_phase0_to_4_5.ipynb   # the full experiment, phase by phase, with outputs
├── src/                                  # importable Python package
│   ├── datacenter_env.py                 #   Phase 1: servers, power/time/energy models (paper Eq. 1-9)
│   ├── solar_data.py                     #   Phase 1.5: PVGIS solar profiles for 10 cities
│   ├── workflow_generator.py             #   Phase 2: Alibaba-based task sizes + synthetic DAGs + Poisson arrivals
│   ├── agents.py                         #   Phase 3: actor/critic networks (global, local, common critic)
│   ├── scheduler_env.py                  #   Phase 3: event-driven scheduling loop + online actor-critic training
│   ├── baselines.py                      #   Phase 4: Random and Green-Opt
│   ├── thermal_model.py                  #   Phase 4.5: rack heat model + hotspot penalty
│   ├── thermal_scheduler_env.py          #   Phase 4.5: thermal-aware episodes (3 ablation conditions)
│   └── dynamic_alpha.py / dynamic_alpha_5b.py  # Phase 5 (exploratory)
├── scripts/
│   ├── generate_workflows.py             # build data/workflows_1000.pkl from the command line
│   └── summarize_results.py              # recompute the article's tables + statistics from results/
├── tests/test_smoke.py                   # fast tests (no training needed)
├── results/                              # CSV histories, summary tables, trained checkpoints (see results/README.md)
├── data/                                 # downloaded / generated inputs (see data/README.md)
├── docs/ARCHITECTURE.md                  # how the system works, in detail
├── requirements.txt  LICENSE  CITATION.cff
```

## 5. Installation

```bash
git clone <your-repo-url>
cd renewable-thermal-marl-scheduler
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Python 3.10+ (developed on 3.11). A GPU is **not** required: each 150-episode training run took roughly 6-7 minutes on the author's machine (from the logged per-episode times).

## 6. How to use it

**A. Check the article's numbers (seconds, no PyTorch needed)**
```bash
python scripts/summarize_results.py
```

**B. Run the tests**
```bash
pytest -q
```

**C. Reproduce the experiments**
```bash
python scripts/generate_workflows.py           # creates data/workflows_1000.pkl  (optional: put pai_task_table.csv in data/ first)
jupyter notebook thesis_pipeline_phase0_to_4_5.ipynb   # launch from the repo root, run cells top to bottom
```
The notebook is organized by phase; each phase saves its outputs to `results/`:

| Phase | What happens | Output |
|---|---|---|
| 0 | environment setup | -- |
| 1 / 1.5 | simulation environment; real PVGIS solar data | `data/pvgis_cache/` |
| 2 | Alibaba-based workflow generator | `data/workflows_1000.pkl` |
| 3 | train the proposed hierarchical agent (150 episodes) | `results/baseline_fixed_alpha*` |
| 4 | Random, Green-Opt, Common-Actor and side-by-side comparison | `results/phase4_summary_table.csv` |
| 4.5 | thermal model, three-condition ablation, severe-hotspot metric, significance tests | `results/thermal_ablation*`, figures |

**D. Use the modules directly**
```python
from src.datacenter_env import build_federation
from src.solar_data import build_real_solar_profiles, DATACENTER_LOCATIONS
solar = build_real_solar_profiles(peak_watts_range=(800, 2500), seed=42)
federation = build_federation(real_solar_profiles=solar,
                              location_names=[n for n, _, _ in DATACENTER_LOCATIONS])
```

## 7. Main settings (as used in the article)

| Item | Value |
|---|---|
| Federation | 10 data centers x 25 servers, 5 hardware templates (SPECpower-derived) |
| Solar sites | Frankfurt, N. Virginia, Singapore, Sydney, Tokyo, Sao Paulo, Mumbai, Cape Town, London, Oregon (PVGIS) |
| Workload | 1000 layered DAGs, 10-30 tasks, Poisson arrivals; task sizes from Alibaba PAI trace |
| Training | 150 episodes x 20 workflows, seed 42, Adam lr 1e-3, alpha = 0.5, gamma = 0 |
| Networks | MLPs with two 128-unit hidden layers; 107,173 trainable parameters in total (4 networks) |
| State sizes | global 32 / local 52 (thermal-observable: 42 / 77) |
| Thermal model | 5 servers per rack, ambient 22 C, heat gain 0.006 C/(W.s), cooling 0.04 /s, penalty threshold 32 C (softplus sharpness 3), thermal weight 0 or 0.5 |

## 8. Reproducibility notes
- All random generators are seeded (seed 42 for the reported runs). PyTorch results can still differ slightly across versions/hardware.
- Reported results used the real Alibaba trace; without `pai_task_table.csv` the generator falls back to a log-normal size distribution (it tells you when it does).
- PVGIS is queried once per city and cached; offline runs fall back to a synthetic solar curve with a warning.
- The notebook assumes it is launched from the repository root (it imports `src/` and reads/writes `data/` and `results/` relatively).

## 9. Citing
If you use this code, please cite this repository (`CITATION.cff`) and the original framework:
A. Jayanetti, S. Halgamuge, R. Buyya, "Multi-Agent Deep Reinforcement Learning Framework for Renewable Energy-Aware Workflow Scheduling on Distributed Cloud Data Centers," *IEEE Trans. Parallel Distrib. Syst.*, 35(4), 604-615, 2024.

## 10. License
MIT -- see `LICENSE`. The Alibaba trace and PVGIS data are subject to their own terms.
