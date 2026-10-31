# Architecture

## 1. Data flow

```
PVGIS (10 cities) ──► src/solar_data.py ──► solar profile per data center ─┐
Alibaba PAI trace ──► src/workflow_generator.py ──► 1000 DAG workflows ────┤
                                                                            ▼
                     src/datacenter_env.py   (servers, power, time, energy, federation)
                                  │
                                  ▼
          src/scheduler_env.py   event-driven loop: a task becomes "ready" when all predecessors finish
                 │                      │
                 │ global agent picks a data center      local agent (of that data center) picks a server
                 ▼                                                  ▼
          src/agents.py  (actor / critic networks)         reward -> online actor-critic update
                 │
                 ├── src/baselines.py             Random, Green-Opt (no learning)
                 └── src/thermal_*.py             rack temperature layer (optional)
```

## 2. The two-level agent hierarchy
- **Global agent** -- observes, per data center, green-energy surplus/deficit, average processing speed and utilization (3 x 10 values) plus the task's CPU and memory demand: **state dim 32**. Action: one of 10 data centers.
- **Local agent** (one per data center; one shared set of weights) -- observes, per server, processing speed and load (2 x 25 values) plus the task's CPU/memory demand: **state dim 52**. Action: one of 25 servers.
- **Local-neighborhood critics (the paper's key idea):** the local critic sees only its own state and action; the global critic sees its own state/action plus those of the *single* local agent that received the task -- not all ten. This keeps critic input size constant as the federation grows.
- **Common-Actor baseline:** same actors, but one common critic is used for all agents (no neighborhood restriction).

## 3. Reward (paper Eq. 13-16)
- Local reward: `-(alpha * T_norm + (1 - alpha) * E_norm)` with normalized execution time `T_norm` and energy `E_norm` (alpha = 0.5), so higher is better.
- Global reward: the normalized green-energy surplus/deficit term of the chosen data center plus the local reward of the agent that executed the task.
- Updates are one-step (gamma = 0): each dispatch is its own decision epoch.

## 4. Implementation adaptations vs. the paper (all documented in the module docstrings)
- PyTorch instead of Keras.
- Server load is a continuous congestion value rather than a binary busy/idle flag, because the binary encoding let the policy repeatedly choose the same "idle" server, causing runaway queueing and infinite energy values.
- Real PVGIS profiles per site, Alibaba-derived task sizes, event-driven simulation clock.

## 5. Thermal extension (`src/thermal_model.py`, `src/thermal_scheduler_env.py`)
- Servers of a data center are grouped into racks of 5. Each rack has a temperature that rises with the power of its servers (heat gain 0.006 C per watt-second) and relaxes toward ambient (22 C) at cooling rate 0.04 per second -- a lumped-capacitance model.
- **Hotspot penalty:** a softplus of (rack temperature - 32 C) with sharpness 3, multiplied by `thermal_weight` and *subtracted* from the local reward.
- **Observability:** in condition C the local state gets one normalized temperature per server (52 -> 77) and the global state one mean rack temperature per data center (32 -> 42), so networks are re-initialized for that condition.
- Migration cost exists in `thermal_model.py` but is fixed at zero weight: re-assigning a running task is not in the action space.

## 6. The three ablation conditions
| Article | Notebook | Reward penalty | State |
|---|---|---|---|
| A | Ablation 0 (floor) | none (weight 0) | original |
| B | Ablation A (reward-only) | weight 0.5 | original (blind) |
| C | Ablation B (full fix) | weight 0.5 | + temperatures |

All three use identical schedules, seeds and hyper-parameters, so differences come only from the thermal mechanism.

## 7. Metrics
Per-task energy and execution time (episodes contain different numbers of tasks, so raw totals would be unfair), mean global reward, mean wait time, mean hotspot penalty, maximum rack temperature per episode, and the **severe-hotspot rate** (share of episodes whose maximum rack temperature exceeds 40 C, an evaluation-only threshold above the 32 C training threshold).
