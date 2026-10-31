"""
Phase 4.5 (new) -- Thermal-aware scheduling episode
=====================================================
Wraps the REAL run_scheduling_episode / run_common_actor_episode /
heuristic-episode loops from scheduler_env.py and baselines.py with rack-level
thermal tracking (thermal_model.py). Nothing in scheduler_env.py, agents.py,
baselines.py, or datacenter_env.py is modified -- this file only imports and
composes them, so your existing Phase 3/4 results (checkpoints, plots,
metrics) stay exactly reproducible untouched.

Scope, stated honestly (documented here so it's clear in the Methods section
of the paper too): this integration adds the HOTSPOT-PENALTY term to the
reward -- it does NOT add a migration action. Migration requires extending
the local actor's action space (server selection -> server selection OR
migrate-existing-task), which is a network-architecture change, not a
reward-shaping change, and is out of scope for "test the results on Phase
0-4 as-is." migration_cost() from thermal_model.py is implemented and unit
tested (see test_thermal.py) and ready to be wired in once that action-space
change is made -- but it is NOT active in the reward computed below, and no
migration decisions are made by any agent in this file. Only the hotspot
penalty (a function of where the SCHEDULER already decided to place the
task) is active, since that requires zero changes to what actions are
available -- it only changes how those choices are scored.

Everything below mirrors the real loops line-for-line except for:
1. building a ThermalState + rack assignment once per episode
2. calling thermal.step(...) after every task assignment
3. computing hotspot_penalty() on the rack the task just landed in and
   subtracting it from reward_local/reward_global before it's used for
   either logging or the online actor-critic update
4. logging "hotspot_penalty" and "rack_temp" as extra per-task metrics, so
   the notebook can plot thermal behavior separately from energy/time --
   this is what lets the results section show whether hotspot FREQUENCY
   actually drops, not just whether the summed reward went up.
"""

import heapq
import numpy as np
import torch

from .datacenter_env import assign_task
from .scheduler_env import (
    global_state, local_state, _build_event_queue, _update_step,
    T_SCALE, E_SCALE, GREEN_DELTA_SCALE, N_DC, N_SERVERS_PER_DC,
    GLOBAL_STATE_DIM, LOCAL_STATE_DIM,
)
from .thermal_model import assign_racks, ThermalState, hotspot_penalty

# ---------------------------------------------------------------------------
# POST-FIRST-RUN UPDATE -- three things changed after seeing real results:
#
# 1. THE AGENT WAS THERMALLY BLIND. global_state/local_state (imported above,
#    UNCHANGED from scheduler_env.py) contain no temperature feature at all.
#    The reward could penalize a hotspot, but the policy network had no input
#    telling it WHICH choice caused the heat, so there was no way for
#    gradient descent to connect action -> consequence. This is why
#    mean_hotspot_penalty went UP over training instead of down in the first
#    run. Fix: global_state_thermal/local_state_thermal below append rack/
#    server temperature as an observed feature, so a genuinely thermal-aware
#    policy is at least possible. This changes state dimensionality, so it
#    needs its own actor/critic networks (GLOBAL_STATE_DIM_THERMAL/
#    LOCAL_STATE_DIM_THERMAL) -- a checkpoint trained on the blind state
#    space is NOT compatible with these.
#
# 2. THE WORKLOAD NEVER ACTUALLY CREATED HOTSPOTS at the original constants.
#    Real rack temperatures stayed in the 30-34 degC band the whole training
#    run, nowhere near the 45 degC threshold, so hotspot_penalty was
#    near-constant floor noise regardless of policy quality. thermal_model.py
#    is recalibrated (see the RECALIBRATION NOTE in that file) so the
#    threshold sits inside the actually-observed operating range.
#
# 3. THERE WAS NO CONTROL RUN, so "thermal-aware beats non-thermal-aware"
#    could not be claimed causally from the first run alone. Below, THREE
#    conditions are now run side by side, all under IDENTICAL thermal
#    instrumentation/logging so they are directly comparable:
#      - Ablation 0 ("floor"):  blind state,   thermal_weight=0    (no thermal awareness at all)
#      - Ablation A ("reward-only"): blind state,   thermal_weight>0   (penalized but can't see it)
#      - Ablation B ("full fix"):    thermal state, thermal_weight>0   (penalized AND can see it)
#    All three use run_scheduling_episode_thermal_v2 below (thermal_weight=0
#    reduces Ablation B to Ablation 0-with-visibility, so we additionally
#    keep the ORIGINAL run_scheduling_episode_thermal function -- now
#    labeled as the blind-state runner -- to produce true Ablation 0/A with
#    zero risk of the state features leaking a signal the reward doesn't
#    use.) The comparison across all three is what makes "the agent learned
#    to avoid hotspots because it could see them" a causal claim rather than
#    an assertion.
# ---------------------------------------------------------------------------

TEMP_SCALE = 40.0    # normalizes (temp - ambient) into a roughly [0,3] range,
AMBIENT_C = 22.0      # comparable in magnitude to the other state features


def _setup_thermal(federation, rack_size=5, **thermal_kwargs):
    """Call once per episode (fresh federation -> fresh racks + thermal state)."""
    for dc in federation:
        assign_racks(dc, rack_size=rack_size)
    thermal = ThermalState.initialize(federation, **thermal_kwargs)
    return thermal


# ---------------------------------------------------------------------------
# NEW: thermally-VISIBLE state functions (Fix #1)
# ---------------------------------------------------------------------------

def global_state_thermal(federation, task_size_mi, now, sim_hours, thermal):
    """Same as scheduler_env.global_state, PLUS one avg-rack-temperature
    feature per datacenter, so the global agent can route away from a
    datacenter whose racks are running hot."""
    base = global_state(federation, task_size_mi, now, sim_hours)
    dc_temps = []
    for dc in federation:
        rack_ids = sorted(set(getattr(s, "rack_id", 0) for s in dc.servers))
        avg_temp = float(np.mean([thermal.rack_temp(dc, rid) for rid in rack_ids]))
        dc_temps.append(np.clip((avg_temp - AMBIENT_C) / TEMP_SCALE, 0, 3))
    return torch.cat([base, torch.tensor(dc_temps, dtype=torch.float32)])


def local_state_thermal(dc, task_size_mi, now, thermal):
    """Same as scheduler_env.local_state, PLUS one per-server temperature
    feature (the server's OWN rack temperature), so the local agent can
    actually avoid the server that would land the task in a hot rack."""
    base = local_state(dc, task_size_mi, now)
    server_temps = [np.clip((thermal.server_temp(s) - AMBIENT_C) / TEMP_SCALE, 0, 3)
                    for s in dc.servers]
    return torch.cat([base, torch.tensor(server_temps, dtype=torch.float32)])


GLOBAL_STATE_DIM_THERMAL = GLOBAL_STATE_DIM + N_DC              # 32 + 10 = 42
LOCAL_STATE_DIM_THERMAL = LOCAL_STATE_DIM + N_SERVERS_PER_DC    # 52 + 25 = 77


def run_scheduling_episode_thermal(federation, workflows, global_actor, global_critic,
                                    local_actor, local_critic, optimizers=None,
                                    alpha=0.5, gamma=0.0, train=True, max_tasks=None,
                                    rack_size=5, thermal_weight=0.15,
                                    hotspot_threshold_c=None, thermal_kwargs=None):
    """BLIND thermal-reward variant (used for Ablation 0 with thermal_weight=0,
    and Ablation A with thermal_weight>0). Uses the ORIGINAL global_state/
    local_state -- no temperature feature anywhere in the policy's input --
    so the reward can penalize hotspots but the agent has no way to see what
    caused one. This is deliberately kept as its own function (rather than
    just running run_scheduling_episode_thermal_v2 with a flag) so there is
    zero risk of an accidental state-feature leak contaminating the "blind"
    condition. Returns the same metrics dict PLUS "hotspot_penalty" and
    "rack_temp" per-task lists.
    """
    thermal = _setup_thermal(federation, rack_size=rack_size, **(thermal_kwargs or {}))

    heap, counter = _build_event_queue(workflows)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows}
    task_finish_time = {}
    task_dc_assignment = {}

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": [],
               "hotspot_penalty": [], "rack_temp": []}

    n_processed = 0
    while heap:
        ready_time, _, wf_id, node = heapq.heappop(heap)
        if max_tasks is not None and n_processed >= max_tasks:
            break
        dag = dags[wf_id]
        task_size = dag.nodes[node]["size_mi"]
        now = ready_time
        sim_hours = now / 3600.0

        g_state = global_state(federation, task_size, now, sim_hours)
        g_action, g_logprob = global_actor.act(g_state)
        chosen_dc = federation[g_action]

        l_state = local_state(chosen_dc, task_size, now)
        l_action, l_logprob = local_actor.act(l_state)
        server = chosen_dc.servers[l_action]

        predecessor_info = []
        for pred in dag.predecessors(node):
            data_mb = dag[pred][node]["data_mb"]
            same_dc = task_dc_assignment.get(pred) == chosen_dc.id
            predecessor_info.append((data_mb, same_dc))

        class _T:
            pass
        task_obj = _T()
        task_obj.size_mi = task_size

        result = assign_task(task_obj, chosen_dc, server, now=now, sim_time_hours=sim_hours,
                              predecessor_finish_info=predecessor_info)

        task_finish_time[node] = result.finish_time
        task_dc_assignment[node] = chosen_dc.id

        # --- advance thermal state to `now`, then read the rack this task landed in ---
        thermal.step(federation, now=now, dt=max(1.0, result.exec_time))
        rack_id = getattr(server, "rack_id", 0)
        rack_temp = thermal.rack_temp(chosen_dc, rack_id)
        hpen = hotspot_penalty(rack_temp) if hotspot_threshold_c is None else hotspot_penalty(rack_temp, threshold_c=hotspot_threshold_c)

        # --- rewards: paper's base terms (Eq. 14, 16) minus the new thermal term ---
        t_norm = np.clip(result.exec_time / T_SCALE, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE, 0, 5)
        reward_local_base = -(alpha * t_norm + (1 - alpha) * e_norm)
        reward_local = reward_local_base - thermal_weight * hpen

        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)
        metrics["hotspot_penalty"].append(hpen)
        metrics["rack_temp"].append(rack_temp)

        if train and optimizers is not None:
            _update_step(optimizers, l_state, l_action, l_logprob, reward_local, local_actor, local_critic,
                         g_state, g_action, g_logprob, reward_global, global_actor, global_critic, gamma)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


def run_scheduling_episode_thermal_v2(federation, workflows, global_actor, global_critic,
                                       local_actor, local_critic, optimizers=None,
                                       alpha=0.5, gamma=0.0, train=True, max_tasks=None,
                                       rack_size=5, thermal_weight=0.15,
                                       hotspot_threshold_c=None, thermal_kwargs=None):
    """FULL FIX variant (Ablation B): thermally-VISIBLE state + thermal-penalized
    reward. `global_actor`/`local_actor` here MUST be built with
    GLOBAL_STATE_DIM_THERMAL / LOCAL_STATE_DIM_THERMAL (not the original
    dims) -- passing Phase-3-sized networks will raise a shape mismatch
    immediately, which is the correct, loud failure mode (silently truncating
    the state vector would hide the bug). hotspot_threshold_c defaults to
    thermal_model.hotspot_penalty's own default (32.0, recalibrated) unless
    overridden.
    """
    kwargs = {} if hotspot_threshold_c is None else {"threshold_c": hotspot_threshold_c}
    thermal = _setup_thermal(federation, rack_size=rack_size, **(thermal_kwargs or {}))

    heap, counter = _build_event_queue(workflows)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows}
    task_finish_time = {}
    task_dc_assignment = {}

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": [],
               "hotspot_penalty": [], "rack_temp": []}

    n_processed = 0
    while heap:
        ready_time, _, wf_id, node = heapq.heappop(heap)
        if max_tasks is not None and n_processed >= max_tasks:
            break
        dag = dags[wf_id]
        task_size = dag.nodes[node]["size_mi"]
        now = ready_time
        sim_hours = now / 3600.0

        g_state = global_state_thermal(federation, task_size, now, sim_hours, thermal)
        g_action, g_logprob = global_actor.act(g_state)
        chosen_dc = federation[g_action]

        l_state = local_state_thermal(chosen_dc, task_size, now, thermal)
        l_action, l_logprob = local_actor.act(l_state)
        server = chosen_dc.servers[l_action]

        predecessor_info = []
        for pred in dag.predecessors(node):
            data_mb = dag[pred][node]["data_mb"]
            same_dc = task_dc_assignment.get(pred) == chosen_dc.id
            predecessor_info.append((data_mb, same_dc))

        class _T:
            pass
        task_obj = _T()
        task_obj.size_mi = task_size

        result = assign_task(task_obj, chosen_dc, server, now=now, sim_time_hours=sim_hours,
                              predecessor_finish_info=predecessor_info)

        task_finish_time[node] = result.finish_time
        task_dc_assignment[node] = chosen_dc.id

        thermal.step(federation, now=now, dt=max(1.0, result.exec_time))
        rack_id = getattr(server, "rack_id", 0)
        rack_temp = thermal.rack_temp(chosen_dc, rack_id)
        hpen = hotspot_penalty(rack_temp, **kwargs)

        t_norm = np.clip(result.exec_time / T_SCALE, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE, 0, 5)
        reward_local = -(alpha * t_norm + (1 - alpha) * e_norm) - thermal_weight * hpen
        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)
        metrics["hotspot_penalty"].append(hpen)
        metrics["rack_temp"].append(rack_temp)

        if train and optimizers is not None:
            _update_step(optimizers, l_state, l_action, l_logprob, reward_local, local_actor, local_critic,
                         g_state, g_action, g_logprob, reward_global, global_actor, global_critic, gamma)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


def run_heuristic_episode_thermal(federation, workflows, dc_selector, server_selector,
                                   alpha=0.5, max_tasks=None, rack_size=5,
                                   thermal_weight=0.15, hotspot_threshold_c=None,
                                   thermal_kwargs=None):
    """Thermal-aware version of baselines._run_heuristic_episode (used for
    Random and Green-Opt so the Phase 4 comparison stays apples-to-apples --
    every method in the comparison is scored with the SAME thermal-penalty
    term, not just the trained model)."""
    thermal = _setup_thermal(federation, rack_size=rack_size, **(thermal_kwargs or {}))

    heap, counter = _build_event_queue(workflows)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows}
    task_finish_time = {}
    task_dc_assignment = {}

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": [],
               "hotspot_penalty": [], "rack_temp": []}

    n_processed = 0
    while heap:
        ready_time, _, wf_id, node = heapq.heappop(heap)
        if max_tasks is not None and n_processed >= max_tasks:
            break
        dag = dags[wf_id]
        task_size = dag.nodes[node]["size_mi"]
        now = ready_time
        sim_hours = now / 3600.0

        chosen_dc = dc_selector(federation, now, sim_hours)
        server = server_selector(chosen_dc, now)

        predecessor_info = []
        for pred in dag.predecessors(node):
            data_mb = dag[pred][node]["data_mb"]
            same_dc = task_dc_assignment.get(pred) == chosen_dc.id
            predecessor_info.append((data_mb, same_dc))

        class _T:
            pass
        task_obj = _T()
        task_obj.size_mi = task_size

        result = assign_task(task_obj, chosen_dc, server, now=now, sim_time_hours=sim_hours,
                              predecessor_finish_info=predecessor_info)

        task_finish_time[node] = result.finish_time
        task_dc_assignment[node] = chosen_dc.id

        thermal.step(federation, now=now, dt=max(1.0, result.exec_time))
        rack_id = getattr(server, "rack_id", 0)
        rack_temp = thermal.rack_temp(chosen_dc, rack_id)
        hpen = hotspot_penalty(rack_temp) if hotspot_threshold_c is None else hotspot_penalty(rack_temp, threshold_c=hotspot_threshold_c)

        t_norm = np.clip(result.exec_time / T_SCALE, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE, 0, 5)
        reward_local = -(alpha * t_norm + (1 - alpha) * e_norm) - thermal_weight * hpen
        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)
        metrics["hotspot_penalty"].append(hpen)
        metrics["rack_temp"].append(rack_temp)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics
