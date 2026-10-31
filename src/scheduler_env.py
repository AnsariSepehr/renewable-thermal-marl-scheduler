"""
Phase 3 -- Scheduling loop, state/reward construction, and online training
============================================================================
Ties together src/datacenter_env.py (the physics) and src/agents.py (the
networks) into the actual hierarchical scheduling algorithm from the paper
(Section IV-B, Algorithms 1 & 2).

Event-driven scheduling: tasks become "ready" the moment all their DAG
predecessors have finished (or immediately, for entry tasks, at their
workflow's arrival time). We process tasks strictly in order of readiness
time using a priority queue -- this mirrors the paper's description ("Upon
receipt of a task completion notification, the global scheduler executes the
DRL agent... with each task that can now be scheduled").

Training method: online one-step actor-critic. For simplicity (and because
scheduling assignment decisions are near-Markovian given the state already
encodes server occupancy/queueing), we do NOT bootstrap the TD target across
tasks (gamma=0 by default: target = the immediate reward). This is a
documented simplification of the paper's general TD framing, kept
identical for the baseline (Phase 3) and the dynamic-alpha model (Phase 5)
so it does not bias their comparison. The `gamma` parameter is still exposed
if you want to experiment with bootstrapped targets later.
"""

import heapq
import itertools
import numpy as np
import torch
import networkx as nx

from .datacenter_env import assign_task, data_transfer_time

N_DC = 10
N_SERVERS_PER_DC = 25

# Normalization constants (documented, tunable) -- keep NN inputs/targets in
# a sane range. Derived from the magnitudes observed in the Phase 1 stress test.
MIPS_MAX = 604_800.0
GREEN_DELTA_SCALE = 3000.0     # Watts
TASK_SIZE_SCALE = 1_000_000.0  # matches paper's up-to-1000k MI task sizes
T_SCALE = 10.0                 # seconds -- normalizes execution time in the reward
E_SCALE = 2000.0               # Joules  -- normalizes energy in the reward
QUEUE_TIME_SCALE = 20.0        # seconds -- normalizes the "time until server free" congestion signal


# ---------------------------------------------------------------------------
# State construction
# ---------------------------------------------------------------------------

def global_state(federation, task_size_mi: float, now: float, sim_hours: float) -> torch.Tensor:
    """Paper's global state: green surplus/deficit, avg speed, avg utilization
    per datacenter, plus the task's size (used as a proxy for both CPU and
    memory requirement -- see note in agents.py / Phase 3 notebook about why
    we don't have separate real CPU/mem-per-task fields)."""
    green = [dc.green_power(sim_hours) - dc.total_power(now) for dc in federation]
    speed = [dc.avg_processing_speed() for dc in federation]
    congestion = [np.mean([np.clip((s.busy_until - now) / QUEUE_TIME_SCALE, 0, 5) for s in dc.servers])
                  for dc in federation]

    green_n = [np.clip(g / GREEN_DELTA_SCALE, -2, 2) for g in green]
    speed_n = [s / MIPS_MAX for s in speed]
    size_n = np.clip(task_size_mi / TASK_SIZE_SCALE, 0, 2)

    vec = green_n + speed_n + congestion + [size_n, size_n]  # size used twice (cpu & mem proxy)
    return torch.tensor(vec, dtype=torch.float32)


def local_state(dc, task_size_mi: float, now: float) -> torch.Tensor:
    """Paper's local state: per-server speed + utilization within the chosen
    datacenter, plus the task's size (cpu/mem proxy, see note above).

    IMPORTANT FIX (post Phase 3/4 validation): utilization was originally a
    binary busy/idle flag (0 or 1), which gives the agent no visibility into
    HOW backed up an already-busy server is. Empirically (see Phase 3/4
    diagnostics) this let policies collapse onto a single "attractive"
    server, causing waiting times -- and therefore execution time and
    energy, since energy = time x power -- to cascade to unbounded values
    within an episode (observed up to ~10^124 J/task for Common-Actor).
    We replace the binary flag with a continuous "seconds until this server
    is free" signal (busy_until - now, floored at 0, normalized), which
    gives the agent an actual congestion signal to learn from. Same vector
    length (25 servers), so GLOBAL_STATE_DIM/LOCAL_STATE_DIM are unchanged
    -- but the MEANING of that slot changed, so any checkpoint trained
    before this fix is stale and must be retrained.
    """
    speed_n = [s.mips / MIPS_MAX for s in dc.servers]
    congestion_n = [np.clip((s.busy_until - now) / QUEUE_TIME_SCALE, 0, 5) for s in dc.servers]
    size_n = np.clip(task_size_mi / TASK_SIZE_SCALE, 0, 2)
    vec = speed_n + congestion_n + [size_n, size_n]
    return torch.tensor(vec, dtype=torch.float32)


GLOBAL_STATE_DIM = N_DC * 3 + 2   # 32
LOCAL_STATE_DIM = N_SERVERS_PER_DC * 2 + 2  # 52


# ---------------------------------------------------------------------------
# Event-driven task scheduling loop
# ---------------------------------------------------------------------------

def _build_event_queue(workflows):
    """Seeds the priority queue with every workflow's entry tasks (in-degree
    0), ready at the workflow's arrival time. Returns (heap, counter) --
    counter is used to break ties without ever comparing graph/task objects."""
    heap = []
    counter = itertools.count()
    for wf in workflows:
        dag = wf["dag"]
        arrival_s = wf["arrival_hour"] * 3600.0
        for node in dag.nodes:
            if dag.in_degree(node) == 0:
                heapq.heappush(heap, (arrival_s, next(counter), wf["workflow_id"], node))
    return heap, counter


def run_scheduling_episode(federation, workflows, global_actor, global_critic,
                            local_actor, local_critic, optimizers=None,
                            alpha=0.5, gamma=0.0, train=True, max_tasks=None):
    """Runs the hierarchical global+local scheduler over `workflows` (a list
    of {"workflow_id", "arrival_hour", "dag"} dicts, e.g. loaded from
    data/workflows_1000.pkl), scheduling tasks in event order as they become
    ready. If `train=True`, performs an online actor-critic update after
    every task using the optimizers dict:
        {"global_actor", "global_critic", "local_actor", "local_critic"}

    `alpha` is the paper's fixed local-reward trade-off (Eq. 16). Phase 5
    will replace this constant with a dynamic, deadline-aware value -- this
    function's structure is written so that swap is a small, contained change
    (see the Phase 5 notebook).

    Returns a dict of per-task metrics (lists) for plotting/evaluation.
    """
    heap, counter = _build_event_queue(workflows)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows}

    task_finish_time = {}      # node_id -> finish time (seconds)
    task_dc_assignment = {}    # node_id -> dc_id

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": []}

    n_processed = 0
    while heap:
        ready_time, _, wf_id, node = heapq.heappop(heap)
        if max_tasks is not None and n_processed >= max_tasks:
            break
        dag = dags[wf_id]
        task_size = dag.nodes[node]["size_mi"]
        now = ready_time
        sim_hours = now / 3600.0

        # --- Global agent: pick a datacenter ---
        g_state = global_state(federation, task_size, now, sim_hours)
        g_action, g_logprob = global_actor.act(g_state)
        chosen_dc = federation[g_action]

        # --- Local agent: pick a server within that datacenter ---
        l_state = local_state(chosen_dc, task_size, now)
        l_action, l_logprob = local_actor.act(l_state)
        server = chosen_dc.servers[l_action]

        # --- Gather predecessor transfer info (needed by the environment) ---
        predecessor_info = []
        for pred in dag.predecessors(node):
            data_mb = dag[pred][node]["data_mb"]
            same_dc = task_dc_assignment.get(pred) == chosen_dc.id
            predecessor_info.append((data_mb, same_dc))

        class _T:  # lightweight task wrapper, matches assign_task's expected interface
            pass
        task_obj = _T()
        task_obj.size_mi = task_size

        result = assign_task(task_obj, chosen_dc, server, now=now, sim_time_hours=sim_hours,
                              predecessor_finish_info=predecessor_info)

        task_finish_time[node] = result.finish_time
        task_dc_assignment[node] = chosen_dc.id

        # --- Rewards (paper Eq. 14, 16) ---
        t_norm = np.clip(result.exec_time / T_SCALE, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE, 0, 5)
        reward_local = -(alpha * t_norm + (1 - alpha) * e_norm)
        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)

        # --- Online actor-critic update ---
        if train and optimizers is not None:
            _update_step(optimizers, l_state, l_action, l_logprob, reward_local, local_actor, local_critic,
                         g_state, g_action, g_logprob, reward_global, global_actor, global_critic, gamma)

        # --- Unlock successors whose predecessors are now all finished ---
        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


def _update_step(optimizers, l_state, l_action, l_logprob, reward_local, local_actor, local_critic,
                  g_state, g_action, g_logprob, reward_global, global_actor, global_critic, gamma):
    """One online actor-critic update (paper Algorithms 1 & 2's inner loop),
    local scheduler first, then global scheduler (global critic also
    consumes the local state/action, per the local-neighborhood design)."""

    # --- Local critic + actor ---
    q_local = local_critic(l_state, l_action)
    target_local = torch.tensor(reward_local, dtype=torch.float32)  # gamma=0 (see module docstring)
    td_error_local = torch.clamp(target_local - q_local, -20.0, 20.0)  # extra safety net

    critic_loss_local = td_error_local.pow(2)
    optimizers["local_critic"].zero_grad()
    critic_loss_local.backward()
    torch.nn.utils.clip_grad_norm_(local_critic.parameters(), max_norm=5.0)
    optimizers["local_critic"].step()

    actor_loss_local = -l_logprob * td_error_local.detach()
    optimizers["local_actor"].zero_grad()
    actor_loss_local.backward()
    torch.nn.utils.clip_grad_norm_(local_actor.parameters(), max_norm=5.0)
    optimizers["local_actor"].step()

    # --- Global critic (sees local state+action too) + global actor ---
    q_global = global_critic(g_state, g_action, l_state, l_action)
    target_global = torch.tensor(reward_global, dtype=torch.float32)
    td_error_global = torch.clamp(target_global - q_global, -20.0, 20.0)

    critic_loss_global = td_error_global.pow(2)
    optimizers["global_critic"].zero_grad()
    critic_loss_global.backward()
    torch.nn.utils.clip_grad_norm_(global_critic.parameters(), max_norm=5.0)
    optimizers["global_critic"].step()

    actor_loss_global = -g_logprob * td_error_global.detach()
    optimizers["global_actor"].zero_grad()
    actor_loss_global.backward()
    torch.nn.utils.clip_grad_norm_(global_actor.parameters(), max_norm=5.0)
    optimizers["global_actor"].step()


# ---------------------------------------------------------------------------
# Phase 4 -- "Common-Actor" comparison baseline
# ---------------------------------------------------------------------------
#
# Paper's description: "multiple actor networks are guided by a common
# critic" -- i.e. the generic (non-hierarchical-restricted) multi-agent
# actor-critic design (Lowe et al., Fig. 2 in the paper), where a critic sees
# *all* agents' info, contrasted against the paper's proposed local-
# neighborhood-restricted design (Fig. 3, which is what Phase 3's baseline
# already implements with its separate global/local critics).
#
# Mapped onto our two-level hierarchy: instead of a separate local_critic
# giving the local actor its own dedicated learning signal (Phase 3's
# design), BOTH the global actor and the local actor here are trained off
# the SAME single shared critic (the GlobalCritic, playing the role of the
# paper's "common critic"). This isolates exactly the variable the paper's
# comparison is testing: does a critic restricted to a local neighborhood
# (Phase 3) train better than one common critic guiding every actor
# (Common-Actor)? Everything else -- environment, workflows, network sizes,
# global/local actor architectures -- stays identical to Phase 3, so this is
# a fair, apples-to-apples comparison.

def run_common_actor_episode(federation, workflows, global_actor, common_critic,
                              local_actor, optimizers=None, alpha=0.5, train=True, max_tasks=None):
    """Same event-driven loop as run_scheduling_episode, but both actors are
    trained off ONE shared critic (`common_critic`, a GlobalCritic instance)
    instead of separate global/local critics. optimizers dict needs only:
        {"global_actor", "local_actor", "common_critic"}
    """
    heap, counter = _build_event_queue(workflows)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows}

    task_finish_time = {}
    task_dc_assignment = {}

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": []}

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

        t_norm = np.clip(result.exec_time / T_SCALE, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE, 0, 5)
        reward_local = -(alpha * t_norm + (1 - alpha) * e_norm)
        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)

        if train and optimizers is not None:
            # ONE shared critic call -> ONE TD error -> drives BOTH actors.
            q_common = common_critic(g_state, g_action, l_state, l_action)
            target = torch.tensor(reward_global, dtype=torch.float32)
            td_error = torch.clamp(target - q_common, -20.0, 20.0)  # extra safety net

            critic_loss = td_error.pow(2)
            optimizers["common_critic"].zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(common_critic.parameters(), max_norm=5.0)
            optimizers["common_critic"].step()

            actor_loss_global = -g_logprob * td_error.detach()
            optimizers["global_actor"].zero_grad()
            actor_loss_global.backward()
            torch.nn.utils.clip_grad_norm_(global_actor.parameters(), max_norm=5.0)
            optimizers["global_actor"].step()

            actor_loss_local = -l_logprob * td_error.detach()
            optimizers["local_actor"].zero_grad()
            actor_loss_local.backward()
            torch.nn.utils.clip_grad_norm_(local_actor.parameters(), max_norm=5.0)
            optimizers["local_actor"].step()

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics
