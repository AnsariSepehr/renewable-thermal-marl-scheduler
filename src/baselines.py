"""
Phase 4 -- Non-learning comparison baselines: Random and Green-Opt
====================================================================
Both reuse the exact same event-driven scheduling loop shape as
scheduler_env.run_scheduling_episode (same environment, same workflows, same
metrics format) so comparisons against the trained baseline (Phase 3) and the
dynamic-alpha model (Phase 5) are apples-to-apples. The only thing that
changes is HOW the datacenter/server are chosen -- no networks, no training.

Random   -- picks a datacenter and a server uniformly at random. The paper's
            floor: if this doesn't get beaten by everything else, nothing
            downstream is working.
Green-Opt -- picks the datacenter with the highest current green energy
            surplus (paper's stated criterion). The paper additionally
            favours "active" hosts with spare capacity over idle ones; our
            environment only supports one task per server at a time (see
            Phase 1 notes), so that specific preference doesn't map onto our
            model -- we substitute the fastest currently-free server, which
            keeps the same underlying goal (finish fast on the greenest DC
            available) without assuming a form of multi-tasking our
            environment doesn't implement. This substitution is documented
            here rather than silently changed.
"""

import heapq
import itertools
import numpy as np

from .datacenter_env import assign_task
from .scheduler_env import T_SCALE, E_SCALE, GREEN_DELTA_SCALE, _build_event_queue


def _run_heuristic_episode(federation, workflows, dc_selector, server_selector,
                            alpha=0.5, max_tasks=None):
    """Shared event-driven loop for any non-learning heuristic. `dc_selector`
    and `server_selector` are plain functions (no gradients, no networks)."""
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

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


# ---------------------------------------------------------------------------
# Random
# ---------------------------------------------------------------------------

def _random_dc_selector(rng):
    def select(federation, now, sim_hours):
        return federation[rng.integers(0, len(federation))]
    return select


def _random_server_selector(rng):
    def select(dc, now):
        return dc.servers[rng.integers(0, len(dc.servers))]
    return select


def run_random_episode(federation, workflows, alpha=0.5, max_tasks=None, seed=0):
    rng = np.random.default_rng(seed)
    return _run_heuristic_episode(federation, workflows,
                                   _random_dc_selector(rng), _random_server_selector(rng),
                                   alpha=alpha, max_tasks=max_tasks)


# ---------------------------------------------------------------------------
# Green-Opt
# ---------------------------------------------------------------------------

def _greenopt_dc_selector():
    def select(federation, now, sim_hours):
        return max(federation, key=lambda dc: dc.green_power(sim_hours) - dc.total_power(now))
    return select


def _greenopt_server_selector():
    def select(dc, now):
        free = dc.free_servers(now)
        if free:
            return max(free, key=lambda s: s.mips)  # fastest free server
        return min(dc.servers, key=lambda s: s.busy_until)  # soonest to free up
    return select


def run_greenopt_episode(federation, workflows, alpha=0.5, max_tasks=None):
    return _run_heuristic_episode(federation, workflows,
                                   _greenopt_dc_selector(), _greenopt_server_selector(),
                                   alpha=alpha, max_tasks=max_tasks)
