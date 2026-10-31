"""
Phase 5a -- Dynamic alpha (deterministic, deadline-urgency-based)
====================================================================
Fixes the paper's stated limitation: alpha in Rlocal = alpha*T + (1-alpha)*E
was a single constant for the entire workflow's lifetime. Here, alpha is
recomputed for EVERY task, as a deterministic function of how much of that
task's workflow deadline "budget" has already been used up:

    urgency  = (now - workflow_arrival) / (workflow_deadline - workflow_arrival)
    alpha(t) = ALPHA_MIN + (ALPHA_MAX - ALPHA_MIN) * clip(urgency, 0, 1)

- urgency ~ 0  (just arrived, lots of slack)      -> alpha near ALPHA_MIN
  (favours energy -- Rlocal = alpha*T + (1-alpha)*E, low alpha weights E more)
- urgency ~ 1  (right at/past the deadline)        -> alpha near ALPHA_MAX
  (favours speed -- high alpha weights T more)

No new learnable parameters (this is 5a, the "safe" design decision). The
ONLY architectural change is that the local agent's state gains ONE extra
scalar (its own current urgency value), so the local policy can *condition*
its action choice on urgency too, not just the reward weighting -- e.g.
learning to prefer faster-but-less-green servers specifically when urgent.
LOCAL_STATE_DIM grows from 52 to 53; nothing else about the network
architecture changes.

Because dynamic alpha requires deadlines, and Phase 2's workflow generator
didn't assign any, this module also provides `assign_workflow_deadlines()`,
which computes each workflow's deadline from its critical path (longest
chain of task sizes, evaluated as if running on a single average-speed
server -- a standard, documented assumption for deadline assignment in the
workflow-scheduling literature) stretched by a random per-workflow slack
factor, so the dataset contains a realistic MIX of tight and loose
deadlines rather than one uniform value (needed to actually exercise the
full range of urgency during training/evaluation).
"""

import heapq
import itertools
import numpy as np
import torch
import networkx as nx

from .datacenter_env import assign_task, SERVER_TEMPLATES
from .scheduler_env import (
    global_state, T_SCALE, E_SCALE, GREEN_DELTA_SCALE, QUEUE_TIME_SCALE,
    MIPS_MAX, TASK_SIZE_SCALE, _build_event_queue, GLOBAL_STATE_DIM,
)

ALPHA_MIN = 0.3   # softened from an earlier 0.1 -- see note below
ALPHA_MAX = 0.95  # near-deadline ceiling -- never fully ignore energy even when urgent
LOCAL_STATE_DIM_DYNAMIC = 25 * 2 + 2 + 1  # speed + congestion (25 each) + size*2 + urgency = 53

# NOTE on ALPHA_MIN (post-5a-validation tuning): the first version of this
# module used ALPHA_MIN=0.1. Since most tasks in a well-trained system are
# NOT urgent (violations become rare once the policy converges), ALPHA_MIN
# is the value actually governing the vast majority of scheduling decisions
# -- and 0.1 turned out to weight energy so aggressively that it dragged
# down average execution time/reward across the whole run, without enough
# genuinely-urgent tasks to earn that cost back. 0.3 keeps meaningfully more
# weight on time for the common case while still relaxing well below the
# paper's fixed 0.5 when there IS slack to spend.

AVG_SERVER_MIPS = float(np.mean([t[2] for t in SERVER_TEMPLATES]))  # for critical-path estimation


# ---------------------------------------------------------------------------
# Deadline assignment
# ---------------------------------------------------------------------------

def _critical_path_hours(dag: nx.DiGraph) -> float:
    """Longest chain of task sizes through the DAG, evaluated as if every
    task ran back-to-back on one server of average speed (ignores queueing
    and parallelism -- a standard simplifying assumption purely for
    estimating a reasonable deadline, not used anywhere in the actual
    scheduling decision)."""
    finish_estimate = {}
    for node in nx.topological_sort(dag):
        own_time = dag.nodes[node]["size_mi"] / AVG_SERVER_MIPS  # seconds
        preds = list(dag.predecessors(node))
        start = max((finish_estimate[p] for p in preds), default=0.0)
        finish_estimate[node] = start + own_time
    critical_path_s = max(finish_estimate.values())
    return critical_path_s / 3600.0


def assign_workflow_deadlines(workflows, slack_range=(1.5, 4.0), seed=0):
    """Returns a NEW list of workflow dicts (doesn't mutate the input), each
    with an added 'deadline_hour' field: arrival_hour + slack * critical_path.
    `slack` is drawn per-workflow from `slack_range`, uniformly at random, so
    the dataset has a realistic mix of tight (low slack) and loose (high
    slack) deadlines -- necessary to actually exercise a range of urgency
    values during training/evaluation."""
    rng = np.random.default_rng(seed)
    out = []
    for wf in workflows:
        cp_hours = _critical_path_hours(wf["dag"])
        slack = rng.uniform(*slack_range)
        deadline_hour = wf["arrival_hour"] + slack * cp_hours
        new_wf = dict(wf)
        new_wf["critical_path_hours"] = cp_hours
        new_wf["slack_factor"] = slack
        new_wf["deadline_hour"] = deadline_hour
        out.append(new_wf)
    return out


# ---------------------------------------------------------------------------
# Urgency / dynamic alpha
# ---------------------------------------------------------------------------

def compute_urgency(now_s: float, arrival_s: float, deadline_s: float) -> float:
    span = deadline_s - arrival_s
    if span <= 0:
        return 1.0
    return float(np.clip((now_s - arrival_s) / span, 0.0, 1.0))


def dynamic_alpha(urgency: float, alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX) -> float:
    return alpha_min + (alpha_max - alpha_min) * urgency


# ---------------------------------------------------------------------------
# State construction (local state + 1 urgency scalar)
# ---------------------------------------------------------------------------

def local_state_dynamic(dc, task_size_mi: float, now: float, urgency: float) -> torch.Tensor:
    """Same as scheduler_env.local_state, plus one extra scalar: this task's
    current urgency (0=just arrived/relaxed, 1=at or past its deadline)."""
    speed_n = [s.mips / MIPS_MAX for s in dc.servers]
    congestion_n = [np.clip((s.busy_until - now) / QUEUE_TIME_SCALE, 0, 5) for s in dc.servers]
    size_n = np.clip(task_size_mi / TASK_SIZE_SCALE, 0, 2)
    vec = speed_n + congestion_n + [size_n, size_n, urgency]
    return torch.tensor(vec, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Training / evaluation loop (mirrors scheduler_env.run_scheduling_episode)
# ---------------------------------------------------------------------------

def run_dynamic_alpha_episode(federation, workflows_with_deadlines, global_actor, global_critic,
                               local_actor, local_critic, optimizers=None, gamma=0.0,
                               train=True, max_tasks=None,
                               alpha_min=ALPHA_MIN, alpha_max=ALPHA_MAX):
    """Identical control flow to scheduler_env.run_scheduling_episode, except:
    (1) alpha is recomputed per task from deadline urgency instead of being a
    fixed constant, and (2) the local agent's state carries that urgency
    value too. Requires `workflows_with_deadlines` to have gone through
    assign_workflow_deadlines() first.

    Additionally logs, per task: `urgency`, `alpha_used`, and
    `deadline_violation` (1 if this task finished after its workflow's
    deadline, else 0) -- everything Phase 5c's ablation will need.
    """
    heap, counter = _build_event_queue(workflows_with_deadlines)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows_with_deadlines}
    deadlines_s = {wf["workflow_id"]: wf["deadline_hour"] * 3600.0 for wf in workflows_with_deadlines}
    arrivals_s = {wf["workflow_id"]: wf["arrival_hour"] * 3600.0 for wf in workflows_with_deadlines}

    task_finish_time = {}
    task_dc_assignment = {}

    metrics = {"exec_time": [], "energy": [], "reward_local": [], "reward_global": [],
               "wait_time": [], "chosen_dc": [], "task_size": [],
               "urgency": [], "alpha_used": [], "deadline_violation": [],
               "chosen_server_mips": [], "compute_time_only": []}

    n_processed = 0
    while heap:
        ready_time, _, wf_id, node = heapq.heappop(heap)
        if max_tasks is not None and n_processed >= max_tasks:
            break
        dag = dags[wf_id]
        task_size = dag.nodes[node]["size_mi"]
        now = ready_time
        sim_hours = now / 3600.0

        urgency = compute_urgency(now, arrivals_s[wf_id], deadlines_s[wf_id])
        alpha_task = dynamic_alpha(urgency, alpha_min, alpha_max)

        # --- Global agent: pick a datacenter (state unchanged from Phase 3) ---
        g_state = global_state(federation, task_size, now, sim_hours)
        g_action, g_logprob = global_actor.act(g_state)
        chosen_dc = federation[g_action]

        # --- Local agent: pick a server (state now includes urgency) ---
        l_state = local_state_dynamic(chosen_dc, task_size, now, urgency)
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
        reward_local = -(alpha_task * t_norm + (1 - alpha_task) * e_norm)
        green_delta_norm = np.clip(result.green_energy_delta / GREEN_DELTA_SCALE, -2, 2)
        reward_global = green_delta_norm + reward_local

        violation = 1 if result.finish_time > deadlines_s[wf_id] else 0

        metrics["exec_time"].append(result.exec_time)
        metrics["energy"].append(result.energy_joules)
        metrics["wait_time"].append(result.waiting_time)
        metrics["reward_local"].append(reward_local)
        metrics["reward_global"].append(reward_global)
        metrics["chosen_dc"].append(chosen_dc.id)
        metrics["task_size"].append(task_size)
        metrics["urgency"].append(urgency)
        metrics["alpha_used"].append(alpha_task)
        metrics["deadline_violation"].append(violation)
        # Decoupled from realized waiting time -- these isolate the AGENT'S
        # CHOICE (which server it picked) from exogenous queueing delay that
        # accumulated before this task was even scheduled, which the earlier
        # confounded exec_time-per-urgency-bucket diagnostic couldn't tell apart.
        metrics["chosen_server_mips"].append(server.mips)
        metrics["compute_time_only"].append(task_size / server.mips)

        if train and optimizers is not None:
            _update_step_dynamic(optimizers, l_state, l_action, l_logprob, reward_local,
                                  local_actor, local_critic, g_state, g_action, g_logprob,
                                  reward_global, global_actor, global_critic)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


def _update_step_dynamic(optimizers, l_state, l_action, l_logprob, reward_local, local_actor, local_critic,
                          g_state, g_action, g_logprob, reward_global, global_actor, global_critic):
    """Identical update rule to scheduler_env._update_step (same gradient
    clipping / TD-clamping safeguards) -- only the inputs differ (l_state now
    includes urgency, reward_local now uses per-task dynamic alpha)."""
    q_local = local_critic(l_state, l_action)
    target_local = torch.tensor(reward_local, dtype=torch.float32)
    td_error_local = torch.clamp(target_local - q_local, -20.0, 20.0)

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
# Fixed-alpha evaluation WITH deadline tracking (for a fair 5a comparison
# against the dynamic-alpha model -- same workflows/deadlines, same
# environment, only the alpha rule and local-state size differ, matching
# each model's own trained state format).
# ---------------------------------------------------------------------------

from .scheduler_env import local_state as local_state_fixed  # 52-dim, no urgency


def run_fixed_alpha_eval_with_deadlines(federation, workflows_with_deadlines, global_actor, global_critic,
                                         local_actor, local_critic, alpha=0.5, max_tasks=None,
                                         deterministic=False):
    """Evaluation-only (no training) pass of a FIXED-alpha model (Phase 3's
    trained networks, 52-dim local state) over deadline-bearing workflows,
    logging the same deadline_violation metric run_dynamic_alpha_episode
    does -- so Phase 5a/5c can compare violation rates on equal footing."""
    heap, counter = _build_event_queue(workflows_with_deadlines)
    dags = {wf["workflow_id"]: wf["dag"] for wf in workflows_with_deadlines}
    deadlines_s = {wf["workflow_id"]: wf["deadline_hour"] * 3600.0 for wf in workflows_with_deadlines}

    task_finish_time = {}
    task_dc_assignment = {}
    metrics = {"exec_time": [], "energy": [], "reward_global": [], "wait_time": [],
               "deadline_violation": []}

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
        g_action, _ = global_actor.act(g_state, deterministic=deterministic)
        chosen_dc = federation[g_action]

        l_state = local_state_fixed(chosen_dc, task_size, now)
        l_action, _ = local_actor.act(l_state, deterministic=deterministic)
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
        metrics["reward_global"].append(reward_global)
        metrics["deadline_violation"].append(1 if result.finish_time > deadlines_s[wf_id] else 0)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))
        n_processed += 1

    return metrics
