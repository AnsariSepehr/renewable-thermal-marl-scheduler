"""
Phase 5b -- Learned dynamic alpha (continuous actor output)
==============================================================
5a's diagnosis: giving the local agent urgency as an extra STATE input, and
reweighting the reward externally via a fixed formula, did NOT teach the
policy to actually condition its ACTIONS on urgency -- the "chosen server
MIPS vs urgency" probe went the wrong direction even after extensive
checkpoint-selection tuning. The reward-shaping half worked; the
action-conditioning half didn't.

5b's fix: instead of a fixed formula living outside the network, alpha
becomes a SECOND, CONTINUOUS output of the local actor's own policy head,
learned via the exact same policy-gradient signal that trains server
selection. This gives urgency-conditioned behavior a much shorter, more
direct gradient path: "this alpha value, in this state, led to this
advantage" instead of "hope the network notices state[52] correlates with
good outcomes."

Design choices, and why:

- Local actor now has a SHARED TRUNK feeding two heads: a categorical head
  (server choice, same as before) and a Beta-distribution head (alpha).
  Beta is a standard, clean choice for a policy-gradient action bounded in
  (0, 1) -- no tanh-squashing log-prob correction needed, unlike a Gaussian.
  The sampled value is linearly rescaled to [ALPHA_MIN, ALPHA_MAX].

- Both local critics (local + the local half of the global critic) now
  condition on the sampled alpha value too, since it's part of the action
  the critic must evaluate.

- ENTROPY REGULARIZATION (new in 5b, not present in 5a): a small bonus
  encouraging both the server-choice distribution and the alpha
  distribution to stay non-degenerate, added to the actor loss. Without
  this, a network free to choose its own alpha has an incentive to collapse
  toward whichever extreme maximizes the immediate reward for the actions
  it happens to already prefer (alpha doesn't change physics, only how an
  already-fixed outcome gets SCORED) -- entropy regularization keeps
  exploration alive long enough to actually learn a state-dependent policy
  instead of collapsing to a single global constant, which would just
  reduce to a worse version of fixed-alpha.

- DETERMINISTIC EVALUATION (new in 5b): Phase 3/4/5a always sampled actions
  stochastically, including at evaluation time, adding noise to every
  reported comparison table we've produced so far. 5b's evaluation path
  uses the policy's MODE instead: argmax for the discrete head, the Beta
  distribution's mean for alpha. This makes reported numbers reproducible
  and removes evaluation-time variance as a confound when comparing against
  fixed-alpha and 5a.

- Everything proven to matter in Phase 3/4/5a is carried over unchanged:
  congestion-aware state (Phase 3/4 fix), gradient clipping + TD clamping
  (Phase 3/4 fix), lr=1e-3 (proven to actually reach good policies within a
  150-episode budget, unlike the 3e-4 attempt), the same deadline
  calibration validated against the real fixed-alpha checkpoint
  (slack_range=(1.5, 4.0), ~16% baseline violation rate), and top-K
  checkpoint tracking + proper held-out validation (not a single noisy
  training-time snapshot).
"""

import heapq
import itertools
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical, Beta

from .datacenter_env import assign_task
from .scheduler_env import (
    global_state, T_SCALE, E_SCALE, GREEN_DELTA_SCALE, QUEUE_TIME_SCALE,
    MIPS_MAX, TASK_SIZE_SCALE, _build_event_queue, GLOBAL_STATE_DIM,
)
from .dynamic_alpha import (
    assign_workflow_deadlines, compute_urgency, local_state_dynamic,
    LOCAL_STATE_DIM_DYNAMIC, ALPHA_MIN, ALPHA_MAX,
)

ENTROPY_COEF = 0.01  # weight on the entropy bonus in the actor loss

# CRITICAL FIX (found via direct training-stability testing, not theory):
# Phase 5a's T_SCALE=10 / E_SCALE=2000 were fine when alpha was an external
# formula -- but now that alpha is a FREE variable the network chooses, those
# scales matter a lot. A "typical" fixed-alpha=0.5 task has t_norm ~ 0.073
# and e_norm ~ 0.183 under the OLD scales -- e_norm is ~2.5x bigger than
# t_norm for no reason related to urgency, just scale mismatch. Since alpha
# doesn't change physics (only how the SAME outcome gets scored), a network
# free to pick alpha immediately discovers it can always score better by
# favoring whichever term the mismatched scales made numerically smaller --
# regardless of context. Observed directly: alpha collapsed toward ~0.9 with
# shrinking variance within 50 episodes, independent of urgency -- reward
# hacking, not learning. Fix: rescale T and E so a "typical" task contributes
# roughly EQUALLY to both terms (both ~1.0 on average), calibrated from the
# real fixed-alpha baseline's own achieved averages (T~0.73s, E~365J, from
# Phase 3/4's validated results). This removes the free-lunch incentive --
# any alpha preference the network learns now has to come from genuine
# state-dependent reasoning, not a scale artifact.
T_SCALE_5B = 0.73    # seconds -- calibrated so typical t_norm ~= 1.0
E_SCALE_5B = 365.0   # Joules  -- calibrated so typical e_norm ~= 1.0


# ---------------------------------------------------------------------------
# Networks
# ---------------------------------------------------------------------------

class HybridLocalActor(nn.Module):
    """Shared trunk -> (categorical server head, Beta alpha head)."""

    def __init__(self, state_dim: int, n_servers: int, hidden: int = 128,
                 alpha_min: float = ALPHA_MIN, alpha_max: float = ALPHA_MAX):
        super().__init__()
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.server_head = nn.Linear(hidden, n_servers)
        # Beta distribution shape params, via softplus to keep them positive.
        # Bias initialized so the network starts near Beta(2,2) -- centered
        # at alpha=0.5 (matching the fixed baseline's starting point) with
        # moderate spread, rather than a random/degenerate initial shape.
        self.alpha_head = nn.Linear(hidden, 2)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, 0.5413)  # softplus(0.5413) ~= 1.0 -> Beta(2,2) after +1

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        server_probs = torch.softmax(self.server_head(h), dim=-1)
        raw_ab = torch.nn.functional.softplus(self.alpha_head(h)) + 1.0  # keep >1: unimodal, not U-shaped
        a_param, b_param = raw_ab[..., 0], raw_ab[..., 1]
        return server_probs, a_param, b_param

    def act(self, state: torch.Tensor, deterministic: bool = False):
        """Returns (server_action, server_logprob, server_entropy,
                     alpha_value, alpha_logprob, alpha_entropy).
        deterministic=True (for evaluation): argmax server choice, Beta mean
        for alpha -- no sampling noise, reproducible reported numbers."""
        server_probs, a_param, b_param = self.forward(state)
        server_dist = Categorical(probs=server_probs)
        alpha_dist = Beta(a_param, b_param)

        if deterministic:
            server_action = torch.argmax(server_probs)
            alpha_raw = a_param / (a_param + b_param)  # Beta mean
        else:
            server_action = server_dist.sample()
            alpha_raw = alpha_dist.sample()

        server_logprob = server_dist.log_prob(server_action)
        alpha_logprob = alpha_dist.log_prob(torch.clamp(alpha_raw, 1e-4, 1 - 1e-4))
        alpha_value = self.alpha_min + (self.alpha_max - self.alpha_min) * alpha_raw

        return (server_action.item(), server_logprob, server_dist.entropy(),
                float(alpha_value), alpha_logprob, alpha_dist.entropy())


class LearnedAlphaCritic(nn.Module):
    """Local critic: Q(state, server_onehot, alpha_scalar)."""

    def __init__(self, state_dim: int, n_servers: int, hidden: int = 128):
        super().__init__()
        self.n_servers = n_servers
        self.net = nn.Sequential(
            nn.Linear(state_dim + n_servers + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state, server_action_idx, alpha_value):
        one_hot = torch.zeros(self.n_servers)
        one_hot[server_action_idx] = 1.0
        alpha_t = torch.tensor([alpha_value], dtype=torch.float32)
        x = torch.cat([state, one_hot, alpha_t], dim=-1)
        return self.net(x).squeeze(-1)


class LearnedAlphaGlobalCritic(nn.Module):
    """Global critic: Q(global_state, global_onehot, local_state,
    local_onehot, local_alpha) -- same local-neighborhood design as Phase 3,
    extended with the local alpha value as part of the local action."""

    def __init__(self, global_state_dim, n_dc, local_state_dim, n_servers, hidden=128):
        super().__init__()
        self.n_dc = n_dc
        self.n_servers = n_servers
        in_dim = global_state_dim + n_dc + local_state_dim + n_servers + 1
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state, global_action_idx, local_state, local_action_idx, alpha_value):
        g_onehot = torch.zeros(self.n_dc)
        g_onehot[global_action_idx] = 1.0
        l_onehot = torch.zeros(self.n_servers)
        l_onehot[local_action_idx] = 1.0
        alpha_t = torch.tensor([alpha_value], dtype=torch.float32)
        x = torch.cat([global_state, g_onehot, local_state, l_onehot, alpha_t], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Scheduling / training loop
# ---------------------------------------------------------------------------

def run_learned_alpha_episode(federation, workflows_with_deadlines, global_actor, global_critic,
                               local_actor, local_critic, optimizers=None, train=True,
                               deterministic=False, max_tasks=None):
    """Same event-driven control flow as dynamic_alpha.run_dynamic_alpha_episode,
    except alpha is SAMPLED from the local actor's own Beta head instead of
    computed from a fixed formula. deterministic=True uses the policy's
    mode (no sampling noise) -- use this for evaluation, not training."""
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

        g_state = global_state(federation, task_size, now, sim_hours)
        g_action, g_logprob = global_actor.act(g_state, deterministic=deterministic)
        chosen_dc = federation[g_action]

        l_state = local_state_dynamic(chosen_dc, task_size, now, urgency)
        (l_action, l_logprob, l_entropy,
         alpha_task, alpha_logprob, alpha_entropy) = local_actor.act(l_state, deterministic=deterministic)
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

        t_norm = np.clip(result.exec_time / T_SCALE_5B, 0, 5)
        e_norm = np.clip(result.energy_joules / E_SCALE_5B, 0, 5)
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
        metrics["chosen_server_mips"].append(server.mips)
        metrics["compute_time_only"].append(task_size / server.mips)

        if train and optimizers is not None:
            _update_step_learned_alpha(
                optimizers, l_state, l_action, l_logprob, l_entropy, alpha_task,
                alpha_logprob, alpha_entropy, reward_local, local_actor, local_critic,
                g_state, g_action, g_logprob, reward_global, global_actor, global_critic)

        for succ in dag.successors(node):
            preds = list(dag.predecessors(succ))
            if all(p in task_finish_time for p in preds):
                succ_ready = max(task_finish_time[p] for p in preds)
                heapq.heappush(heap, (succ_ready, next(counter), wf_id, succ))

        n_processed += 1

    return metrics


def _update_step_learned_alpha(optimizers, l_state, l_action, l_logprob, l_entropy, alpha_task,
                                alpha_logprob, alpha_entropy, reward_local, local_actor, local_critic,
                                g_state, g_action, g_logprob, reward_global, global_actor, global_critic):
    # --- Local critic + hybrid local actor (server head + alpha head) ---
    q_local = local_critic(l_state, l_action, alpha_task)
    target_local = torch.tensor(reward_local, dtype=torch.float32)
    td_error_local = torch.clamp(target_local - q_local, -20.0, 20.0)

    critic_loss_local = td_error_local.pow(2)
    optimizers["local_critic"].zero_grad()
    critic_loss_local.backward()
    torch.nn.utils.clip_grad_norm_(local_critic.parameters(), max_norm=5.0)
    optimizers["local_critic"].step()

    # Joint policy-gradient loss over BOTH heads, sharing the same advantage
    # (they share a trunk too, so this jointly shapes the shared representation).
    # Entropy bonus (subtracted, since we're minimizing) keeps both
    # distributions from collapsing prematurely.
    actor_loss_local = (
        -(l_logprob + alpha_logprob) * td_error_local.detach()
        - ENTROPY_COEF * (l_entropy + alpha_entropy)
    )
    optimizers["local_actor"].zero_grad()
    actor_loss_local.backward()
    torch.nn.utils.clip_grad_norm_(local_actor.parameters(), max_norm=5.0)
    optimizers["local_actor"].step()

    # --- Global critic (sees local state+action+alpha too) + global actor ---
    q_global = global_critic(g_state, g_action, l_state, l_action, alpha_task)
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
