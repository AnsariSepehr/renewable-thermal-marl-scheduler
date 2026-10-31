"""
Phase 3 -- Actor-Critic networks
=================================
Two levels, matching the paper's Fig. 2/3 architecture:

- Global actor/critic: picks WHICH DATACENTER a task goes to.
- Local actor/critic:  picks WHICH SERVER (within the chosen datacenter) runs it.

Critic design follows the paper's "local neighborhood" restriction (Section IV-B):
- The LOCAL critic only ever sees its own agent's state + action (Eq. 12 with N=1
  neighbor) -- this is what makes training fast; no info from other local agents.
- The GLOBAL critic sees its own state + action, PLUS the state + action of the
  ONE local scheduler it dispatched the task to (not all 10) -- again, local
  neighborhood only, not the full multi-agent critic in Fig. 2.

Implementation simplification (documented on purpose): the paper instantiates
one local scheduler per datacenter. Since every datacenter here has the same
number of servers (25) and the local state format is identical, we use ONE
shared-parameter local actor-critic network reused across all 10 datacenters,
instead of 10 separately-trained copies. This is standard parameter sharing
for homogeneous agents in MARL, keeps training tractable, and -- importantly
for our later dynamic-alpha comparison -- applies identically to the baseline
and improved model, so it doesn't bias the comparison between them.
"""

import torch
import torch.nn as nn


class Actor(nn.Module):
    """Outputs a categorical policy pi(a|s) over a discrete action set
    (datacenters for the global actor, servers for the local actor)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        logits = self.net(state)
        return torch.softmax(logits, dim=-1)

    def act(self, state: torch.Tensor, deterministic: bool = False):
        """Samples an action from the current policy. Returns (action_index,
        log_prob) -- both needed for the policy-gradient update.
        deterministic=True (needed for Phase 5b's deterministic-evaluation
        path): picks argmax instead of sampling -- use for evaluation, not
        training. Default is unchanged (stochastic), fully backward
        compatible with Phase 3/4/5a code that doesn't pass this argument."""
        probs = self.forward(state)
        dist = torch.distributions.Categorical(probs=probs)
        if deterministic:
            action = torch.argmax(probs)
        else:
            action = dist.sample()
        return action.item(), dist.log_prob(action)


class Critic(nn.Module):
    """Estimates Q(s, a) for a DISCRETE action, given as a one-hot vector
    concatenated onto the state. Used for the local critic (Eq. 12, N=1)."""

    def __init__(self, state_dim: int, action_dim: int, hidden: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, state: torch.Tensor, action_index: int) -> torch.Tensor:
        one_hot = torch.zeros(self.action_dim)
        one_hot[action_index] = 1.0
        x = torch.cat([state, one_hot], dim=-1)
        return self.net(x).squeeze(-1)


class GlobalCritic(nn.Module):
    """Estimates Q(x, a_global, a_local) where x includes the global state AND
    the local state of the ONE datacenter the task was dispatched to (paper
    Eq. 12 / Fig. 3's local-neighborhood restriction -- not all local agents,
    just the one involved in this step)."""

    def __init__(self, global_state_dim: int, n_dc: int,
                 local_state_dim: int, n_servers: int, hidden: int = 128):
        super().__init__()
        self.n_dc = n_dc
        self.n_servers = n_servers
        in_dim = global_state_dim + n_dc + local_state_dim + n_servers
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, global_state, global_action_idx, local_state, local_action_idx):
        g_onehot = torch.zeros(self.n_dc)
        g_onehot[global_action_idx] = 1.0
        l_onehot = torch.zeros(self.n_servers)
        l_onehot[local_action_idx] = 1.0
        x = torch.cat([global_state, g_onehot, local_state, l_onehot], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# Phase 5b -- Actor/Critic variants with a LEARNED continuous alpha output
# ---------------------------------------------------------------------------
#
# LocalActorWithAlpha adds a second head to the local actor: alongside the
# usual categorical distribution over servers, it also outputs a continuous
# alpha in [alpha_min, alpha_max] -- a genuine upgrade from Phase 5a's fixed
# urgency-formula, since alpha is now a function the NETWORK learns (and can
# condition on the full state, not just urgency).
#
# IMPORTANT: naively training this alpha head via policy gradient on
# Rlocal = alpha*T + (1-alpha)*E is a known trap -- that reward is LINEAR in
# alpha, so it has no genuine interior optimum; the gradient just pushes
# alpha to whichever extreme (0 or 1) makes whichever of T/E is smaller at
# that instant, collapsing to a near-constant value divorced from urgency.
# The fix (implemented in dynamic_alpha_5b.py, not here) is an explicit,
# ALPHA-INDEPENDENT deadline-violation penalty added to the reward, which
# breaks the pure linearity and gives alpha a real, non-trivial reason to
# track urgency.

class LocalActorWithAlpha(nn.Module):
    """Alpha is a DETERMINISTIC, LEARNED second output of the shared trunk --
    literally "a second continuous output of the Local Actor's policy head",
    per the original 5b design brief. It is NOT trained via policy gradient
    on the reward it parametrizes (see the module-level warning above and
    dynamic_alpha_5b.py's docstring for why that collapses); instead it's
    trained via a separate supervised regression loss toward a
    deadline-grounded target. The server-selection head is unaffected by
    this choice and still trains via policy gradient exactly as in Phase 3/5a."""

    def __init__(self, state_dim: int, n_servers: int, alpha_min: float, alpha_max: float,
                 hidden: int = 128):
        super().__init__()
        self.n_servers = n_servers
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max

        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.server_head = nn.Linear(hidden, n_servers)
        self.alpha_head = nn.Linear(hidden, 1)

    def forward(self, state: torch.Tensor):
        h = self.trunk(state)
        server_probs = torch.softmax(self.server_head(h), dim=-1)
        alpha_value = torch.sigmoid(self.alpha_head(h)).squeeze(-1) * (self.alpha_max - self.alpha_min) + self.alpha_min
        return server_probs, alpha_value

    def act(self, state: torch.Tensor):
        """Samples the server choice (stochastic, policy-gradient action) and
        returns the network's current alpha PREDICTION (deterministic --
        this is a regression output, not a sampled action). Returns
        (server_idx, alpha_value_float, alpha_value_tensor, server_log_prob).
        alpha_value_tensor keeps the graph connection needed for the
        regression loss; alpha_value_float is the plain number used
        everywhere else (reward computation, state features, logging)."""
        server_probs, alpha_value = self.forward(state)
        server_dist = torch.distributions.Categorical(probs=server_probs)
        server_action = server_dist.sample()
        return server_action.item(), float(alpha_value.item()), alpha_value, server_dist.log_prob(server_action)


class LocalCriticWithAlpha(nn.Module):
    """Same role as Critic, but the action representation now includes the
    continuous alpha value alongside the one-hot server choice."""

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

    def forward(self, state: torch.Tensor, server_action_idx: int, alpha_value: float) -> torch.Tensor:
        one_hot = torch.zeros(self.n_servers)
        one_hot[server_action_idx] = 1.0
        alpha_t = torch.tensor([alpha_value], dtype=torch.float32)
        x = torch.cat([state, one_hot, alpha_t], dim=-1)
        return self.net(x).squeeze(-1)


class GlobalCriticWithAlpha(nn.Module):
    """Same role as GlobalCritic, extended to include the local alpha value
    in what it conditions on (since the local action is now
    (server, alpha) instead of just server)."""

    def __init__(self, global_state_dim: int, n_dc: int, local_state_dim: int,
                 n_servers: int, hidden: int = 128):
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
