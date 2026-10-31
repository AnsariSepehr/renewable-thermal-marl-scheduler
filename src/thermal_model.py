"""
Phase 6 (new) -- Rack-level thermal modeling + VM/task migration penalty
==========================================================================
This module adds the piece the proposal calls "innovation 4" (joint
thermodynamic + migration-aware reward) which was NOT present in Phases
0-5b. It is written as a pure ADD-ON: it does not modify datacenter_env.py,
agents.py, or baselines.py. It only *reads* the existing Server/Datacenter
dataclasses and adds a parallel thermal-state layer + a reward-shaping
function you plug into the existing training loop.

Why this design (documented, so it's defensible in the paper's Methods):
- The paper (and our Phases 0-5b) has NO notion of physical rack position
  or temperature at all -- servers only have a busy/idle utilization state.
  We are not "replacing" anything, we are adding a first thermodynamic
  layer on top of the existing power model, which is the natural place
  for it: power draw (already modeled, Eq. 1) is the heat SOURCE; we now
  give that heat somewhere to go and something to track it.
- We deliberately use a lightweight lumped-capacitance thermal model
  (1 temperature scalar per server, first-order ODE) rather than full CFD
  (computational fluid dynamics). CFD is what real thermal-facilities papers
  use (see Ran et al. in your related work), but it needs airflow geometry
  data we don't have and is overkill for an RL reward signal -- the
  lumped-capacitance approach is standard in the *scheduling* literature
  (as opposed to the facilities-engineering literature) for exactly this
  reason: it's fast enough to compute inside a training loop, and it's
  still physically grounded (Newton's law of cooling), not ad hoc.

RECALIBRATION NOTE (added after the first real training run): the original
constants (heat_gain_per_watt=0.003, cooling_rate=0.05, threshold_c=45.0)
were calibrated only against a synthetic "one rack pinned at 100% for 300s"
stress test, which reaches ~57 degC there. Under the REAL scheduler's actual
task-arrival intensity, spread across 10 DCs x 5 racks x 25 servers, no rack
ever got close to that -- observed rack temperatures stayed in the 30-34 degC
band the entire training run, so the 45 degC threshold was simply never
reachable and hotspot_penalty was near-constant floor noise the whole time
(mean_hotspot_penalty went UP over training, not down -- the honest signal
that there was no real gradient to learn from). The constants below
(heat_gain_per_watt=0.006, cooling_rate=0.04, threshold_c=32.0) are
recalibrated so the threshold sits inside the ACTUALLY OBSERVED operating
range instead of far above it -- this is what makes hotspot_penalty a live,
varying signal instead of a floor.

Physical model:
- Each server has a scalar temperature T (deg C).
- Heat generated per step ~ proportional to the server's current power
  draw (Watts) -- a hot, busy server heats up; an idle one cools toward
  ambient.
- Servers are grouped into racks (contiguous chunks of a datacenter's
  server list, configurable rack_size). A rack's cooling capacity is
  SHARED among its servers -- this is what creates hotspots: pile too much
  load into one rack and its shared cooling can't keep up, even if the
  datacenter's *average* utilization looks fine. This directly models the
  gap flagged in the proposal ("چالش تمرکز بار در سرورهای مجاور... نقاط
  داغ حرارتی") that Phase 1's server-only power model could not represent.
- A rack is a "hotspot" once its mean temperature exceeds a threshold; the
  hotspot penalty is a SMOOTH (soft-plus-like) function of how far over the
  threshold it is, not a hard cutoff -- this gives the RL agent a gradient
  to learn from instead of a cliff.

Migration cost model:
- Migrating a running task from server A to server B is not free: it costs
  (a) a fixed data-transfer time/energy proportional to the task's memory
  footprint (reuses datacenter_env.data_transfer_time / P_COMM_WATTS so the
  unit system is consistent with the rest of the pipeline), and (b) a
  disruption penalty (the task's execution is paused during transfer).
  This is exactly the "هزینه سربار مهاجرت" (migration overhead cost) named
  as a research question (Q3) in the proposal, and it was NOT modeled
  anywhere in Phases 0-5b -- there was no migration ACTION at all before
  this module.
"""

from dataclasses import dataclass, field
import numpy as np

from .datacenter_env import data_transfer_time, P_COMM_WATTS


# ---------------------------------------------------------------------------
# 1. Rack assignment
# ---------------------------------------------------------------------------

def assign_racks(datacenter, rack_size: int = 5):
    """Splits a datacenter's server list into contiguous racks of `rack_size`.
    Returns dict {rack_id: [server_ids]}. Also tags each server with a
    `.rack_id` attribute (monkey-patched -- Server is a plain dataclass with
    no __slots__, so this is safe) so later lookups are O(1).
    """
    racks = {}
    for i, server in enumerate(datacenter.servers):
        rack_id = i // rack_size
        server.rack_id = rack_id
        racks.setdefault(rack_id, []).append(server.id)
    return racks


# ---------------------------------------------------------------------------
# 2. Thermal state -- one scalar temperature per server, updated each step
# ---------------------------------------------------------------------------

@dataclass
class ThermalState:
    """Holds temperature (deg C) for every server in the federation, plus
    the fixed physical constants of the lumped-capacitance model.

    heat_gain_per_watt : how many deg C/second a server heats up per Watt
                          of power draw above its cooling-driven baseline.
                          Calibrated so a fully-loaded high-power server
                          (~900W, see SERVER_TEMPLATES in datacenter_env.py)
                          reaches thermal steady-state (heat_gain ==
                          cooling loss) at roughly 55-60 degC, which matches
                          typical air-cooled rack inlet/outlet deltas
                          reported in datacenter thermal literature.
    cooling_rate        : Newton's-law-of-cooling coefficient PER RACK
                           (1/second). Larger = better/shared cooling
                           capacity = temperature relaxes to ambient faster.
                           This is the parameter that creates hotspots: a
                           rack with many busy servers and a modest
                           cooling_rate cannot dissipate heat fast enough.
    ambient_c            : the cold-aisle supply air temperature (deg C)
                            each rack's servers cool toward when idle.
    """
    temps: dict = field(default_factory=dict)       # {server_id: temp_c}
    heat_gain_per_watt: float = 0.006
    cooling_rate: float = 0.04
    ambient_c: float = 22.0

    @classmethod
    def initialize(cls, federation, **kwargs):
        state = cls(**kwargs)
        for dc in federation:
            for s in dc.servers:
                state.temps[s.id] = state.ambient_c
        return state

    def step(self, federation, now: float, dt: float = 1.0):
        """Advances every server's temperature by one simulation step of
        length `dt` seconds. Must be called once per scheduling event (or
        on a regular clock tick if you prefer -- either works, since dt is
        explicit) so the thermal state stays in sync with the event-driven
        simulation clock already used in scheduler_env.run_scheduling_episode.
        """
        for dc in federation:
            # group current power draw by rack so cooling is shared, not per-server
            rack_power = {}
            rack_members = {}
            for s in dc.servers:
                rid = getattr(s, "rack_id", 0)
                rack_power.setdefault(rid, 0.0)
                rack_power[rid] += s.power(now)
                rack_members.setdefault(rid, []).append(s)

            for rid, members in rack_members.items():
                total_power = rack_power[rid]
                # heat input distributed evenly across the rack's servers,
                # but driven by the RACK's total power (this is what makes
                # cooling "shared" -- one hot neighbor raises everyone's temp)
                heat_in_per_server = self.heat_gain_per_watt * total_power / max(1, len(members))
                for s in members:
                    T = self.temps[s.id]
                    # Newton's law of cooling toward ambient, plus heat input
                    dT = (heat_in_per_server - self.cooling_rate * (T - self.ambient_c)) * dt
                    self.temps[s.id] = T + dT

    def rack_temp(self, datacenter, rack_id: int) -> float:
        members = [s for s in datacenter.servers if getattr(s, "rack_id", None) == rack_id]
        return float(np.mean([self.temps[s.id] for s in members]))

    def server_temp(self, server) -> float:
        return self.temps[server.id]

    def all_rack_temps(self, datacenter) -> dict:
        rack_ids = sorted(set(getattr(s, "rack_id", 0) for s in datacenter.servers))
        return {rid: self.rack_temp(datacenter, rid) for rid in rack_ids}


# ---------------------------------------------------------------------------
# 3. Hotspot penalty -- smooth, not a hard cutoff
# ---------------------------------------------------------------------------

def hotspot_penalty(rack_temp_c: float, threshold_c: float = 32.0, sharpness: float = 3.0) -> float:
    """Softplus-shaped penalty: ~0 well below threshold, grows smoothly and
    then roughly linearly above it. `sharpness` controls how quickly the
    penalty ramps up around the threshold (smaller = softer knee).
    Returns a NON-NEGATIVE penalty in the same rough numeric range (0-5ish)
    as the other reward terms in scheduler_env.py's T_SCALE/E_SCALE-normalized
    components, so it can be added to the existing reward without swamping it.
    """
    x = (rack_temp_c - threshold_c) / max(sharpness, 1e-6)
    # softplus: log(1+exp(x)), numerically stable version
    penalty = np.log1p(np.exp(-abs(x))) + max(x, 0)
    return float(penalty)


# ---------------------------------------------------------------------------
# 4. Migration cost -- reuses the existing comm-energy/time model
# ---------------------------------------------------------------------------

@dataclass
class MigrationResult:
    transfer_time_s: float
    transfer_energy_j: float
    disruption_penalty: float  # unitless, added directly to reward


def migration_cost(task_memory_mb: float, from_server, to_server,
                    disruption_weight: float = 0.3) -> MigrationResult:
    """Cost of live-migrating a task's memory footprint from one server to
    another. Reuses datacenter_env's own bandwidth/energy constants so units
    stay consistent with the rest of the pipeline (no new made-up constants
    for the transfer physics itself -- only `disruption_weight`, which is a
    reward-shaping hyperparameter, is new).
    """
    same_dc = (from_server.dc_id == to_server.dc_id)
    t = data_transfer_time(task_memory_mb, same_dc)
    energy = t * P_COMM_WATTS
    disruption = disruption_weight * t  # longer transfer -> more service disruption
    return MigrationResult(transfer_time_s=t, transfer_energy_j=energy,
                            disruption_penalty=disruption)


# ---------------------------------------------------------------------------
# 5. Reward augmentation -- drop-in extension of the existing local reward
# ---------------------------------------------------------------------------

def augmented_local_reward(base_reward_local: float, rack_temp_c: float,
                            migrated: bool = False, migration: MigrationResult = None,
                            thermal_weight: float = 0.15, migration_weight: float = 0.25,
                            hotspot_threshold_c: float = 45.0) -> dict:
    """Combines the EXISTING reward_local (= -(alpha*T_norm + (1-alpha)*E_norm),
    computed exactly as before in baselines.py / scheduler_env.py) with the
    two new terms this module adds. Returns a dict (not just a float) so the
    notebook can log each component separately for the diagnostic plots --
    this matters for an honest ablation: reviewers will ask "how much of the
    improvement comes from thermal-awareness vs. migration-awareness", and
    without separately logged components that question is unanswerable.
    """
    thermal_pen = hotspot_penalty(rack_temp_c, threshold_c=hotspot_threshold_c)
    migration_pen = migration.disruption_penalty if (migrated and migration is not None) else 0.0

    reward_total = base_reward_local - thermal_weight * thermal_pen - migration_weight * migration_pen

    return {
        "reward_base": base_reward_local,
        "thermal_penalty": thermal_pen,
        "migration_penalty": migration_pen,
        "reward_augmented": reward_total,
    }
