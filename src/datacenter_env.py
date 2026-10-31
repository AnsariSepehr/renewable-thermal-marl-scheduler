"""
Phase 1 -- Simulation Environment
==================================
This module implements the physical "world" the RL agents will act in later:

- Server / Datacenter / Federation data structures (Table II specs from the paper)
- Power model                    (paper Eq. 1-2)
- Execution time model           (paper Eq. 6)
- Energy model                   (paper Eq. 7-9)
- A synthetic green-energy (solar) signal per datacenter, since we don't have
  literal Gatton-campus PV data
- A `step()` function that, given a task + chosen datacenter + chosen server,
  advances the simulation and returns (waiting_time, transfer_time, exec_time,
  energy, reward_components)

Everything here is intentionally environment-only: no RL, no neural nets.
We validate it with simple rule-based decisions before any agent touches it.
"""

from dataclasses import dataclass, field
import numpy as np


# ---------------------------------------------------------------------------
# 1. Server specs -- derived from the paper's Table II (SPECpower_ssj2008 benchmark)
# ---------------------------------------------------------------------------

# (name, cores, MIPS, RAM_GB, bandwidth_GBps, idle_watts, active_watts)
SERVER_TEMPLATES = [
    ("Dell_PowerEdge_R740",   56, 604800, 64, 1.5, 50,  432),
    ("PowerEdge_C6100",       48, 587520, 64, 1.5, 227, 895),
    ("Dell_PowerEdge_R820",   32, 332800, 64, 1.0, 110, 446),
    ("IBM_iDataPlex_dx360",   16, 187712, 48, 1.0, 116, 475),
    ("Dell_PowerEdge_R710",   12, 146880, 64, 1.0, 62,  227),
]


@dataclass
class Server:
    id: int
    dc_id: int
    name: str
    cores: int
    mips: float          # processing speed F (millions of instructions / sec)
    ram_gb: float
    bandwidth_gbps: float
    p_idle: float         # Watts
    p_active: float        # Watts
    busy_until: float = 0.0     # simulation time at which this server becomes free
    current_task: object = None  # task currently executing (or None)

    def utilization(self, now: float) -> float:
        """1.0 if busy at time `now`, else 0.0 (paper treats servers as single-task workers)."""
        return 1.0 if self.busy_until > now else 0.0

    def power(self, now: float) -> float:
        """Paper Eq. 1: Pmi = Pidle + (Pactive - Pidle)*u   if u > 0
                          Pmi = 0                            otherwise
        i.e. an idle server (u=0) is assumed powered down / draws no power."""
        u = self.utilization(now)
        if u > 0:
            return self.p_idle + (self.p_active - self.p_idle) * u
        return 0.0


@dataclass
class Datacenter:
    id: int
    servers: list
    # Solar-like green power generation curve, indexed by hour-of-day (0-23), in Watts.
    solar_profile: np.ndarray
    location_name: str = "synthetic"

    def green_power(self, sim_time_hours: float) -> float:
        """Green power available right now, from the synthetic solar profile."""
        hour = int(sim_time_hours) % 24
        base = self.solar_profile[hour]
        return max(0.0, base)

    def total_power(self, now: float) -> float:
        return sum(s.power(now) for s in self.servers)

    def brown_power(self, now: float, sim_time_hours: float) -> float:
        """Paper Eq. 3: Pbrown = Ptotal - Pgreen (floored at 0)."""
        return max(0.0, self.total_power(now) - self.green_power(sim_time_hours))

    def avg_processing_speed(self) -> float:
        return float(np.mean([s.mips for s in self.servers]))

    def avg_utilization(self, now: float) -> float:
        return float(np.mean([s.utilization(now) for s in self.servers]))

    def free_servers(self, now: float):
        return [s for s in self.servers if s.busy_until <= now]


def make_solar_profile(peak_watts: float, seed: int = None) -> np.ndarray:
    """Generates one synthetic 24h solar generation curve: zero at night, a smooth
    bell shape peaking at midday, with a bit of cloud-cover-like noise.
    This stands in for the paper's real Gatton-campus PV data."""
    rng = np.random.default_rng(seed)
    hours = np.arange(24)
    curve = peak_watts * np.exp(-((hours - 12) ** 2) / (2 * 3.2 ** 2))
    curve[(hours < 6) | (hours > 18)] = 0.0
    noise = rng.normal(0, peak_watts * 0.05, size=24)
    curve = np.clip(curve + noise, 0, None)
    return curve


def build_federation(n_datacenters: int = 10, servers_per_dc: int = 25, seed: int = 0,
                      real_solar_profiles: dict = None, location_names: list = None):
    """Builds the 10-datacenter x 25-server federation described in the paper's
    experimental setup (Section V-A).

    real_solar_profiles: optional dict {location_name: 24-length np.array}, e.g.
        from src.solar_data.build_real_solar_profiles(). If given (together with
        location_names), datacenters use real PVGIS-derived curves instead of
        the synthetic bell-curve generator. See Phase 1.5.
    location_names: optional list of length n_datacenters giving each DC's
        location key into `real_solar_profiles`.
    """
    rng = np.random.default_rng(seed)
    datacenters = []
    server_id_counter = 0
    for dc_id in range(n_datacenters):
        servers = []
        for _ in range(servers_per_dc):
            template = SERVER_TEMPLATES[rng.integers(0, len(SERVER_TEMPLATES))]
            name, cores, mips, ram, bw, p_idle, p_active = template
            servers.append(Server(
                id=server_id_counter, dc_id=dc_id, name=name, cores=cores,
                mips=mips, ram_gb=ram, bandwidth_gbps=bw,
                p_idle=p_idle, p_active=p_active,
            ))
            server_id_counter += 1

        if real_solar_profiles is not None and location_names is not None:
            loc_name = location_names[dc_id]
            solar = real_solar_profiles[loc_name]
        else:
            peak = rng.uniform(800, 2500)
            solar = make_solar_profile(peak, seed=seed * 100 + dc_id)
            loc_name = "synthetic"

        datacenters.append(Datacenter(id=dc_id, servers=servers, solar_profile=solar,
                                       location_name=loc_name))
    return datacenters


# ---------------------------------------------------------------------------
# 2. Execution time and energy models -- paper Eq. 6-9
# ---------------------------------------------------------------------------

B_IN_GBPS = 10.0    # within the same datacenter
B_OUT_GBPS = 1.0    # between datacenters
P_COMM_WATTS = 5.0  # power draw associated with data transmission


def data_transfer_time(data_size_mb: float, same_datacenter: bool) -> float:
    """Part of paper Eq. 6: DT(ti, tj)."""
    bw_gbps = B_IN_GBPS if same_datacenter else B_OUT_GBPS
    bw_mbps = bw_gbps * 1000
    return data_size_mb / bw_mbps  # seconds


def execution_time(task_size_mi: float, server: Server, waiting_time: float,
                    predecessor_transfer_times: list) -> float:
    """Paper Eq. 6: Ttj = L/F + WT + max_transfer_from_predecessors."""
    compute_time = task_size_mi / server.mips
    max_transfer = max(predecessor_transfer_times) if predecessor_transfer_times else 0.0
    return compute_time + waiting_time + max_transfer


def task_energy(exec_time_s: float, server: Server, predecessor_transfer_mb: list) -> tuple:
    """Paper Eq. 7-9: computation energy + communication energy = total task energy.
    Returns (compute_energy_J, comm_energy_J, total_energy_J)."""
    compute_energy = exec_time_s * (1.0 * server.p_active + server.p_idle)
    comm_energy = 0.0
    for data_mb, same_dc in predecessor_transfer_mb:
        t = data_transfer_time(data_mb, same_dc)
        comm_energy += t * P_COMM_WATTS
    total_energy = compute_energy + comm_energy
    return compute_energy, comm_energy, total_energy


# ---------------------------------------------------------------------------
# 3. The step function -- ties it all together for one task assignment
# ---------------------------------------------------------------------------

@dataclass
class StepResult:
    waiting_time: float
    transfer_time: float
    exec_time: float
    energy_joules: float
    finish_time: float
    green_energy_delta: float  # Pgreen - Ptotal of the chosen DC at assignment time (for Rglobal)


def assign_task(task, datacenter: Datacenter, server: Server, now: float,
                 sim_time_hours: float, predecessor_finish_info: list) -> StepResult:
    """Simulates assigning `task` to `server` in `datacenter` at simulation time `now`.

    predecessor_finish_info: list of (data_size_mb, was_same_datacenter) for this
    task's predecessors -- used for transfer time/energy.
    """
    waiting_time = max(0.0, server.busy_until - now)
    predecessor_transfer_times = [
        data_transfer_time(mb, same_dc) for mb, same_dc in predecessor_finish_info
    ]

    exec_t = execution_time(task.size_mi, server, waiting_time, predecessor_transfer_times)
    _, _, energy = task_energy(exec_t, server, predecessor_finish_info)

    finish_time = now + waiting_time + exec_t
    server.busy_until = finish_time
    server.current_task = task

    green = datacenter.green_power(sim_time_hours)
    total = datacenter.total_power(now)
    green_energy_delta = green - total

    return StepResult(
        waiting_time=waiting_time,
        transfer_time=max(predecessor_transfer_times) if predecessor_transfer_times else 0.0,
        exec_time=exec_t,
        energy_joules=energy,
        finish_time=finish_time,
        green_energy_delta=green_energy_delta,
    )
