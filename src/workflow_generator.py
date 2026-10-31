"""
Phase 2 -- Workflow data pipeline
=================================
Builds the 1000-workflow benchmark used in every experiment.

* Task *attributes* (sizes) are resampled from the real Alibaba Cluster Trace
  GPU-v2020 (``pai_task_table.csv``), rescaled into the paper's 0.5k-1000k MI range.
  The trace has no precedence information, so
* the *structure* (DAG) is synthetic: layered fan-out / fan-in graphs in the
  style of the Pegasus Montage/Epigenomics workflows.
* Arrival times follow a Poisson process (paper Section V-B).

If the Alibaba file is not available, a log-normal fallback with a similar
shape is used and the source is reported, so the choice is never silent.

This module is the notebook's Phase 2 code (cells 32-36) moved into ``src/`` so
it can be run from the command line (``scripts/generate_workflows.py``).
"""
import glob
import os
import pickle

import networkx as nx
import numpy as np
import pandas as pd

PAI_TASK_COLUMNS = ["job_name", "task_name", "inst_num", "status", "start_time",
                    "end_time", "plan_cpu", "plan_mem", "plan_gpu", "gpu_type"]


def load_alibaba_task_attributes(path):
    """Load ``pai_task_table.csv`` under *path* (headerless raw format or with
    a header row). Returns a cleaned DataFrame with a ``duration`` column, or
    ``None`` if the file is not found."""
    candidates = glob.glob(os.path.join(path, "**", "pai_task_table.csv"), recursive=True)
    if not candidates:
        return None
    file = candidates[0]
    peek = pd.read_csv(file, nrows=5, header=None)
    first_row_is_header = str(peek.iloc[0, 0]).strip().lower() in ("job_name", "jobname")
    if first_row_is_header:
        df = pd.read_csv(file)
        df.columns = [c.strip().lower() for c in df.columns]
    else:
        df = pd.read_csv(file, header=None, names=PAI_TASK_COLUMNS)
    df["duration"] = df["end_time"] - df["start_time"]
    if "status" in df.columns:
        df = df[df["status"].astype(str).str.lower().isin(["terminated", "finished", "succeeded"])]
    df = df[df["duration"] > 0]
    return df.reset_index(drop=True)


def build_task_size_sampler(task_attrs_df, seed=0):
    """Return ``(sample_task_size, source_name)``. ``sample_task_size(n)`` gives
    n task sizes in MI (0.5k-1000k), resampled from real Alibaba durations
    (1st-99th percentile rescaled) or, as a fallback, from a log-normal."""
    rng = np.random.default_rng(seed)
    if task_attrs_df is not None and "duration" in task_attrs_df.columns:
        real = task_attrs_df["duration"].dropna()
        real = real[real > 0].values
        lo, hi = 500, 1_000_000
        d_lo, d_hi = np.percentile(real, [1, 99])

        def sample_task_size(n):
            picks = np.clip(rng.choice(real, size=n, replace=True), d_lo, d_hi)
            return lo + (picks - d_lo) / (d_hi - d_lo) * (hi - lo)

        return sample_task_size, "alibaba_pai_trace"

    def sample_task_size(n):
        return np.clip(rng.lognormal(mean=8.5, sigma=1.1, size=n), 500, 1_000_000)

    return sample_task_size, "synthetic_lognormal"


def generate_workflow_dag(workflow_id, sample_task_size, rng,
                          n_tasks_range=(10, 30), n_levels_range=(3, 6),
                          fanout_range=(1, 3), data_dep_range=(100, 10_000)):
    """One random layered DAG. Nodes carry ``size_mi`` and ``level``; edges carry
    ``data_mb``. Every non-entry node has at least one predecessor."""
    n_tasks = int(rng.integers(*n_tasks_range))
    n_levels = min(int(rng.integers(*n_levels_range)), n_tasks)

    tasks_per_level = np.ones(n_levels, dtype=int)
    remaining = n_tasks - n_levels
    if remaining > 0:
        tasks_per_level += rng.multinomial(remaining, [1 / n_levels] * n_levels)

    G = nx.DiGraph()
    sizes = sample_task_size(n_tasks)
    counter, levels = 0, []
    for level_idx, count in enumerate(tasks_per_level):
        level_nodes = []
        for _ in range(count):
            node_id = f"wf{workflow_id}_t{counter}"
            G.add_node(node_id, size_mi=float(sizes[counter]), level=level_idx)
            level_nodes.append(node_id)
            counter += 1
        levels.append(level_nodes)

    for level_idx in range(len(levels) - 1):
        for src in levels[level_idx]:
            n_links = min(int(rng.integers(fanout_range[0], fanout_range[1] + 1)),
                          len(levels[level_idx + 1]))
            for dst in rng.choice(levels[level_idx + 1], size=n_links, replace=False):
                G.add_edge(src, dst, data_mb=float(rng.uniform(*data_dep_range)))

    for level_idx in range(1, len(levels)):
        for node in levels[level_idx]:
            if G.in_degree(node) == 0:
                src = rng.choice(levels[level_idx - 1])
                G.add_edge(src, node, data_mb=float(rng.uniform(*data_dep_range)))

    assert nx.is_directed_acyclic_graph(G), "Generated graph is not a valid DAG!"
    return G


def generate_workflow_batch(n_workflows, sample_task_size, seed=0, arrival_rate_per_hour=5):
    """``n_workflows`` DAGs with Poisson-distributed inter-arrival gaps.
    Returns a list of ``{"workflow_id", "arrival_hour", "dag"}`` dicts."""
    from scipy.stats import poisson
    rng = np.random.default_rng(seed)
    gaps = poisson.rvs(mu=1, size=n_workflows, random_state=seed) / arrival_rate_per_hour + 0.01
    arrivals = np.cumsum(gaps)
    return [{"workflow_id": i, "arrival_hour": arrivals[i],
             "dag": generate_workflow_dag(i, sample_task_size, rng)}
            for i in range(n_workflows)]


def build_and_save(data_dir="data", n_workflows=1000, seed=42, out_name="workflows_1000.pkl"):
    """Full Phase 2: load trace (if present) -> sampler -> workflows -> pickle."""
    task_df = load_alibaba_task_attributes(data_dir)
    sampler, source = build_task_size_sampler(task_df, seed=seed)
    workflows = generate_workflow_batch(n_workflows, sampler, seed=seed)
    os.makedirs(data_dir, exist_ok=True)
    out_path = os.path.join(data_dir, out_name)
    with open(out_path, "wb") as f:
        pickle.dump(workflows, f)
    return out_path, source, workflows
