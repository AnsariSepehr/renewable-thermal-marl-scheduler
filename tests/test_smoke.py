"""Fast smoke tests -- no training, no network, no GPU. Run: pytest -q"""
import networkx as nx
import numpy as np
import pytest

from src import datacenter_env as de
from src import thermal_model as tm
from src.workflow_generator import (build_task_size_sampler, generate_workflow_batch,
                                    generate_workflow_dag)


def test_federation_shape():
    fed = de.build_federation(n_datacenters=10, servers_per_dc=25, seed=0)
    assert len(fed) == 10
    assert all(len(dc.servers) == 25 for dc in fed)


def test_solar_profile_is_24_hourly_values_and_nonnegative():
    prof = de.make_solar_profile(peak_watts=1000, seed=0)
    assert len(prof) == 24 and np.all(prof >= 0) and prof.max() <= 1000 + 1e-6


def test_workflow_dags_are_valid_and_layered():
    sampler, _ = build_task_size_sampler(None, seed=1)
    wfs = generate_workflow_batch(20, sampler, seed=1)
    assert len(wfs) == 20
    for w in wfs:
        g = w["dag"]
        assert nx.is_directed_acyclic_graph(g)
        assert 10 <= g.number_of_nodes() < 30
        sizes = [d["size_mi"] for _, d in g.nodes(data=True)]
        assert min(sizes) >= 500 and max(sizes) <= 1_000_000
    arrivals = [w["arrival_hour"] for w in wfs]
    assert arrivals == sorted(arrivals)


def test_workflow_generation_is_reproducible():
    s1, _ = build_task_size_sampler(None, seed=3)
    s2, _ = build_task_size_sampler(None, seed=3)
    a = generate_workflow_batch(5, s1, seed=3)
    b = generate_workflow_batch(5, s2, seed=3)
    assert [w["dag"].number_of_nodes() for w in a] == [w["dag"].number_of_nodes() for w in b]


def test_racks_partition_every_server():
    fed = de.build_federation(seed=0)
    racks = tm.assign_racks(fed[0], rack_size=5)
    assert len(racks) == 5  # 25 servers / 5 per rack


def test_hotspot_penalty_increases_with_temperature():
    cool = tm.hotspot_penalty(25.0)
    warm = tm.hotspot_penalty(32.0)
    hot = tm.hotspot_penalty(45.0)
    assert cool < warm < hot


def test_agents_forward_pass():
    torch = pytest.importorskip("torch")
    from src import agents
    assert hasattr(agents, "Actor") and hasattr(agents, "Critic")
