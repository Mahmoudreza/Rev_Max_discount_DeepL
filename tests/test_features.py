"""
tests/test_features.py

Tests for the 20-dim node feature computation (src/utils/features.py).
"""

import pytest
import numpy as np
import networkx as nx

from src.utils.features import compute_static_features, compute_node_features
from src.env.revenue_env import RevenueEnv, RevenueEnvConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    return nx.barabasi_albert_graph(20, 2, seed=42)


@pytest.fixture
def env_and_static(small_graph):
    cfg = RevenueEnvConfig(seed=42)
    env = RevenueEnv(small_graph, cfg)
    env.reset()
    static = compute_static_features(small_graph)
    return env, static, small_graph


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_feature_shape(env_and_static):
    """test_feature_shape: compute_node_features returns shape (n, 20)."""
    env, static, graph = env_and_static
    n = graph.number_of_nodes()
    features = compute_node_features(
        graph=graph,
        static_features=static,
        S=frozenset(env.S),
        offered=frozenset(env.offered),
        t=env.t,
        n=n,
        k=n,
        env=env,
    )
    assert features.shape == (n, 20), (
        f"Expected shape ({n}, 20), got {features.shape}"
    )


def test_static_features_cached(small_graph):
    """test_static_features_cached: calling compute_static_features twice gives same result."""
    static1 = compute_static_features(small_graph)
    static2 = compute_static_features(small_graph)
    for node in small_graph.nodes():
        np.testing.assert_array_almost_equal(
            static1[node], static2[node],
            err_msg=f"Static features for node {node} differ between calls"
        )


def test_current_influence_zero_at_start(env_and_static):
    """test_current_influence_zero_at_start: dim 16 = 0 for all nodes when S is empty."""
    env, static, graph = env_and_static
    n = graph.number_of_nodes()
    # S is empty after reset
    features = compute_node_features(
        graph=graph,
        static_features=static,
        S=frozenset(),
        offered=frozenset(),
        t=0,
        n=n,
        k=n,
        env=env,
    )
    # dim 16 = current_influence (should be 0 when S is empty)
    np.testing.assert_array_equal(
        features[:, 16], 0.0,
        err_msg="current_influence (dim 16) should be 0 when S is empty"
    )


def test_influence_updates_after_step(env_and_static):
    """test_influence_updates_after_step: dim 16 increases for neighbors after a buyer joins S."""
    env, static, graph = env_and_static
    n = graph.number_of_nodes()

    # Add the highest-degree node to S
    max_deg_node = max(graph.nodes(), key=lambda v: graph.degree(v))
    neighbors = list(graph.neighbors(max_deg_node))
    assert len(neighbors) > 0

    # Compute features with S = {max_deg_node}
    env.S.add(max_deg_node)
    env._influence_cache = {}

    features_after = compute_node_features(
        graph=graph,
        static_features=static,
        S=frozenset(env.S),
        offered=frozenset(),
        t=1,
        n=n,
        k=n,
        env=env,
    )

    # At least one neighbor should have nonzero current_influence (dim 16)
    nodes_list = list(graph.nodes())
    neighbor_indices = [nodes_list.index(nb) for nb in neighbors if nb in nodes_list]
    neighbor_influences = features_after[neighbor_indices, 16]
    assert any(v > 0.0 for v in neighbor_influences), (
        "At least one neighbor should have nonzero current_influence after seed joins"
    )


def test_was_offered_flag(env_and_static):
    """test_was_offered_flag: dim 18 = 1 for nodes in offered set, 0 otherwise."""
    env, static, graph = env_and_static
    n = graph.number_of_nodes()
    nodes_list = list(graph.nodes())

    # Mark first 3 nodes as offered
    offered_nodes = set(nodes_list[:3])

    features = compute_node_features(
        graph=graph,
        static_features=static,
        S=frozenset(),
        offered=frozenset(offered_nodes),
        t=3,
        n=n,
        k=n,
        env=env,
    )

    # Check was_offered (dim 18)
    for i, node in enumerate(nodes_list):
        expected = 1.0 if node in offered_nodes else 0.0
        assert features[i, 18] == pytest.approx(expected), (
            f"Node {node}: was_offered dim={features[i, 18]}, expected={expected}"
        )


def test_all_features_in_range(env_and_static):
    """test_all_features_in_range: all feature values are finite (no NaN, no inf)."""
    env, static, graph = env_and_static
    n = graph.number_of_nodes()

    # Add a few nodes to S for more realistic features
    nodes_list = list(graph.nodes())
    for node in nodes_list[:3]:
        env.S.add(node)
    env._influence_cache = {}

    features = compute_node_features(
        graph=graph,
        static_features=static,
        S=frozenset(env.S),
        offered=frozenset(nodes_list[:3]),
        t=3,
        n=n,
        k=n,
        env=env,
    )

    assert np.all(np.isfinite(features)), (
        f"Features contain NaN or inf:\n"
        f"NaN count: {np.isnan(features).sum()}\n"
        f"Inf count: {np.isinf(features).sum()}"
    )
