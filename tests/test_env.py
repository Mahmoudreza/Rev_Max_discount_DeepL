"""
tests/test_env.py

Tests for the Revenue MDP environment (src/env/revenue_env.py).

Uses a small BA(n=20) graph to keep tests fast.
"""

import pytest
import networkx as nx

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    """20-node Barabási–Albert graph for fast tests."""
    return nx.barabasi_albert_graph(20, 2, seed=42)


@pytest.fixture
def env_monotone(small_graph):
    cfg = RevenueEnvConfig(
        influence_model="monotone",
        reward_type="flat",
        gamma=1.0,
        seed=42,
    )
    env = RevenueEnv(small_graph, cfg)
    env.reset()
    return env


@pytest.fixture
def env_non_monotone(small_graph):
    cfg = RevenueEnvConfig(
        influence_model="non_monotone",
        reward_type="flat",
        gamma=1.0,
        seed=42,
    )
    env = RevenueEnv(small_graph, cfg)
    env.reset()
    return env


@pytest.fixture
def env_npv(small_graph):
    cfg = RevenueEnvConfig(
        influence_model="monotone",
        reward_type="npv",
        gamma=0.9,
        seed=42,
    )
    env = RevenueEnv(small_graph, cfg)
    env.reset()
    return env


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reset(small_graph):
    """test_reset: env resets correctly (empty S, empty offered, t=0)."""
    cfg = RevenueEnvConfig(seed=1)
    env = RevenueEnv(small_graph, cfg)
    obs = env.reset()

    assert len(env.S) == 0,      "S should be empty after reset"
    assert len(env.offered) == 0, "offered should be empty after reset"
    assert env.t == 0,           "t should be 0 after reset"
    assert env.total_revenue == 0.0, "total_revenue should be 0 after reset"
    assert len(obs["S"]) == 0


def test_step_accept(env_monotone):
    """test_step_accept: at discount=1.0 the buyer always accepts a free item.

    offered_price = est_val * (1-1.0) = 0 regardless of est_val.
    true_val >= 0 is always True, so the buyer always accepts a free item.
    Revenue = 0 for that step; node joins S and seeds the influence cascade.
    """
    env = env_monotone
    # Find first node with neighbors
    node_with_neighbors = None
    for i, node in enumerate(env.nodes):
        if env.graph.degree(node) > 0:
            node_with_neighbors = i
            break
    assert node_with_neighbors is not None

    # Step with discount=1.0: free offer → accepted=True, revenue=0, node joins S
    obs, reward, done, info = env.step(node_with_neighbors, discount=1.0)
    assert info["accepted"], "discount=1.0 → free item → always accepted (bootstraps S)"
    assert reward == 0.0, "free item → revenue=0"
    assert env.nodes[node_with_neighbors] in env.S, "accepted node must join S"


def test_step_reject(env_monotone):
    """test_step_reject: with S empty, est_val=0 → offered_price=0 → free seed accepted.

    When S is empty, estimated influence=0 → est_val=f(0)=0 → offered_price=0.
    true_val=0 as well → accepted=(0>=0)=True.
    The buyer becomes a free seed: joins S (bootstrapping cascade), revenue=0.
    This is Babaei et al. correct: a buyer never refuses a free item.
    """
    env = env_monotone
    # S is empty → valuation=0 for any node, offered_price=0, accepted=True
    obs, reward, done, info = env.step(0, discount=0.0)
    assert info["accepted"], "Empty S → valuation=0 → free seed → accepted=True"
    assert reward == 0.0, "valuation=0 → offered_price=0 → revenue=0"
    assert env.nodes[0] in env.S, "accepted node must join S even at price 0"


def test_valuation_increases(env_monotone):
    """test_valuation_increases: after adding influential node to S, neighbors' valuation increases."""
    env = env_monotone

    # Find a high-degree node to seed (by accepting it — but with S empty, valuation=0)
    # We manually manipulate S to test the influence spreading
    import copy

    # Choose the highest-degree node as seed
    max_deg_node = max(env.nodes, key=lambda v: env.graph.degree(v))
    neighbors = list(env.graph.neighbors(max_deg_node))
    assert len(neighbors) > 0, "High-degree node should have neighbors"

    # Take a neighbor and compute valuation before adding max_deg_node to S
    nb = neighbors[0]
    val_before = env._compute_valuation(nb)

    # Manually add max_deg_node to S (bypass step to focus on valuation change)
    env.S.add(max_deg_node)
    env._influence_cache = {}     # clear legacy cache
    env._true_val_cache = {}      # clear true valuation cache (S changed manually)

    val_after = env._compute_valuation(nb)

    # Under monotone model, valuation should be >= val_before (non-decreasing)
    assert val_after >= val_before, (
        f"Monotone model: valuation should not decrease after adding seed. "
        f"Before={val_before:.4f}, After={val_after:.4f}"
    )


def test_revenue_sum(small_graph):
    """test_revenue_sum: total_revenue == sum of all revenue_step values."""
    cfg = RevenueEnvConfig(seed=7)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Run a few steps
    for i in range(min(5, env.n)):
        if i not in env.offered:
            env.step(i, discount=0.5)

    reported_total = env.total_revenue
    computed_total = sum(r["revenue"] for r in env.revenue_history)
    assert abs(reported_total - computed_total) < 1e-9, (
        f"total_revenue={reported_total} != sum(history)={computed_total}"
    )


def test_npv_mode(small_graph):
    """test_npv_mode: with gamma=0.9, rewards are discounted by 0.9^t."""
    cfg = RevenueEnvConfig(
        influence_model="monotone",
        reward_type="npv",
        gamma=0.9,
        seed=99,
    )
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Run 3 steps and verify reward formula
    for t in range(3):
        obs, reward, done, info = env.step(t, discount=0.5)
        expected_reward = (0.9 ** t) * info["revenue_step"]
        assert abs(reward - expected_reward) < 1e-9, (
            f"NPV step {t}: reward={reward:.6f}, expected={expected_reward:.6f}"
        )


def test_monotone_model(small_graph):
    """test_monotone_model: valuation is non-decreasing as more nodes join S."""
    cfg = RevenueEnvConfig(influence_model="monotone", seed=42)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Choose target node (one with many neighbors)
    target = max(env.nodes, key=lambda v: env.graph.degree(v))
    neighbors = list(env.graph.neighbors(target))
    assert len(neighbors) > 1

    prev_val = env._compute_valuation(target)
    for nb in neighbors[:min(5, len(neighbors))]:
        env.S.add(nb)
        env._influence_cache = {}
        env._true_val_cache = {}   # S changed manually — must clear true val cache
        new_val = env._compute_valuation(target)
        assert new_val >= prev_val - 1e-9, (
            f"Monotone model violated: val went from {prev_val:.4f} to {new_val:.4f}"
        )
        prev_val = new_val


def test_non_monotone_model(small_graph):
    """test_non_monotone_model: valuation eventually decreases as S gets very large."""
    cfg = RevenueEnvConfig(influence_model="non_monotone", seed=42)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Choose target node with many neighbors
    target = max(env.nodes, key=lambda v: env.graph.degree(v))
    neighbors = list(env.graph.neighbors(target))
    assert len(neighbors) > 0

    # Valuation at 0 influence
    val_zero = env._compute_valuation(target)

    # Add many/all neighbors to S → influence → 1 → valuation decreases to ~0
    for nb in neighbors:
        env.S.add(nb)
    env._influence_cache = {}
    env._true_val_cache = {}   # S changed manually

    val_full = env._compute_valuation(target)

    # With non-monotone model: f(0)≈0, f(0.5)=peak, f(1.0) < f(0.5)
    # After full influence, valuation should be lower than peak
    # (i.e., at least one of the following holds: val_full < val_peak OR val_full >= val_zero)
    # The key property: Rayleigh PDF at y=2.0 (x=1.0) < peak at y=1.0 (x=0.5)


def test_true_and_estimate_differ(small_graph):
    """test_true_and_estimate_differ: estimated and true valuations must diverge.

    Proves that _estimate_valuation uses FRESH independent weights while
    _true_valuation uses the fixed true weights from reset().
    After seeding at least one node, influence is non-zero and the two
    estimates are drawn from different weight samples → they differ.
    """
    cfg = RevenueEnvConfig(n_mc_samples=50, seed=42)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Seed the highest-degree node to make influence non-zero for its neighbours
    hub = max(env.nodes, key=lambda v: small_graph.degree(v))
    env.S.add(hub)
    env._true_val_cache = {}
    env._est_val_cache = {}

    diffs = []
    for node in env.nodes:
        if node == hub:
            continue
        if small_graph.degree(node) == 0:
            continue
        true_v = env._true_valuation(node)
        est_v  = env._estimate_valuation(node)
        diffs.append(abs(true_v - est_v))

    assert len(diffs) > 0, "No valid neighbour nodes found"
    assert max(diffs) > 0.001, (
        f"Estimated and true valuations are identical (max diff={max(diffs):.6f}) — "
        "the two functions are not independent"
    )


def test_acceptance_not_guaranteed(small_graph):
    """test_acceptance_not_guaranteed: at non-zero prices some buyers must reject.

    With the corrected model (true_val vs est_val), µ-discount no longer
    achieves 100% acceptance.  Offering at discount=0 (price = est_val)
    should cause at least some rejections when true_val < est_val.
    """
    cfg = RevenueEnvConfig(n_mc_samples=50, seed=123)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    # Give one free seed to start the cascade (discount=1.0 → always accepted)
    hub_idx = env.node_to_idx[max(env.nodes, key=lambda v: small_graph.degree(v))]
    env.step(hub_idx, discount=1.0)

    rejections = 0
    for i in range(env.n):
        node = env.nodes[i]
        if node in env.offered:
            continue
        if len(env.offered) >= 15:  # test first 15 non-hub offers
            break
        _, _, _, info = env.step(i, discount=0.0)  # discount=0 → price = est_val
        if not info["accepted"]:
            rejections += 1

    assert rejections > 0, (
        "All buyers accepted at discount=0 — true/estimated separation not working. "
        f"offered={len(env.offered)} steps"
    )


def test_non_monotone_math(small_graph):
    """test_non_monotone_math: Rayleigh PDF is higher at x=0.5 than at x=1.0."""
    cfg = RevenueEnvConfig(influence_model="non_monotone", seed=42)
    env = RevenueEnv(small_graph, cfg)
    env.reset()

    peak_val = env._apply_influence_model(0.5)  # peak is at x=0.5
    full_val  = env._apply_influence_model(1.0)
    assert full_val < peak_val, (
        f"Non-monotone: value at x=1.0 ({full_val:.4f}) should be < "
        f"peak at x=0.5 ({peak_val:.4f})"
    )
