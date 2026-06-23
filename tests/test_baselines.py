"""
tests/test_baselines.py

Sanity checks for Babaei et al. (2013) baseline strategies.
"""

import pytest
import networkx as nx
from omegaconf import OmegaConf

from src.evaluation.baselines import (
    ie_strategy,
    mu_discount,
    greedy_discount,
    sigma_discount,
    greedy_discount_trajectory,
    run_all_baselines,
)


@pytest.fixture
def small_graph():
    """25-node BA graph for fast baseline tests."""
    return nx.barabasi_albert_graph(25, 3, seed=42)


@pytest.fixture
def cfg():
    """Load base config."""
    return OmegaConf.load("configs/base_config.yaml")


def test_ie_strategy_nonnegative(small_graph, cfg):
    """ie_strategy returns non-negative revenue."""
    rev = ie_strategy(small_graph, cfg)
    assert rev >= 0.0, f"IE-strategy revenue={rev} should be >= 0"


def test_mu_discount_nonnegative(small_graph, cfg):
    """mu_discount returns non-negative revenue."""
    rev = mu_discount(small_graph, cfg)
    assert rev >= 0.0, f"µ-discount revenue={rev} should be >= 0"


def test_greedy_discount_nonnegative(small_graph, cfg):
    """greedy_discount returns non-negative revenue."""
    rev = greedy_discount(small_graph, cfg)
    assert rev >= 0.0, f"greedy_discount revenue={rev} should be >= 0"


def test_sigma_discount_nonnegative(small_graph, cfg):
    """sigma_discount returns non-negative revenue."""
    rev = sigma_discount(small_graph, cfg)
    assert rev >= 0.0, f"σ-discount revenue={rev} should be >= 0"


def test_trajectory_length(small_graph, cfg):
    """greedy_discount_trajectory returns n steps."""
    traj = greedy_discount_trajectory(small_graph, cfg)
    n = small_graph.number_of_nodes()
    assert len(traj) == n, f"Trajectory length={len(traj)}, expected {n}"


def test_trajectory_discount_in_range(small_graph, cfg):
    """All trajectory discounts are in [0, 1]."""
    traj = greedy_discount_trajectory(small_graph, cfg)
    for step, (node_idx, discount, marginal) in enumerate(traj):
        assert 0.0 <= discount <= 1.0, (
            f"Step {step}: discount={discount} out of [0,1]"
        )


def test_trajectory_node_indices_unique(small_graph, cfg):
    """Each node appears exactly once in the greedy_discount trajectory."""
    traj = greedy_discount_trajectory(small_graph, cfg)
    node_indices = [t[0] for t in traj]
    assert len(node_indices) == len(set(node_indices)), (
        "Duplicate node indices in trajectory"
    )


def test_run_all_baselines_keys(small_graph, cfg):
    """run_all_baselines returns all 4 strategy keys."""
    results = run_all_baselines(small_graph, cfg, n_trials=2)
    expected_keys = {"ie_strategy", "mu_discount", "greedy_discount", "sigma_discount"}
    assert set(results.keys()) == expected_keys


def test_run_all_baselines_nonnegative(small_graph, cfg):
    """All baseline revenues are non-negative."""
    results = run_all_baselines(small_graph, cfg, n_trials=2)
    for name, rev in results.items():
        assert rev >= 0.0, f"{name} revenue={rev} should be >= 0"


def test_greedy_beats_ie(small_graph, cfg):
    """greedy_discount should generally produce higher revenue than ie_strategy.

    The IE-Strategy sacrifices all revenue from the seed set (free giveaway),
    so greedy_discount should typically outperform it.
    """
    # Average over a few trials for robustness
    from src.evaluation.baselines import _override_seed
    gd_revs = []
    ie_revs = []
    for i in range(3):
        trial_cfg = _override_seed(cfg, 42 + i)
        gd_revs.append(greedy_discount(small_graph, trial_cfg))
        ie_revs.append(ie_strategy(small_graph, trial_cfg))
    avg_gd = sum(gd_revs) / len(gd_revs)
    avg_ie = sum(ie_revs) / len(ie_revs)
    # Greedy-discount should beat IE (which gives items away for free)
    assert avg_gd >= avg_ie, (
        f"greedy_discount avg={avg_gd:.4f} should be >= ie_strategy avg={avg_ie:.4f}"
    )
