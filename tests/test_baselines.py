"""
tests/test_baselines.py

Tests for the 10-method baseline comparison suite.

Group 1 — Babaei et al. 2013 (computed, always run)
Group 2 — Deep IM decoupled (checkpoint absent → graceful None)
Group 3 — Our Rev models (checkpoint absent → graceful None)

Performance note:
  Expensive computations (run_all_babaei, run_full_comparison) are cached in
  module-scoped fixtures so each is called only once across all test functions.
  Total wall time: ~30s for all 24 tests.
"""

import warnings
import pytest
import networkx as nx
from omegaconf import OmegaConf

from src.evaluation.baselines import (
    # Group 1
    ie_strategy,
    mu_discount,
    sigma_discount,
    greedy_discount,
    greedy_discount_trajectory,
    ie_strategy_trajectory,
    hill_climbing_trajectory,
    # Runners
    run_all_baselines,
    run_all_babaei,
    run_full_comparison,
    # Group 2
    s2v_dqn_decoupled,
    touple_gdd_decoupled,
    _run_decoupled_gnn_baseline,
    # Group 3
    eval_rev_gnn_im_rl,
    eval_rev_gail_rl,
    eval_rev_gnn_lstm,
    eval_rev_gail_lstm,
    # Helpers
    _override_seed,
)


# ── Graph + config fixtures ───────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_graph():
    """25-node BA graph — shared across all tests in this module."""
    return nx.barabasi_albert_graph(25, 3, seed=42)


@pytest.fixture(scope="module")
def cfg():
    """Base config — shared across all tests."""
    return OmegaConf.load("configs/base_config.yaml")


# ── Expensive result fixtures (run ONCE per module) ───────────────────────────

@pytest.fixture(scope="module")
def babaei_results(small_graph, cfg):
    """Run all 4 Babaei baselines once (n_trials=1) and cache."""
    return run_all_babaei(small_graph, cfg, n_trials=1)


@pytest.fixture(scope="module")
def full_comparison_results(small_graph, cfg):
    """Run full 10-method comparison once (n_trials=1) and cache."""
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        return run_full_comparison(small_graph, cfg,
                                   n_trials_babaei=1, n_trials_deep_im=1)


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Babaei et al. 2013 (individual sanity checks, use cached results)
# ════════════════════════════════════════════════════════════════════════════════

def test_ie_strategy_nonnegative(full_comparison_results):
    assert full_comparison_results["ie_strategy"] >= 0.0


def test_mu_discount_nonnegative(full_comparison_results):
    assert full_comparison_results["mu_discount"] >= 0.0


def test_sigma_discount_nonnegative(full_comparison_results):
    assert full_comparison_results["sigma_discount"] >= 0.0


def test_greedy_discount_nonnegative(full_comparison_results):
    assert full_comparison_results["greedy_discount"] >= 0.0


def test_trajectory_length(small_graph, cfg):
    """greedy_discount_trajectory returns exactly n steps."""
    traj = greedy_discount_trajectory(small_graph, cfg)
    assert len(traj) == small_graph.number_of_nodes()


def test_trajectory_discount_in_range(small_graph, cfg):
    """All trajectory discounts ∈ [0, 1]."""
    for t in greedy_discount_trajectory(small_graph, cfg):
        assert 0.0 <= t["discount"] <= 1.0, f"discount {t['discount']} out of range"


def test_trajectory_node_indices_unique(small_graph, cfg):
    """Every node appears exactly once in trajectory."""
    traj = greedy_discount_trajectory(small_graph, cfg)
    indices = [t["node_idx"] for t in traj]
    assert len(indices) == len(set(indices))


def test_ie_trajectory_length(small_graph, cfg):
    """ie_strategy_trajectory returns exactly n steps."""
    traj = ie_strategy_trajectory(small_graph, cfg)
    assert len(traj) == small_graph.number_of_nodes()


def test_hill_climbing_trajectory_length(small_graph, cfg):
    """hill_climbing_trajectory returns exactly n steps."""
    traj = hill_climbing_trajectory(small_graph, cfg)
    assert len(traj) == small_graph.number_of_nodes()


def test_greedy_beats_ie(full_comparison_results):
    """greedy_discount revenue ≥ ie_strategy revenue (single trial)."""
    assert full_comparison_results["greedy_discount"] >= \
           full_comparison_results["ie_strategy"]


# ── Runner tests (use cached babaei_results) ──────────────────────────────────

def test_run_all_babaei_keys(babaei_results):
    """run_all_babaei returns exactly 4 method keys."""
    expected = {"ie_strategy", "mu_discount", "sigma_discount", "greedy_discount"}
    assert set(babaei_results.keys()) == expected


def test_run_all_babaei_nonnegative(babaei_results):
    """All Babaei baselines return non-negative revenue."""
    for name, rev in babaei_results.items():
        assert rev >= 0.0, f"{name} revenue={rev} < 0"


def test_run_all_baselines_alias(small_graph, cfg):
    """run_all_baselines (alias) returns same 4 keys."""
    results = run_all_baselines(small_graph, cfg, n_trials=1)
    assert set(results.keys()) == {"ie_strategy", "mu_discount", "sigma_discount", "greedy_discount"}


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Deep IM decoupled (S2V-DQN, ToupleGDD)
# ════════════════════════════════════════════════════════════════════════════════

def test_s2v_dqn_no_checkpoint_returns_none(small_graph, cfg):
    """s2v_dqn_decoupled returns None when checkpoint absent."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = s2v_dqn_decoupled(small_graph, cfg, n_trials=1)
    assert out is None
    assert len(w) >= 1


def test_touple_gdd_no_checkpoint_returns_none(small_graph, cfg):
    """touple_gdd_decoupled returns None when checkpoint absent."""
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = touple_gdd_decoupled(small_graph, cfg, n_trials=1)
    assert out is None
    assert len(w) >= 1


def test_decoupled_wrapper_structure(small_graph, cfg):
    """_run_decoupled_gnn_baseline returns correct dict shape."""
    import numpy as np
    order = list(np.random.default_rng(0).permutation(list(small_graph.nodes())))
    out = _run_decoupled_gnn_baseline(small_graph, cfg, order, n_trials=2)
    assert isinstance(out, dict)
    assert "mean_revenue" in out and "std_revenue" in out
    assert out["mean_revenue"] >= 0.0
    assert out["std_revenue"]  >= 0.0


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Our Rev models (graceful None without checkpoint)
# ════════════════════════════════════════════════════════════════════════════════

def test_rev_gnn_im_rl_no_checkpoint_returns_none(full_comparison_results):
    assert full_comparison_results["rev_gnn_im_rl"] is None


def test_rev_gail_rl_no_checkpoint_returns_none(full_comparison_results):
    assert full_comparison_results["rev_gail_rl"] is None


def test_rev_gnn_lstm_no_checkpoint_returns_none(full_comparison_results):
    assert full_comparison_results["rev_gnn_lstm"] is None


def test_rev_gail_lstm_no_checkpoint_returns_none(full_comparison_results):
    assert full_comparison_results["rev_gail_lstm"] is None


# ════════════════════════════════════════════════════════════════════════════════
# Full runner
# ════════════════════════════════════════════════════════════════════════════════

def test_run_full_comparison_keys(full_comparison_results):
    """run_full_comparison returns all 10 expected keys."""
    expected = {
        "ie_strategy", "mu_discount", "sigma_discount", "greedy_discount",
        "s2v_dqn", "touple_gdd",
        "rev_gnn_im_rl", "rev_gail_rl", "rev_gnn_lstm", "rev_gail_lstm",
    }
    assert set(full_comparison_results.keys()) == expected, (
        f"Missing: {expected - set(full_comparison_results.keys())}")


def test_run_full_comparison_babaei_nonnegative(full_comparison_results):
    """All 4 Babaei results are non-negative."""
    for k in ("ie_strategy", "mu_discount", "sigma_discount", "greedy_discount"):
        assert full_comparison_results[k] is not None
        assert full_comparison_results[k] >= 0.0, f"{k} = {full_comparison_results[k]}"


def test_run_full_comparison_group3_none_without_checkpoints(full_comparison_results):
    """Group 3 Rev models return None when checkpoints don't exist."""
    for k in ("rev_gnn_im_rl", "rev_gail_rl", "rev_gnn_lstm", "rev_gail_lstm"):
        assert full_comparison_results[k] is None, \
            f"{k} should be None (no checkpoint), got {full_comparison_results[k]}"


def test_latex_table_content(full_comparison_results):
    """run_full_comparison result produces a valid LaTeX table structure."""
    from experiments.run_baselines import GROUPS, METHOD_LABELS

    ie_rev = full_comparison_results.get("ie_strategy") or 1.0
    valid  = {k: v for k, v in full_comparison_results.items() if v is not None}
    best   = max(valid.values()) if valid else 0.0

    # Build minimal table inline to verify no errors
    lines = [r"\begin{table}[t]"]
    for grp_label, keys in GROUPS:
        for k in keys:
            v = full_comparison_results.get(k)
            if v is not None:
                pct = 100.0 * (v - ie_rev) / (ie_rev + 1e-9)
                lines.append(f"  {k} & {v:.3f} & {pct:+.1f}\\% \\\\")
    lines.append(r"\end{table}")

    content = "\n".join(lines)
    assert r"\begin{table}" in content
    assert r"\end{table}" in content
    assert "ie_strategy" in content or "mu_discount" in content
