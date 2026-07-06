"""src/utils/budget_features.py — Node features for budget-constrained model (Idea 3).

Extends the 20-dim Idea 1 features with 1 budget dimension:
  dims 0–19: standard features from src/utils/features.py
  dim    20: B_t / B_0 (remaining budget as fraction of initial)

Total: 21 dimensions. The extra dimension teaches the policy to be
budget-aware: it should be more conservative (higher prices) when B is low,
and can afford discounts when B is high.

Usage:
    features = compute_budget_node_features(
        graph, static_feats, env.S, env.offered, env.t, env.n,
        k=cfg.budget.k, env=env
    )
    # shape: (n, 21)
"""

import numpy as np
from src.utils.features import compute_static_features, compute_node_features


def compute_budget_node_features(
    graph,
    static_features: np.ndarray,
    S: frozenset,
    offered: frozenset,
    t: int,
    n: int,
    k: int,
    env,
    group_labels=None,
) -> np.ndarray:
    """Compute 21-dim node features: 20 from Idea 1 + 1 budget fraction.

    The budget fraction dim is IDENTICAL for all nodes at a given timestep
    (it is a global property of the episode state, not node-specific).
    Despite being constant across nodes, it provides a gradient signal:
    the LSTM can change behaviour based on how budget-constrained the
    company currently is.

    Args:
        graph:           NetworkX graph.
        static_features: Pre-computed static features (shape n × k_static),
                         e.g. from compute_static_features().
        S:               Current seed set (frozenset of node labels).
        offered:         Already-offered nodes (frozenset of node labels).
        t:               Current step count.
        n:               Total number of nodes in graph.
        k:               Seed budget (cfg.budget.k, used for normalisation).
        env:             BudgetRevenueEnv instance — provides .budget_fraction.
        group_labels:    Optional group labels for group-aware features.

    Returns:
        np.ndarray of shape (n, 21): standard 20-dim features + budget col.
    """
    # Base 20-dim features from Idea 1
    base_features = compute_node_features(
        graph, static_features, S, offered, t, n, k,
        env, group_labels,
    )  # shape: (n, 20)

    # Budget fraction: scalar, broadcast to all nodes
    budget_frac = float(env.budget_fraction)
    budget_col  = np.full((n, 1), budget_frac, dtype=np.float32)

    return np.concatenate([base_features, budget_col], axis=1)  # (n, 21)
