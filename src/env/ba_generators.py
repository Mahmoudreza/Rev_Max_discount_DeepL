"""src/env/ba_generators.py — Barabási–Albert graph generator for topology coverage.

Generates hub-heavy BA graphs to complement Forest-Fire training data.
Target signature: polblogs LCC (n=1222, mean_deg≈27, max/median≈27).
Configs: n ∈ {200,260,320,380,440}, m ∈ {8,12} → 10 (n,m) pairs.

Design choice: m=8 gives mean_deg≈16, m=12 gives mean_deg≈24,
max/median ratio ≈ 15-30 (BA power law; matches polblogs heavy tail).
"""
from __future__ import annotations

import hashlib
import random
from typing import List, Tuple

import networkx as nx
import numpy as np

# ── BA graph configurations ───────────────────────────────────────────────────

SIZES = [200, 260, 320, 380, 440]
M_VALUES = [8, 12]  # attachment edges; m=8 → mean_deg≈16; m=12 → mean_deg≈24

BA_CONFIGS: List[Tuple[int, int]] = [
    (n, m) for n in SIZES for m in M_VALUES
]  # 10 configs in (n, m) order


# ── Graph generation ──────────────────────────────────────────────────────────

def generate_ba(n: int, m: int, seed: int = 0) -> nx.Graph:
    """Return a Barabási–Albert random graph with n nodes and m edges per node.

    Edge weights are NOT set here — BudgetRevenueEnv draws U(0,2) per episode.

    Args:
        n:    Number of nodes.
        m:    Number of edges to attach per new node (controls hub degree).
        seed: RNG seed for reproducible topology.

    Returns:
        Undirected NetworkX graph with integer node labels 0..n-1.
    """
    return nx.barabasi_albert_graph(n, m, seed=seed)


def ba_degree_stats(G: nx.Graph) -> dict:
    """Return degree statistics for a graph."""
    degs = sorted(dict(G.degree()).values())
    return {
        "n": G.number_of_nodes(),
        "m_edges": G.number_of_edges(),
        "mean_deg": float(np.mean(degs)),
        "median_deg": float(np.median(degs)),
        "max_deg": int(np.max(degs)),
        "max_over_median": float(np.max(degs) / np.median(degs)),
    }


def make_ba_training_graphs(seed: int = 0) -> List[nx.Graph]:
    """Return one BA graph per config (10 total), fixed topology seed."""
    return [generate_ba(n, m, seed=seed * 100 + i)
            for i, (n, m) in enumerate(BA_CONFIGS)]


def ba_config_label(n: int, m: int) -> str:
    return f"BA_n{n}_m{m}"


def check_feature_anomalies(G: nx.Graph, static_feats: np.ndarray) -> dict:
    """Check for NaN/Inf/degenerate columns in static feature matrix.

    Args:
        G:            NetworkX graph.
        static_feats: (n, d) feature matrix from compute_static_features.

    Returns:
        dict with any anomaly flags; empty if all OK.
    """
    issues = {}
    if np.any(np.isnan(static_feats)):
        cols = np.where(np.any(np.isnan(static_feats), axis=0))[0].tolist()
        issues["nan_cols"] = cols
    if np.any(np.isinf(static_feats)):
        cols = np.where(np.any(np.isinf(static_feats), axis=0))[0].tolist()
        issues["inf_cols"] = cols
    # Check for all-zero columns (degenerate)
    zero_cols = np.where(np.all(static_feats == 0, axis=0))[0].tolist()
    if zero_cols:
        issues["zero_cols"] = zero_cols
    return issues
