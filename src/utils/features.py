"""
src/utils/features.py

Node feature computation for the revenue maximization task.
Extends WSDM 2027's 16-dim feature vector to 20-dim by adding
4 pricing-specific features.

WSDM features (dims 1-16):
  Static (1-10):  deg, cc, bc, pr, kc, ec, tc, cl, ecc, and
  Dynamic (11-16): seed_flag, round_ratio, hop1_seed_frac,
                   log_degree, cluster_repeat, group_flag

NEW pricing features (dims 17-20):
  current_influence  - normalized influence on v from S
  current_valuation  - f(current_influence) under Rayleigh model
  was_offered        - 1 if v already offered (and rejected)
  steps_remaining    - (n - t) / n
"""

import numpy as np
import networkx as nx
from typing import Dict, Optional


# ── Precomputed static features (once per graph) ─────────────────────────────

def compute_static_features(graph: nx.Graph) -> Dict:
    """Compute all static structural features for every node.

    These are computed once per graph and reused across all episode steps.
    Matches WSDM 2027 Section 4.1 exactly (dims 1-10).

    Args:
        graph: NetworkX graph.

    Returns:
        Dict mapping node → np.ndarray of shape (10,).
    """
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())

    # Degree rank (normalized)
    degrees = dict(graph.degree())
    max_deg = max(degrees.values()) if degrees else 1
    deg_rank = {v: degrees[v] / max_deg for v in nodes}

    # Clustering coefficient
    cc = nx.clustering(graph)

    # Betweenness centrality (normalized by default in networkx)
    bc = nx.betweenness_centrality(graph, normalized=True)

    # PageRank
    pr = nx.pagerank(graph, alpha=0.85)

    # K-core number
    kc_raw = nx.core_number(graph)
    max_kc = max(kc_raw.values()) if kc_raw else 1
    kc = {v: kc_raw[v] / max_kc for v in nodes}

    # Eigenvector centrality — use numpy LAPACK solver for speed (O(n^3) but
    # negligible vs power iteration for n < 2000). Falls back gracefully.
    try:
        ec = nx.eigenvector_centrality_numpy(graph)
    except Exception:
        try:
            ec = nx.eigenvector_centrality(graph, max_iter=100, tol=1e-4)
        except nx.PowerIterationFailedConvergence:
            ec = {v: 0.0 for v in nodes}

    # Triangle count (normalized)
    tri_raw = nx.triangles(graph)
    max_tri = max(tri_raw.values()) if tri_raw else 1
    tc = {v: tri_raw[v] / max_tri for v in nodes}

    # Closeness centrality
    cl = nx.closeness_centrality(graph)

    # Eccentricity (only for connected graphs; use 0 otherwise)
    try:
        ecc_raw = nx.eccentricity(graph)
        max_ecc = max(ecc_raw.values()) if ecc_raw else 1
        ecc = {v: ecc_raw[v] / max_ecc for v in nodes}
    except nx.NetworkXError:
        ecc = {v: 0.0 for v in nodes}

    # Average neighbor degree
    and_raw = nx.average_neighbor_degree(graph)
    max_and = max(and_raw.values()) if and_raw else 1
    and_ = {v: and_raw[v] / max_and for v in nodes}

    static = {}
    for v in nodes:
        static[v] = np.array([
            deg_rank[v],
            cc[v],
            bc[v],
            pr[v],
            kc[v],
            ec[v],
            tc[v],
            cl[v],
            ecc[v],
            and_[v],
        ], dtype=np.float32)

    return static


# ── Graph-level feature cache (precompute once per graph) ────────────────────

def build_graph_feature_cache(graph: nx.Graph, static_features: Dict) -> Dict:
    """Pre-build all graph-level arrays that are constant across episode steps.

    Call ONCE per graph before the training loop.  The returned cache is passed
    to compute_node_features_fast() at each step instead of recomputing.

    Returns dict with keys:
        nodes, n, static_matrix, neighbor_idx, deg_log_norm
    """
    nodes = list(graph.nodes())
    n = len(nodes)
    node_pos = {v: i for i, v in enumerate(nodes)}

    # (n, 10) static feature matrix in positional order
    static_matrix = np.array([static_features[v] for v in nodes], dtype=np.float32)

    # neighbor list as positional int arrays (for vectorized hop-1 computation)
    neighbor_idx = [np.array([node_pos[nb] for nb in graph.neighbors(v)], dtype=np.int32)
                    for v in nodes]

    # normalized log-degree (dim 13)
    degrees = np.array([graph.degree(v) for v in nodes], dtype=np.float32)
    max_log_deg = float(np.log1p(degrees.max())) if len(degrees) else 1.0
    deg_log_norm = np.log1p(degrees) / max(max_log_deg, 1e-8)

    return {
        "nodes": nodes,
        "node_pos": node_pos,
        "n": n,
        "static_matrix": static_matrix,
        "neighbor_idx": neighbor_idx,
        "deg_log_norm": deg_log_norm,
    }


def compute_node_features_fast(
    cache: Dict,
    S: frozenset,
    offered: frozenset,
    t: int,
    k: int,
    env,
    group_labels: Optional[Dict] = None,
) -> np.ndarray:
    """Vectorized 20-dim feature computation using precomputed graph cache.

    ~5–8× faster than compute_node_features() for large n.

    Args:
        cache:   Output of build_graph_feature_cache().
        S, offered, t, k, env, group_labels: same as compute_node_features().

    Returns:
        np.ndarray of shape (n, 20).
    """
    nodes = cache["nodes"]
    n = cache["n"]
    static_matrix = cache["static_matrix"]
    neighbor_idx = cache["neighbor_idx"]
    deg_log_norm = cache["deg_log_norm"]

    features = np.empty((n, 20), dtype=np.float32)
    features[:, :10] = static_matrix

    # Boolean masks using set lookups → numpy array (O(n))
    S_set = set(S); off_set = set(offered)
    S_mask = np.array([v in S_set for v in nodes], dtype=bool)
    off_mask = np.array([v in off_set for v in nodes], dtype=bool)

    features[:, 10] = S_mask.astype(np.float32)
    features[:, 11] = t / max(k, 1)

    # hop-1 seed fraction: vectorized over precomputed neighbor arrays
    hop1 = np.zeros(n, dtype=np.float32)
    for i, nbrs in enumerate(neighbor_idx):
        if len(nbrs):
            hop1[i] = S_mask[nbrs].sum() / len(nbrs)
    features[:, 12] = hop1

    features[:, 13] = deg_log_norm
    features[:, 14] = static_matrix[:, 1]          # cc (already stored)

    if group_labels is not None:
        features[:, 15] = np.array([float(group_labels.get(v, 0.5)) for v in nodes])
    else:
        features[:, 15] = 0.5

    # dims 16-17: env influence (small loop, ~0.003ms/call)
    for i, v in enumerate(nodes):
        ci = env.get_current_influence(v)
        features[i, 16] = ci
        features[i, 17] = env._apply_influence_model(ci)

    features[:, 18] = off_mask.astype(np.float32)
    features[:, 19] = max(0.0, (n - t) / n)
    return features


# ── Full 20-dim feature vector (called at every step) ────────────────────────

def compute_node_features(
    graph: nx.Graph,
    static_features: Dict,
    S: frozenset,
    offered: frozenset,
    t: int,
    n: int,
    k: int,
    env,                        # RevenueEnv instance (for influence/valuation)
    group_labels: Optional[Dict] = None,
) -> np.ndarray:
    """Compute the full 20-dim feature matrix for all nodes at step t.

    Matches WSDM 2027 Section 4.1 (dims 1-16) and adds 4 pricing dims (17-20).

    Args:
        graph: NetworkX graph.
        static_features: Precomputed static features dict (from compute_static_features).
        S: Current seed set (buyers who purchased).
        offered: Set of nodes already offered.
        t: Current episode step.
        n: Total number of nodes.
        k: Budget (used for round_ratio; set to n for full-episode RL).
        env: RevenueEnv instance for current_influence and current_valuation.
        group_labels: Optional dict mapping node → {0, 1} (minority/majority).

    Returns:
        np.ndarray of shape (n, 20).
    """
    nodes = list(graph.nodes())
    features = np.zeros((n, 20), dtype=np.float32)

    for i, v in enumerate(nodes):
        # ── Static dims 0-9 (from WSDM) ────────────────────────────────────
        features[i, :10] = static_features[v]

        # ── Dynamic dims 10-15 (from WSDM) ─────────────────────────────────
        # dim 10: seed membership
        features[i, 10] = 1.0 if v in S else 0.0

        # dim 11: round ratio t/k
        features[i, 11] = t / max(k, 1)

        # dim 12: hop-1 seed fraction
        neighbors = list(graph.neighbors(v))
        if neighbors:
            seed_neighbors = sum(1 for nb in neighbors if nb in S)
            features[i, 12] = seed_neighbors / len(neighbors)

        # dim 13: log-degree normalization
        deg = graph.degree(v)
        features[i, 13] = np.log1p(deg) / np.log1p(max(dict(graph.degree()).values()))

        # dim 14: clustering coefficient (repeated as diffusion signal)
        features[i, 14] = static_features[v][1]   # cc is dim 1

        # dim 15: group membership (0=minority, 1=majority; 0.5 if unknown)
        if group_labels is not None and v in group_labels:
            features[i, 15] = float(group_labels[v])
        else:
            features[i, 15] = 0.5

        # ── NEW pricing dims 16-19 ───────────────────────────────────────────
        # dim 16: current normalized influence on v from S
        current_influence = env.get_current_influence(v)
        features[i, 16] = current_influence

        # dim 17: current valuation estimate f(current_influence)
        features[i, 17] = env._apply_influence_model(current_influence)

        # dim 18: was_offered — 1 if v already received an offer (accepted or not)
        features[i, 18] = 1.0 if v in offered else 0.0

        # dim 19: steps remaining fraction
        features[i, 19] = max(0.0, (n - t) / n)

    return features
