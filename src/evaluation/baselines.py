"""
src/evaluation/baselines.py

Baseline strategies for AAAI 2027 paper comparison table.

10 methods in 3 groups:
  Group 1 — Babaei et al. (2013) hand-crafted:
    1. ie_strategy       Influence-and-Exploit
    2. mu_discount        µ-rule
    3. sigma_discount     σ-rule
    4. greedy_discount    Greedy degree-based discount (also GAIL expert)

  Group 2 — Deep IM (decoupled seed selection):
    5. s2v_dqn_decoupled    S2V-DQN (Dai et al. 2017) seeds + greedy pricing
    6. touple_gdd_decoupled ToupleGDD (Chen et al. 2022) seeds + greedy pricing

  Group 3 — Ours (joint seed selection + pricing):
    7.  Rev-GNN-IM-RL    GNN + REINFORCE
    8.  Rev-GAIL-RL      GNN + GAIL
    9.  Rev-GNN-LSTM     GNN + LSTM + REINFORCE
    10. Rev-GAIL-LSTM    GNN + LSTM + GAIL

Groups 2–3 return None gracefully when checkpoints are absent.

References:
  Babaei et al. (2013) ICWSM.
  Dai et al. (2017) NeurIPS. (S2V-DQN)
  Chen et al. (2022) WWW. (ToupleGDD)
"""

import os
import json
import time
import warnings
import tempfile
import subprocess
import numpy as np
import networkx as nx
from typing import Dict, List, Optional, Tuple

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _make_env(graph: nx.Graph, cfg) -> RevenueEnv:
    """Construct a RevenueEnv from an OmegaConf config."""
    env_cfg = RevenueEnvConfig(
        influence_model=cfg.influence.model,
        b=cfg.influence.b,
        weight_low=cfg.influence.weight_low,
        weight_high=cfg.influence.weight_high,
        n_mc_samples=cfg.influence.n_mc_samples,
        reward_type=cfg.reward.type,
        gamma=cfg.reward.gamma,
        seed=cfg.project.seed,
    )
    return RevenueEnv(graph, env_cfg)


def _override_seed(cfg, new_seed: int):
    """Return a new config with project.seed overridden."""
    from omegaconf import OmegaConf
    return OmegaConf.merge(cfg, OmegaConf.create({"project": {"seed": new_seed}}))


def _invalidate_caches(env, node) -> None:
    """Selective cache invalidation: only neighbours of the accepted node.

    Replaces the O(n^2)-per-run full-clear with an O(degree) per-step clear.
    Must be called whenever a node is accepted and added to env.S.

    Args:
        env:  RevenueEnv instance.
        node: The node that was just accepted (joined S).
    """
    for _nb in env.graph.neighbors(node):
        env._influence_cache.pop(_nb, None)
        env._true_val_cache.pop(_nb, None)
        env._est_val_cache.pop(_nb, None)



def _seed_top_k_by_degree(graph: nx.Graph, env: RevenueEnv, k: int) -> List:
    """Give the top-k highest-degree nodes as FREE seeds (revenue = 0).

    Adds them to env.S, env.offered; clears influence cache.
    Used as the seeding phase for µ/σ/greedy-discount baselines so that
    all methods compete under the same seed-budget constraint.

    Args:
        graph: NetworkX graph.
        env:   RevenueEnv (already reset).
        k:     Seed budget.

    Returns:
        List of seeded nodes (in degree-descending order).
    """
    degrees = dict(graph.degree())
    sorted_nodes = sorted(graph.nodes(), key=lambda v: degrees[v], reverse=True)
    seeds = sorted_nodes[:min(k, graph.number_of_nodes())]
    for node in seeds:
        env.S.add(node)
        _invalidate_caches(env, node)
        env.offered.add(node)
        env.t += 1
    return seeds


def _greedy_seed_selection(graph: nx.Graph, env: RevenueEnv, k: int) -> List:
    """Greedy hill-climbing seed selection by marginal influence gain.

    At each step adds the node maximising the incremental influence spread.

    Args:
        graph: NetworkX graph.
        env:   RevenueEnv (already reset — link weights sampled).
        k:     Seed budget.

    Returns:
        List of selected node identifiers.
    """
    S: List = []
    remaining = set(graph.nodes())

    for _ in range(min(k, graph.number_of_nodes())):
        best_node, best_gain = None, -1.0

        for node in remaining:
            gain = 0.0
            for nb in graph.neighbors(node):
                if nb in remaining and nb not in S:
                    infl_before = env.get_current_influence(nb)
                    w = env._link_weights.get((nb, node), 0.0)
                    total_w = sum(env._link_weights.get((nb, n2), 0.0)
                                  for n2 in graph.neighbors(nb))
                    if total_w > 0:
                        infl_after = min(1.0, infl_before + w / total_w)
                    else:
                        infl_after = infl_before
                    gain += infl_after - infl_before

            if gain > best_gain:
                best_gain, best_node = gain, node

        if best_node is not None:
            S.append(best_node)
            env.S.add(best_node)
            _invalidate_caches(env, best_node)
            remaining.discard(best_node)

    return S


# ── Greedy-discount pricing helpers (Babaei 2013, Section 4.2) ────────────────

def _rayleigh_price(x: float, b: float = 1.0) -> float:
    """Rayleigh PDF evaluated at normalised influence x (Babaei 2013 pricing).

    Formula:  f(x) = (2x / b²) · exp(−(2x)² / (2b²))

    Used as the Babaei greedy-discount price at influence tier x:
      - f(1/6) ≈ 0.315 for mid-influence buyers
      - f(2/6) ≈ 0.533 for high-influence buyers

    Args:
        x: Normalised influence in [0, 1].
        b: Rayleigh scale parameter (default 1.0, matches CLAUDE.md).

    Returns:
        Willingness-to-pay estimate at influence level x.
    """
    y = 2.0 * x
    return (y / (b * b)) * float(np.exp(-(y * y) / (2.0 * b * b)))


def _compute_normalized_infl(
    graph: nx.Graph,
    v,
    S_test: set,
    link_weights: dict,
) -> float:
    """Normalised influence on node v from seed set S_test.

    Formula (Babaei 2013):
        ι(v, S) = Σ_{j ∈ S ∩ neighbors(v)} w_{vj}  /  Σ_{k ∈ neighbors(v)} w_{vk}

    Args:
        graph:        NetworkX graph.
        v:            Target node.
        S_test:       Current seed set (subset of nodes).
        link_weights: Dict {(u, v): weight} from env._link_weights.

    Returns:
        Normalised influence in [0, 1].
    """
    neighbor_set = set(graph.neighbors(v))
    total_w = sum(link_weights.get((v, nb), 0.0) for nb in neighbor_set)
    if total_w <= 0.0:
        return 0.0
    seed_w = sum(link_weights.get((v, j), 0.0) for j in S_test if j in neighbor_set)
    return min(1.0, seed_w / total_w)


def _compute_k1_k2(
    graph: nx.Graph,
    env: "RevenueEnv",
    cfg,
) -> Tuple[int, int, List]:
    """Find k1, k2 that maximise buyers in target influence intervals.

    Algorithm (Babaei 2013, Section 4.2 — Greedy-Discount):
      k1: number of top-degree nodes given FREE so that the count of buyers
          with influence in [2/6, 4/6) is maximised.
      k2: additional free nodes so that buyers in [3/6, 5/6) are maximised.

    Only considers candidate nodes with degree ≥ µ/2 (high-degree pool).

    Args:
        graph: Social network graph.
        env:   RevenueEnv after reset() — provides link weights.
        cfg:   OmegaConf config (unused; kept for interface consistency).

    Returns:
        (k1, k2, candidate_nodes) — candidate_nodes sorted degree-descending
        with degree ≥ µ/2.  k2 ≥ k1 ≥ 0.
    """
    degrees = dict(graph.degree())
    mu = float(np.mean(list(degrees.values()))) if degrees else 1.0
    lw = env._link_weights

    nodes_by_deg = sorted(graph.nodes(), key=lambda v: degrees[v], reverse=True)
    candidates: List = [v for v in nodes_by_deg if degrees[v] >= mu / 2.0]
    if not candidates:
        candidates = nodes_by_deg  # fallback: use all nodes

    # ── Phase 1: find k1 (free seeds) ─────────────────────────────────────
    best_k1, best_score1 = 0, -1
    S_test: set = set()
    for i, node in enumerate(candidates):
        S_test.add(node)
        score = sum(
            1 for v in graph.nodes()
            if v not in S_test
            and degrees.get(v, 0) >= mu / 2.0
            and 2.0 / 6.0 <= _compute_normalized_infl(graph, v, S_test, lw) < 4.0 / 6.0
        )
        if score > best_score1:
            best_score1 = score
            best_k1 = i + 1

    # ── Phase 2: find k2 (additional to maximise mid-high influence) ──────
    best_k2, best_score2 = best_k1, -1
    S_test = set(candidates[:best_k1])
    for i, node in enumerate(candidates[best_k1:], start=best_k1):
        S_test.add(node)
        score = sum(
            1 for v in graph.nodes()
            if v not in S_test
            and degrees.get(v, 0) >= mu / 2.0
            and 3.0 / 6.0 <= _compute_normalized_infl(graph, v, S_test, lw) < 5.0 / 6.0
        )
        if score > best_score2:
            best_score2 = score
            best_k2 = i + 1

    return best_k1, best_k2, candidates


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 1 — Babaei et al. (2013)
# ════════════════════════════════════════════════════════════════════════════════

def ie_strategy(graph: nx.Graph, cfg, return_stats: bool = False):
    """Influence-and-Exploit (Babaei et al. 2013).

    Phase 1: Greedy seed selection — give k nodes for FREE (revenue = 0).
    Phase 2: Offer remaining n-k buyers at their exact current valuation (myopic).

    Uses cfg.influence.n_mc_samples for valuation estimates (consistent with
    other Babaei baselines — 96.9% acceptance observed on rice_facebook).

    Args:
        graph:        Social network graph.
        cfg:          OmegaConf DictConfig.
        return_stats: If True, return stats dict instead of float.

    Returns:
        float (default) or dict with revenue, n_offered, n_accepted, etc.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes

    seed_set = _greedy_seed_selection(graph, env, k)
    for node in seed_set:
        env.offered.add(node)

    total_revenue = 0.0
    n_offered = 0
    n_accepted = 0
    for node in nodes:
        if node in env.offered:
            continue
        # Seller prices at estimated valuation (discount=0 → myopic price)
        est_val = env._estimate_valuation(node)
        n_offered += 1
        if est_val > 0:
            # Buyer accepts iff TRUE valuation >= offered price (est_val)
            true_val = env._true_valuation(node)
            if true_val >= est_val:
                env.S.add(node)
                env._influence_cache = {}
                env._true_val_cache = {}
                env._est_val_cache = {}
                total_revenue += est_val
                n_accepted += 1
        env.offered.add(node)

    if return_stats:
        return {
            "revenue": total_revenue,
            "n_offered": n_offered,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / max(1, n_offered),
            "avg_price": total_revenue / max(1, n_accepted),
        }
    return total_revenue


def mu_discount(graph: nx.Graph, cfg, return_stats: bool = False):
    """µ-Discount (Babaei et al. 2013, Section 4.1).

    Sorts ALL n buyers by degree (high → low) and offers each a discount:
        d(j) = max(0,  1 − j / µ)    where µ = mean degree.

    k (cfg.budget.k) is NOT used — µ-Discount runs on ALL n buyers.
    Revenue is therefore k-independent (flat line in budget sweep).

    Args:
        graph:        Social network graph.
        cfg:          OmegaConf config.
        return_stats: If True, return stats dict instead of float.

    Returns:
        float (default) or dict with revenue, n_offered, n_accepted, etc.
    """
    env = _make_env(graph, cfg)
    env.reset()

    degrees = dict(graph.degree())
    mu = float(np.mean(list(degrees.values())))
    sorted_nodes = sorted(env.nodes, key=lambda v: degrees[v], reverse=True)
    total_revenue = 0.0
    n_offered = 0
    n_accepted = 0

    for j, node in enumerate(sorted_nodes):
        discount = max(0.0, min(1.0, 1.0 - float(j) / mu)) if mu > 0 else 0.0
        # Seller estimates valuation to set price
        est_val = env._estimate_valuation(node)
        offered_price = est_val * (1.0 - discount)
        n_offered += 1

        # Buyer accepts iff TRUE valuation >= offered price
        true_val = env._true_valuation(node)
        if true_val >= offered_price:
            env.S.add(node)
            _invalidate_caches(env, node)
            total_revenue += offered_price
            n_accepted += 1

        env.offered.add(node)
        env.t += 1

    if return_stats:
        return {
            "revenue": total_revenue,
            "n_offered": n_offered,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / max(1, n_offered),
            "avg_price": total_revenue / max(1, n_accepted),
        }
    return total_revenue


def sigma_discount(graph: nx.Graph, cfg, return_stats: bool = False):
    """σ-Discount (Babaei et al. 2013, Section 4.2.1).

    Offers ALL n buyers (sorted by degree, high → low) with 3-tier Rayleigh
    pricing based on degree thresholds (Babaei 2013 Section 4.2.1):

      deg > µ+σ         → FREE (super-influencer: joins S, influences others)
      µ < deg ≤ µ+σ     → price = f(1/6) ≈ 0.315  (above-average influencer)
      deg ≤ µ           → price = f(2/6) ≈ 0.534  (average/below-average buyer)

    k (cfg.budget.k) is NOT used — σ-Discount runs on ALL n buyers.
    Revenue is k-independent (flat line in budget sweep).

    Expected ordering (Babaei 2013 Figure 3):
      Greedy-Discount > σ-Discount > µ-Discount > IE-Strategy (at small k)

    Args:
        graph:        Social network graph.
        cfg:          OmegaConf config.
        return_stats: If True, return stats dict instead of float.

    Returns:
        float (default) or dict with revenue, n_offered, n_accepted, etc.
    """
    env = _make_env(graph, cfg)
    env.reset()

    degrees = dict(graph.degree())
    deg_values = np.array(list(degrees.values()), dtype=float)
    mu = float(np.mean(deg_values))
    sigma = float(np.std(deg_values))
    b = float(cfg.influence.b)
    lw = env._link_weights
    sorted_nodes = sorted(env.nodes, key=lambda v: degrees[v], reverse=True)
    total_revenue = 0.0
    n_offered = 0
    n_accepted = 0

    for node in sorted_nodes:
        deg = degrees[node]
        n_offered += 1

        if deg > mu + sigma:
            price = 0.0                            # FREE — super-influencer seed
        elif deg > mu:
            price = _rayleigh_price(1.0 / 6.0, b) # above-average → f(1/6)≈0.315
        else:
            price = _rayleigh_price(2.0 / 6.0, b) # rest → f(2/6)≈0.534

        if price == 0.0:
            # Free seed — always accepted
            env.S.add(node)
            _invalidate_caches(env, node)
        else:
            # Buyer accepts iff TRUE valuation >= Rayleigh fixed price
            true_val = env._true_valuation(node)
            if true_val >= price:
                env.S.add(node)
                env._influence_cache = {}
                env._true_val_cache = {}
                env._est_val_cache = {}
                total_revenue += price
                n_accepted += 1

        env.offered.add(node)
        env.t += 1

    if return_stats:
        return {
            "revenue": total_revenue,
            "n_offered": n_offered,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / max(1, n_offered),
            "avg_price": total_revenue / max(1, n_accepted),
        }
    return total_revenue


def greedy_discount(graph: nx.Graph, cfg, return_stats: bool = False):
    """Greedy influence-tier discount (Babaei et al. 2013, Section 4.2).

    Iterates over ALL n buyers, selecting the highest-valuation remaining
    buyer at each step (greedy), then prices based on CURRENT influence:

      influence < 2/6  → FREE (too little influence; seeds the cascade)
      2/6 ≤ infl < 4/6 → Rayleigh price f(2/6) ≈ 0.534 (tier lower bound)
      influence ≥ 4/6  → Rayleigh price f(4/6) ≈ 0.548 (tier lower bound)

    k (cfg.budget.k) is NOT used — Greedy-Discount runs on ALL n buyers.
    Revenue is k-independent (flat line in budget sweep).

    Also used as the GAIL expert teacher (see greedy_discount_trajectory).

    Args:
        graph:        Social network graph.
        cfg:          OmegaConf config.
        return_stats: If True, return stats dict instead of float.

    Returns:
        float (default) or dict with revenue, n_offered, n_accepted, etc.
    """
    env = _make_env(graph, cfg)
    env.reset()

    b = float(cfg.influence.b)
    lw = env._link_weights
    offered_set: set = set()
    total_revenue = 0.0
    n_offered = 0
    n_accepted = 0

    for _ in range(env.n):
        remaining = [v for v in env.nodes if v not in offered_set]
        if not remaining:
            break

        # Greedy: highest ESTIMATED valuation first (seller ranks by estimate)
        target = max(remaining, key=lambda v: env._estimate_valuation(v))
        infl = _compute_normalized_infl(graph, target, env.S, lw)

        if infl < 2.0 / 6.0:
            price = 0.0                           # FREE — below threshold
        elif infl < 4.0 / 6.0:
            # Tier-1 price = lower boundary of tier: f(2/6) ≈ 0.534
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            # Tier-2 price = lower boundary of tier: f(4/6) ≈ 0.548
            price = _rayleigh_price(4.0 / 6.0, b)

        # Buyer accepts iff TRUE valuation >= offered price
        true_val = env._true_valuation(target)
        n_offered += 1

        if price == 0.0:
            # Free seed — always accepted
            env.S.add(target)
            _invalidate_caches(env, target)
        elif true_val >= price:
            env.S.add(target)
            _invalidate_caches(env, target)
            total_revenue += price
            n_accepted += 1

        offered_set.add(target)
        env.offered.add(target)
        env.t += 1

    if return_stats:
        return {
            "revenue": total_revenue,
            "n_offered": n_offered,
            "n_accepted": n_accepted,
            "acceptance_rate": n_accepted / max(1, n_offered),
            "avg_price": total_revenue / max(1, n_accepted),
        }
    return total_revenue


def greedy_discount_trajectory(graph: nx.Graph, cfg) -> List[Tuple]:
    """Expert trajectory from greedy_discount for imitation learning (GAIL).

    Mirrors greedy_discount() exactly but runs n steps (full episode) for
    GAIL training.  At each step: select highest-valuation remaining buyer,
    price based on current influence tier:

      influence < 2/6  → discount = 1.0, revenue = 0 (FREE)
      2/6 ≤ infl < 4/6 → price = f(1/6); discount = 1 − price/valuation
      influence ≥ 4/6  → price = f(2/6); discount = 1 − price/valuation

    Returns:
        List of (node_idx, discount, marginal_revenue) per step (length n).
    """
    env = _make_env(graph, cfg)
    env.reset()

    n = env.n
    b = float(cfg.influence.b)
    lw = env._link_weights
    offered_set: set = set()
    trajectory: List[Tuple] = []

    for _ in range(n):
        remaining = [v for v in env.nodes if v not in offered_set]
        if not remaining:
            break

        target = max(remaining, key=lambda v: env._compute_valuation(v))
        infl = _compute_normalized_infl(graph, target, env.S, lw)

        if infl < 2.0 / 6.0:
            price = 0.0
        elif infl < 4.0 / 6.0:
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            price = _rayleigh_price(4.0 / 6.0, b)

        valuation = env._compute_valuation(target)
        node_idx = env.node_to_idx[target]

        if price == 0.0:
            env.S.add(target)
            env._influence_cache = {}
            discount_val = 1.0
            marginal = 0.0
        elif valuation >= price:
            env.S.add(target)
            env._influence_cache = {}
            discount_val = max(0.0, 1.0 - price / valuation) if valuation > 0 else 0.0
            marginal = price
        else:
            discount_val = max(0.0, 1.0 - price / valuation) if valuation > 0 else 0.0
            marginal = 0.0

        trajectory.append((node_idx, discount_val, marginal))
        offered_set.add(target)
        env.offered.add(target)
        env.t += 1

    return trajectory


def ie_strategy_trajectory(graph: nx.Graph, cfg) -> List[Tuple]:
    """Expert trajectory from IE-strategy for imitation learning.

    Phase 1 — k free seeds:  (node_idx, 1.0, 0.0)
    Phase 2 — myopic pricing: (node_idx, 0.0, val)

    Returns:
        List of (node_idx, discount, marginal_revenue) per step.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes
    seed_set = _greedy_seed_selection(graph, env, k)

    trajectory: List[Tuple] = []
    offered_set: set = set()

    for node in seed_set:
        trajectory.append((env.node_to_idx[node], 1.0, 0.0))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    remaining = [(v, env._compute_valuation(v)) for v in nodes if v not in offered_set]
    remaining.sort(key=lambda x: -x[1])
    for node, val in remaining:
        revenue = float(val) if val > 0 else 0.0
        trajectory.append((env.node_to_idx[node], 0.0, revenue))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    return trajectory


def hill_climbing_trajectory(graph: nx.Graph, cfg) -> List[Tuple]:
    """Expert trajectory: greedy IM seeds (free) + greedy_discount pricing.

    Used as the GAIL-RL-Rich expert teacher.

    Returns:
        List of (node_idx, discount, marginal_revenue) per step.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes
    n = env.n
    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    region_boundaries = [sorted_degrees[min(int(i * n / 6), n - 1)]
                         for i in range(7)]
    region_boundaries[-1] = 0

    trajectory: List[Tuple] = []
    offered_set: set = set()

    seed_set = _greedy_seed_selection(graph, env, k)
    for node in seed_set:
        trajectory.append((env.node_to_idx[node], 1.0, 0.0))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    for _ in range(n - len(offered_set)):
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break
        valuations = {v: env._compute_valuation(v) for v in remaining}
        target = max(remaining, key=lambda v: valuations[v])
        val = valuations[target]
        idx = env.node_to_idx[target]
        deg = degrees[target]

        if deg >= region_boundaries[1]:
            discount = 0.7
        elif deg >= region_boundaries[2]:
            discount = 0.55
        elif deg >= region_boundaries[3]:
            discount = 0.40
        elif deg >= region_boundaries[4]:
            discount = 0.25
        elif deg >= region_boundaries[5]:
            discount = 0.15
        else:
            discount = 0.05

        offered_price = val * (1.0 - discount)
        if val >= offered_price:
            env.S.add(target)
            env._influence_cache = {}
            marginal = offered_price if offered_price > 0 else 0.0
        else:
            marginal = 0.0

        trajectory.append((idx, discount, marginal))
        offered_set.add(target)
        env.offered.add(target)
        env.t += 1

    return trajectory


# ── Babaei multi-trial runner ──────────────────────────────────────────────────

def run_all_babaei(
    graph: nx.Graph,
    cfg,
    n_trials: int = 10,
) -> Dict[str, float]:
    """Run all 4 Babaei et al. baselines averaged over n_trials.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Monte Carlo trials over link-weight samples.

    Returns:
        Dict: method → mean revenue.
    """
    methods = {
        "ie_strategy":    ie_strategy,
        "mu_discount":    mu_discount,
        "sigma_discount": sigma_discount,
        "greedy_discount": greedy_discount,
    }
    results: Dict[str, float] = {}
    for name, fn in methods.items():
        revenues = [fn(graph, _override_seed(cfg, cfg.project.seed + t))
                    for t in range(n_trials)]
        results[name] = float(np.mean(revenues))
    return results


def run_all_babaei_stats(
    graph: nx.Graph,
    cfg,
    n_trials: int = 5,
) -> Dict[str, dict]:
    """Run all 4 Babaei baselines and return detailed per-method stats.

    Each stat is averaged over n_trials MC link-weight samples.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Monte Carlo trials.

    Returns:
        Dict: method → {revenue, n_offered, n_accepted, acceptance_rate, avg_price}
    """
    methods = {
        "ie_strategy":     ie_strategy,
        "mu_discount":     mu_discount,
        "sigma_discount":  sigma_discount,
        "greedy_discount": greedy_discount,
    }
    results: Dict[str, dict] = {}
    for name, fn in methods.items():
        trial_stats = []
        for t in range(n_trials):
            s = fn(graph, _override_seed(cfg, cfg.project.seed + t), return_stats=True)
            trial_stats.append(s)
        results[name] = {
            "revenue":         float(np.mean([s["revenue"]         for s in trial_stats])),
            "n_offered":       float(np.mean([s["n_offered"]       for s in trial_stats])),
            "n_accepted":      float(np.mean([s["n_accepted"]      for s in trial_stats])),
            "acceptance_rate": float(np.mean([s["acceptance_rate"] for s in trial_stats])),
            "avg_price":       float(np.mean([s["avg_price"]       for s in trial_stats])),
        }
    return results


# Backwards compatibility alias
def run_all_baselines(graph, cfg, n_trials=10):
    """Alias for run_all_babaei (kept for test compatibility)."""
    return run_all_babaei(graph, cfg, n_trials)


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 2 — Deep IM (decoupled): S2V-DQN and ToupleGDD
# ════════════════════════════════════════════════════════════════════════════════

def _apply_greedy_pricing_to_order(graph: nx.Graph, cfg, node_order: List) -> float:
    """Apply Babaei greedy-discount pricing to a GNN-derived node ordering.

    The GNN (S2V-DQN / ToupleGDD) supplies the ORDER.  Each buyer is priced
    based on their CURRENT normalised influence at offer time (same rule as
    greedy_discount):

      influence < 2/6  → FREE
      2/6 ≤ infl < 4/6 → offered at Rayleigh price f(1/6) ≈ 0.315
      influence ≥ 4/6  → offered at Rayleigh price f(2/6) ≈ 0.533

    Stops after k total offers.  Revenue INCREASES with k.

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig (uses cfg.budget.k).
        node_order: GNN-derived node visit order (typically length n).

    Returns:
        Total revenue from up to k offers.
    """
    env = _make_env(graph, cfg)
    env.reset()

    b = float(cfg.influence.b)
    lw = env._link_weights
    total_revenue = 0.0

    for i, node in enumerate(node_order):
        infl = _compute_normalized_infl(graph, node, env.S, lw)

        if infl < 2.0 / 6.0:
            price = 0.0
        elif infl < 4.0 / 6.0:
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            price = _rayleigh_price(4.0 / 6.0, b)

        valuation = env._compute_valuation(node)

        if price == 0.0:
            env.S.add(node)
            env._influence_cache = {}
        elif valuation >= price:
            env.S.add(node)
            env._influence_cache = {}
            total_revenue += price

        env.offered.add(node)
        env.t += 1

    return total_revenue


def _run_decoupled_gnn_baseline(
    graph: nx.Graph,
    cfg,
    node_order: List,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Apply greedy_discount pricing to GNN order over n_trials.

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig.
        node_order: GNN-derived node visit order.
        n_trials:   Link-weight sampling trials.

    Returns:
        {"mean_revenue": float, "std_revenue": float}
    """
    revenues = [
        _apply_greedy_pricing_to_order(graph, _override_seed(cfg, cfg.project.seed + t), node_order)
        for t in range(n_trials)
    ]
    return {"mean_revenue": float(np.mean(revenues)), "std_revenue": float(np.std(revenues))}


def _graph_to_edgelist_file(graph: nx.Graph, path: str) -> None:
    """Write graph to ToupleGDD-compatible edge-list (0-indexed, undirected)."""
    nodes = sorted(graph.nodes())
    id_map = {v: i for i, v in enumerate(nodes)}
    with open(path, "w") as f:
        for u, v in graph.edges():
            w = graph[u][v].get("weight", 1.0)
            f.write(f"{id_map[u]} {id_map[v]} {w:.6f}\n")
            if not graph.is_directed():
                f.write(f"{id_map[v]} {id_map[u]} {w:.6f}\n")


def _call_touplegdd_shim(
    graph: nx.Graph,
    cfg,
    model_name: str,
    checkpoint_filename: str,
) -> Optional[List]:
    """Call ToupleGDD subprocess shim; returns ordered node list or None."""
    touple_dir = str(cfg.baselines.touple_gdd_dir)
    ckpt_path  = os.path.join(touple_dir, checkpoint_filename)
    shim_path  = os.path.join(touple_dir, "touplegdd_seed_printer.py")

    for path, label in [(touple_dir,  "touple_gdd_dir"),
                        (ckpt_path,   f"{model_name} checkpoint"),
                        (shim_path,   "touplegdd_seed_printer.py")]:
        if not os.path.exists(path):
            warnings.warn(f"[Group-2] {label} not found: {path}. "
                          f"Skipping {model_name}.", stacklevel=3)
            return None

    budget = int(cfg.budget.k)
    nodes  = sorted(graph.nodes())

    with tempfile.NamedTemporaryFile(suffix=".txt", mode="w", delete=False) as tf:
        tmp_path = tf.name

    try:
        _graph_to_edgelist_file(graph, tmp_path)
        result = subprocess.run(
            ["python", "touplegdd_seed_printer.py",
             tmp_path, model_name, ckpt_path, str(budget)],
            cwd=touple_dir,
            capture_output=True, text=True, timeout=300,
        )
        if result.returncode != 0:
            warnings.warn(f"[Group-2] {model_name} subprocess error: "
                          f"{result.stderr[:200]}. Skipping.", stacklevel=3)
            return None

        data = json.loads(result.stdout.strip())
        if "error" in data:
            warnings.warn(f"[Group-2] {model_name} returned error: "
                          f"{data['error']}. Skipping.", stacklevel=3)
            return None

        seed_indices = data.get("seeds", [])
        seed_nodes   = [nodes[i] for i in seed_indices if i < len(nodes)]
        non_seeds = [v for v in nodes if v not in set(seed_nodes)]
        deg = dict(graph.degree())
        non_seeds.sort(key=lambda v: deg[v], reverse=True)
        return seed_nodes + non_seeds

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        warnings.warn(f"[Group-2] {model_name} failed: {e}. Skipping.", stacklevel=3)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def s2v_dqn_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[float]:
    """S2V-DQN (Dai et al. 2017) decoupled: GNN seeds + greedy_discount pricing.

    Loads pretrained S2V-DQN checkpoint via ToupleGDD subprocess shim.
    Returns mean revenue over n_trials or None if checkpoint absent.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig (needs cfg.baselines.touple_gdd_dir,
                  cfg.baselines.s2v_dqn_checkpoint).
        n_trials: Number of pricing trials.

    Returns:
        Mean revenue (float) or None.
    """
    node_order = _call_touplegdd_shim(
        graph, cfg, "S2V_DQN", str(cfg.baselines.s2v_dqn_checkpoint))
    if node_order is None:
        return None
    out = _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)
    return out["mean_revenue"]


def touple_gdd_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[float]:
    """ToupleGDD (Chen et al. 2022) decoupled: GNN seeds + greedy_discount pricing.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Number of pricing trials.

    Returns:
        Mean revenue (float) or None.
    """
    node_order = _call_touplegdd_shim(
        graph, cfg, "Tripling", str(cfg.baselines.touple_gdd_checkpoint))
    if node_order is None:
        return None
    out = _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)
    return out["mean_revenue"]


# ════════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Our joint models (load from checkpoint)
# ════════════════════════════════════════════════════════════════════════════════

def _eval_joint_policy_from_checkpoint(
    graph: nx.Graph,
    cfg,
    ckpt_path: str,
    model_name: str,
) -> Optional[float]:
    """Load a JointPolicy checkpoint and run a greedy joint episode.

    The model simultaneously selects the next buyer AND sets the discount.
    Revenue = sum of offered_price where buyer accepts.

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig.
        ckpt_path:  Path to the .pt checkpoint (state_dict).
        model_name: Name for warning messages.

    Returns:
        Total revenue (float) or None if checkpoint absent / load fails.
    """
    if not os.path.exists(ckpt_path):
        warnings.warn(f"[Group-3] {model_name} checkpoint not found: {ckpt_path}. "
                      "Skipping.", stacklevel=3)
        return None

    try:
        import torch
        from src.models.encoders.graphsage import GraphSAGEEncoder
        from src.models.policies.joint_policy import JointPolicy
        from src.utils.features import compute_static_features, compute_node_features
        from src.utils.helpers import graph_to_pyg_data, get_available_mask

        device = torch.device("cpu")
        enc = GraphSAGEEncoder(
            in_dim=cfg.features.dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)
        policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
        policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        policy.eval()

        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        env = _make_env(graph, cfg)
        env.reset()
        total_revenue = 0.0

        with torch.no_grad():
            for _ in range(n):
                available = list(env.available_nodes) if hasattr(env, "available_nodes") else \
                    [v for v in nodes if v not in env.offered]
                if not available:
                    break

                feats = compute_node_features(
                    graph=graph, static_features=static,
                    S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, n=n, k=n, env=env)
                data = graph_to_pyg_data(graph, feats, device)
                mask = get_available_mask(n, frozenset(env.offered), nodes, device)

                node_idx, discount, _ = policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=True)

                # Clip to valid available node
                if nodes[node_idx] not in set(available):
                    node_idx = nodes.index(available[0])

                selected_node = nodes[node_idx]
                valuation = env._compute_valuation(selected_node)
                offered_price = valuation * (1.0 - discount)

                if valuation >= offered_price:
                    env.S.add(selected_node)
                    env._influence_cache = {}
                    if offered_price > 0:
                        total_revenue += offered_price

                env.offered.add(selected_node)
                env.t += 1

        return total_revenue

    except Exception as e:
        warnings.warn(f"[Group-3] {model_name} evaluation failed: {e}", stacklevel=2)
        return None


def _eval_sequential_policy_from_checkpoint(
    graph: nx.Graph,
    cfg,
    ckpt_path: str,
    model_name: str,
) -> Optional[float]:
    """Load a SequentialJointPolicy (GNN + LSTM) checkpoint and run greedy episode.

    Handles the LSTM hidden state across steps via reset_episode() /
    update_sequence_state().

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig.
        ckpt_path:  Path to .pt checkpoint (state_dict).
        model_name: Name for warning messages.

    Returns:
        Total revenue (float) or None.
    """
    if not os.path.exists(ckpt_path):
        warnings.warn(f"[Group-3] {model_name} checkpoint not found: {ckpt_path}. "
                      "Skipping.", stacklevel=3)
        return None

    try:
        import torch
        from src.models.encoders.graphsage import GraphSAGEEncoder
        from src.models.encoders.sequence_models import EpisodeLSTM
        from src.models.policies.sequential_joint_policy import SequentialJointPolicy
        from src.utils.features import compute_static_features, compute_node_features
        from src.utils.helpers import graph_to_pyg_data, get_available_mask

        device = torch.device("cpu")
        enc = GraphSAGEEncoder(
            in_dim=cfg.features.dim,
            hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers,
            dropout=cfg.encoder.dropout,
        ).to(device)
        seq_model = EpisodeLSTM(
            gnn_dim=cfg.encoder.hidden_dim,
            hidden_dim=cfg.sequence_model.hidden_dim,
            n_layers=cfg.sequence_model.n_layers,
        ).to(device)
        policy = SequentialJointPolicy(
            encoder=enc,
            sequence_model=seq_model,
            gnn_dim=cfg.encoder.hidden_dim,
            context_dim=cfg.sequence_model.hidden_dim,
        ).to(device)
        policy.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
        policy.eval()
        policy.reset_episode(device)

        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        env = _make_env(graph, cfg)
        env.reset()
        total_revenue = 0.0

        with torch.no_grad():
            for _ in range(n):
                available = [v for v in nodes if v not in env.offered]
                if not available:
                    break

                feats = compute_node_features(
                    graph=graph, static_features=static,
                    S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, n=n, k=n, env=env)
                data = graph_to_pyg_data(graph, feats, device)
                mask = get_available_mask(n, frozenset(env.offered), nodes, device)

                node_idx, discount, _ = policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=True)

                if nodes[node_idx] not in set(available):
                    node_idx = nodes.index(available[0])

                selected_node = nodes[node_idx]
                valuation = env._compute_valuation(selected_node)
                offered_price = valuation * (1.0 - discount)
                accepted = valuation >= offered_price

                if accepted:
                    env.S.add(selected_node)
                    env._influence_cache = {}
                    if offered_price > 0:
                        total_revenue += offered_price

                policy.update_sequence_state(
                    discount=discount,
                    accepted=accepted,
                    revenue=offered_price if accepted else 0.0,
                )
                env.offered.add(selected_node)
                env.t += 1

        return total_revenue

    except Exception as e:
        warnings.warn(f"[Group-3] {model_name} evaluation failed: {e}", stacklevel=2)
        return None


def eval_rev_gnn_im_rl(graph: nx.Graph, cfg) -> Optional[float]:
    """Evaluate Rev-GNN-IM-RL from checkpoint (GNN + REINFORCE joint policy).

    Checkpoint: checkpoints/rev_gnn_im_rl/best.pt

    Returns:
        Total revenue or None if checkpoint absent.
    """
    ckpt = os.path.join("checkpoints", "rev_gnn_im_rl", "best.pt")
    return _eval_joint_policy_from_checkpoint(graph, cfg, ckpt, "Rev-GNN-IM-RL")


def eval_rev_gail_rl(graph: nx.Graph, cfg) -> Optional[float]:
    """Evaluate Rev-GAIL-RL from checkpoint (GNN + GAIL joint policy).

    Checkpoint: checkpoints/rev_gail_rl/best.pt

    Returns:
        Total revenue or None if checkpoint absent.
    """
    ckpt = os.path.join("checkpoints", "rev_gail_rl", "best.pt")
    return _eval_joint_policy_from_checkpoint(graph, cfg, ckpt, "Rev-GAIL-RL")


def eval_rev_gnn_lstm(graph: nx.Graph, cfg) -> Optional[float]:
    """Evaluate Rev-GNN-LSTM from checkpoint (GNN + LSTM + REINFORCE).

    Checkpoint: checkpoints/rev_gnn_lstm/best.pt

    Returns:
        Total revenue or None if checkpoint absent.
    """
    ckpt = os.path.join("checkpoints", "rev_gnn_lstm", "best.pt")
    return _eval_sequential_policy_from_checkpoint(graph, cfg, ckpt, "Rev-GNN-LSTM")


def eval_rev_gail_lstm(graph: nx.Graph, cfg) -> Optional[float]:
    """Evaluate Rev-GAIL-LSTM from checkpoint (GNN + LSTM + GAIL).

    Checkpoint: checkpoints/rev_gail_lstm/best.pt

    Returns:
        Total revenue or None if checkpoint absent.
    """
    ckpt = os.path.join("checkpoints", "rev_gail_lstm", "best.pt")
    return _eval_sequential_policy_from_checkpoint(graph, cfg, ckpt, "Rev-GAIL-LSTM")


# ════════════════════════════════════════════════════════════════════════════════
# Full 10-method runner
# ════════════════════════════════════════════════════════════════════════════════

def run_full_comparison(
    graph: nx.Graph,
    cfg,
    n_trials_babaei: int = 5,
    n_trials_deep_im: int = 3,
) -> Dict[str, Optional[float]]:
    """Run all 10 methods and return a comparison dict.

    Groups:
      1. Babaei et al. 2013 (4 methods, averaged over n_trials_babaei)
      2. Deep IM decoupled  (2 methods, averaged over n_trials_deep_im)
      3. Our Rev models     (4 methods, single greedy episode from checkpoint)

    Args:
        graph:            Social network graph.
        cfg:              OmegaConf DictConfig.
        n_trials_babaei:  MC trials for Babaei baselines.
        n_trials_deep_im: MC pricing trials for Group 2.

    Returns:
        Ordered dict: method_name → revenue (float or None).
    """
    results: Dict[str, Optional[float]] = {}

    # ── Group 1: Babaei ───────────────────────────────────────────────────────
    for name, fn in [("ie_strategy",    ie_strategy),
                     ("mu_discount",    mu_discount),
                     ("sigma_discount", sigma_discount),
                     ("greedy_discount", greedy_discount)]:
        revenues = [fn(graph, _override_seed(cfg, cfg.project.seed + t))
                    for t in range(n_trials_babaei)]
        results[name] = float(np.mean(revenues))

    # ── Group 2: Deep IM decoupled ────────────────────────────────────────────
    results["s2v_dqn"]    = s2v_dqn_decoupled(graph, cfg, n_trials_deep_im)
    results["touple_gdd"] = touple_gdd_decoupled(graph, cfg, n_trials_deep_im)

    # ── Group 3: Our Rev models (greedy, joint) ───────────────────────────────
    results["rev_gnn_im_rl"]  = eval_rev_gnn_im_rl(graph, cfg)
    results["rev_gail_rl"]    = eval_rev_gail_rl(graph, cfg)
    results["rev_gnn_lstm"]   = eval_rev_gnn_lstm(graph, cfg)
    results["rev_gail_lstm"]  = eval_rev_gail_lstm(graph, cfg)

    return results
