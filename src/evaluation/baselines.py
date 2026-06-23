"""
src/evaluation/baselines.py

Babaei et al. (2013) baseline strategies for revenue maximization.

Four strategies:
  1. ie_strategy      — Influence-and-Exploit: greedy seed selection + myopic pricing
  2. mu_discount      — µ-rule: discount threshold based on average degree
  3. greedy_discount  — Greedy degree-based discount (best in Babaei et al., used as expert)
  4. sigma_discount   — σ-rule: µ + σ degree distribution threshold

Reference:
  Babaei et al. (2013) "Revenue Maximization in Social Networks through Discounting"
  ICWSM 2013.
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
    """Construct a RevenueEnv from an OmegaConf config.

    Args:
        graph: Social network graph.
        cfg: OmegaConf DictConfig with influence, reward, budget sub-configs.

    Returns:
        RevenueEnv instance.
    """
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


def _greedy_seed_selection(
    graph: nx.Graph,
    env: RevenueEnv,
    k: int,
) -> List:
    """Greedy hill-climbing seed selection by marginal influence gain.

    At each step, adds the node that maximizes the marginal increase in
    total influence across all other nodes (proxy for spread).

    Args:
        graph: NetworkX graph.
        env: RevenueEnv (link weights already sampled via reset()).
        k: Budget (number of seeds).

    Returns:
        List of selected node identifiers (not indices).
    """
    S = []
    remaining = set(graph.nodes())

    for _ in range(min(k, graph.number_of_nodes())):
        best_node = None
        best_gain = -1.0

        for node in remaining:
            # Marginal influence gain: how much does adding `node` to S
            # increase the total valuation of all neighbors?
            gain = 0.0
            for nb in graph.neighbors(node):
                if nb in remaining and nb not in S:
                    # Influence on nb if node were added to S
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
                best_gain = gain
                best_node = node

        if best_node is not None:
            S.append(best_node)
            env.S.add(best_node)
            env._influence_cache = {}
            remaining.discard(best_node)

    return S


# ── Strategy 1: IE-Strategy (Influence-and-Exploit) ───────────────────────────

def ie_strategy(graph: nx.Graph, cfg) -> float:
    """Influence-and-Exploit baseline (Babaei et al. 2013).

    Phase 1: Greedy hill climbing selects top-k buyers by influence gain (seed set S).
    Phase 2: Give item FREE to S (revenue = 0 from these buyers).
    Phase 3: Offer remaining buyers at their current valuation (myopic pricing, discount=0).
             Buyers accept iff valuation > 0 (they always accept at their exact valuation).

    Args:
        graph: Social network graph.
        cfg: OmegaConf DictConfig with graph, influence, reward, budget sub-configs.

    Returns:
        Total revenue collected.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes
    n = env.n

    # Phase 1 + 2: greedy selection + free give-away
    seed_set = _greedy_seed_selection(graph, env, k)
    seed_set_set = set(seed_set)

    # Mark seeds as offered at discount=1.0 (free), revenue=0 from each
    total_revenue = 0.0
    for node in seed_set:
        idx = env.node_to_idx[node]
        env.offered.add(node)
        # Node is already in env.S from _greedy_seed_selection

    # Phase 3: Offer remaining buyers at myopic price (discount=0.0)
    for node in nodes:
        if node in env.offered:
            continue
        valuation = env._compute_valuation(node)
        # Myopic: offer at full valuation. Buyer accepts iff valuation > 0.
        if valuation > 0:
            total_revenue += valuation
        env.offered.add(node)

    return total_revenue


# ── Strategy 2: µ-Discount ─────────────────────────────────────────────────────

def mu_discount(graph: nx.Graph, cfg) -> float:
    """µ-Discount baseline (Babaei et al. 2013, Section 4.1).

    The µ-rule prices based on expected (average) degree.
    µ = average degree of the graph.

    Ordering: sort buyers by current valuation (high → low influenceable first).
    For buyer at rank j (0-indexed):
      - If j < µ (high influence): offer at higher price using f(j / µ)
        normalized by peak. These are the most valuable buyers.
      - If j >= µ: offer at a discount to encourage acceptance.

    Practical implementation:
      For each buyer, compute their current valuation v = f(influence).
      Offer discount d = max(0, 1 - (rank / µ)) so that higher-ranked buyers
      get less discount (i.e., higher price).
      Offered price = v * (1 - d). Accept if offered_price > 0.

    Args:
        graph: Social network graph.
        cfg: OmegaConf config.

    Returns:
        Total revenue.
    """
    env = _make_env(graph, cfg)
    env.reset()

    degrees = dict(graph.degree())
    mu = float(np.mean(list(degrees.values())))

    nodes = env.nodes
    total_revenue = 0.0

    # Sort nodes by degree (high degree first — most influential first)
    sorted_nodes = sorted(nodes, key=lambda v: degrees[v], reverse=True)

    for j, node in enumerate(sorted_nodes):
        # µ-rule discount: fraction j / µ but capped at 0
        # Lower j (higher degree) → less discount → higher price
        discount = max(0.0, 1.0 - float(j) / mu) if mu > 0 else 0.0
        discount = min(discount, 1.0)

        valuation = env._compute_valuation(node)
        offered_price = valuation * (1.0 - discount)

        # Accept if price ≤ valuation (includes free seeding when offered_price=0)
        if valuation >= offered_price:
            env.S.add(node)
            env._influence_cache = {}
            total_revenue += offered_price   # 0 when discounted fully, positive otherwise

        env.offered.add(node)
        env.t += 1

    return total_revenue


# ── Strategy 3: Greedy-Discount (expert for imitation learning) ────────────────

def greedy_discount(graph: nx.Graph, cfg) -> float:
    """Greedy degree-based discount (Babaei et al. 2013, Section 4.2).

    Divides buyers into k=6 influence regions (Blue, Green, ... by degree quartile),
    then greedily selects the discount for each buyer to maximize buyers in the
    "blue interval" (optimal discount region: high valuation + high acceptance).

    At each step:
      1. Sort remaining buyers by CURRENT valuation (descending).
      2. Offer the highest-valuation buyer a discount that maximizes:
           offered_price = valuation * (1 - d)
         subject to the buyer accepting (offered_price > threshold).
      3. The discount threshold is determined by the buyer's influence region:
           - High influence (top 1/6): aggressive discount (d in [0.6, 0.8])
           - Mid influence (mid 2/6): moderate discount (d in [0.3, 0.5])
           - Low influence (bottom 3/6): low discount (d in [0.0, 0.3])

    This approximates the "blue interval" maximization in Babaei et al.

    This function is also used to generate EXPERT TRAJECTORIES for the
    ImitationTrainer and GAILTrainer.

    Args:
        graph: Social network graph.
        cfg: OmegaConf config.

    Returns:
        Total revenue.
    """
    env = _make_env(graph, cfg)
    env.reset()

    nodes = env.nodes
    n = env.n
    k_regions = 6  # number of influence regions (as in Babaei et al.)

    # Degree-based influence thresholds (6 regions by sorted degree)
    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    # Cap index at n-1 to avoid out-of-range (last region boundary = 0)
    region_boundaries = [sorted_degrees[min(int(i * n / k_regions), n - 1)]
                         for i in range(k_regions + 1)]
    region_boundaries[-1] = 0  # last boundary is 0

    total_revenue = 0.0
    offered_set = set()

    for step in range(n):
        # Get remaining (not yet offered) nodes
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break

        # Sort by current valuation descending (greedy: target highest-value buyer)
        valuations = {}
        for node in remaining:
            valuations[node] = env._compute_valuation(node)

        target_node = max(remaining, key=lambda v: valuations[v])
        val = valuations[target_node]

        # Determine influence region → discount
        deg = degrees[target_node]
        if deg >= region_boundaries[1]:      # top 1/6 by degree
            discount = 0.7   # generous discount to highly influential node
        elif deg >= region_boundaries[2]:    # top 2/6
            discount = 0.55
        elif deg >= region_boundaries[3]:    # top 3/6 (median)
            discount = 0.4
        elif deg >= region_boundaries[4]:    # below median
            discount = 0.25
        elif deg >= region_boundaries[5]:    # bottom 2/6
            discount = 0.15
        else:                                # bottom 1/6
            discount = 0.05

        offered_price = val * (1.0 - discount)

        # Accept when price ≤ valuation (free seeding bootstraps influence cascade)
        if val >= offered_price:
            env.S.add(target_node)
            env._influence_cache = {}
            if offered_price > 0:
                total_revenue += offered_price

        offered_set.add(target_node)
        env.offered.add(target_node)
        env.t += 1

    return total_revenue


def greedy_discount_trajectory(
    graph: nx.Graph,
    cfg,
) -> List[Tuple]:
    """Generate expert trajectory for imitation learning.

    Same as greedy_discount() but returns each step's action + marginal gain.

    Args:
        graph: Social network graph.
        cfg: OmegaConf config.

    Returns:
        List of tuples (node_idx, discount, marginal_revenue) per step.
    """
    env = _make_env(graph, cfg)
    env.reset()

    nodes = env.nodes
    n = env.n
    k_regions = 6

    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    region_boundaries = [sorted_degrees[min(int(i * n / k_regions), n - 1)]
                         for i in range(k_regions + 1)]
    region_boundaries[-1] = 0

    trajectory = []
    offered_set = set()

    for step in range(n):
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break

        valuations = {node: env._compute_valuation(node) for node in remaining}
        target_node = max(remaining, key=lambda v: valuations[v])
        val = valuations[target_node]
        target_idx = env.node_to_idx[target_node]

        # Determine influence region → discount (mirrors greedy_discount exactly)
        deg = degrees[target_node]
        if deg >= region_boundaries[1]:
            discount = 0.7
        elif deg >= region_boundaries[2]:
            discount = 0.55
        elif deg >= region_boundaries[3]:
            discount = 0.4
        elif deg >= region_boundaries[4]:
            discount = 0.25
        elif deg >= region_boundaries[5]:
            discount = 0.15
        else:
            discount = 0.05

        offered_price = val * (1.0 - discount)
        # Accept when price ≤ valuation (free seeding bootstraps influence cascade)
        if val >= offered_price:
            env.S.add(target_node)
            env._influence_cache = {}
            marginal_revenue = offered_price if offered_price > 0 else 0.0
        else:
            marginal_revenue = 0.0

        trajectory.append((target_idx, discount, marginal_revenue))

        offered_set.add(target_node)
        env.offered.add(target_node)
        env.t += 1

    return trajectory


# ── Strategy 4: σ-Discount ────────────────────────────────────────────────────

def sigma_discount(graph: nx.Graph, cfg) -> float:
    """σ-Discount baseline (Babaei et al. 2013, Section 4.2.1).

    Uses µ (mean degree) and σ (std dev of degree) to identify influential nodes.
    Nodes with degree > µ + σ are "super-influencers" and get deeper discounts.
    Nodes with µ < degree <= µ + σ get moderate discounts.
    Nodes with degree <= µ get small discounts.

    This mirrors the original paper's idea: σ identifies outlier high-degree nodes
    that benefit most from aggressive discounting.

    Args:
        graph: Social network graph.
        cfg: OmegaConf config.

    Returns:
        Total revenue.
    """
    env = _make_env(graph, cfg)
    env.reset()

    degrees = dict(graph.degree())
    deg_values = np.array(list(degrees.values()), dtype=float)
    mu = float(np.mean(deg_values))
    sigma = float(np.std(deg_values))

    nodes = env.nodes
    total_revenue = 0.0

    # Sort nodes by degree (high → low)
    sorted_nodes = sorted(nodes, key=lambda v: degrees[v], reverse=True)

    for node in sorted_nodes:
        deg = degrees[node]
        valuation = env._compute_valuation(node)

        # σ-rule discount thresholds
        if deg > mu + sigma:
            # Super-influencer: deep discount to acquire → spreads influence wide
            discount = 0.65
        elif deg > mu:
            # Above average: moderate discount
            discount = 0.35
        else:
            # Below average: minimal discount
            discount = 0.10

        offered_price = valuation * (1.0 - discount)

        # Accept when price ≤ valuation (free seeding bootstraps influence cascade)
        if valuation >= offered_price:
            env.S.add(node)
            env._influence_cache = {}
            if offered_price > 0:
                total_revenue += offered_price

        env.offered.add(node)
        env.t += 1

    return total_revenue


def ie_strategy_trajectory(
    graph: nx.Graph,
    cfg,
) -> List[Tuple]:
    """Generate expert trajectory for imitation learning from the IE strategy.

    IE strategy is the STRONGEST Babaei et al. baseline (~40.73 revenue), far
    better than greedy_discount (~32.56).  Using it as the imitation expert gives
    the GNN a high starting point from which RL can push BEYOND 40.73.

    The IE trajectory is converted to a SEQ format the GNN can imitate:
      Phase 1 — k seed nodes (greedy influence order): (node_idx, 1.0, 0.0)
                 Discount=1.0 = FREE seeding to trigger cascade.
      Phase 2 — remaining n-k nodes (sorted by valuation DESC):
                 (node_idx, 0.0, val) — full price (no discount) after cascade.

    Args:
        graph: Social network graph.
        cfg: OmegaConf config (used for budget.k, link weights, etc.).

    Returns:
        List of (node_idx, discount, marginal_revenue) per step.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes

    # Greedy seed selection — populates env.S and env._link_weights
    seed_set = _greedy_seed_selection(graph, env, k)
    seed_set_ids = set(seed_set)

    trajectory = []
    offered_set = set()

    # Phase 1: seeds offered for FREE (discount=1.0 → loss-leader seeding)
    for node in seed_set:
        idx = env.node_to_idx[node]
        trajectory.append((idx, 1.0, 0.0))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    # Phase 2: remaining nodes at full valuation DESC (discount=0.0)
    # Influence cascade from seed_set already in env.S → valuations reflect spread
    remaining = [(v, env._compute_valuation(v)) for v in nodes if v not in offered_set]
    remaining.sort(key=lambda x: -x[1])

    for node, val in remaining:
        idx = env.node_to_idx[node]
        revenue = float(val) if val > 0 else 0.0
        trajectory.append((idx, 0.0, revenue))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    return trajectory


# ── Run all baselines ──────────────────────────────────────────────────────────

def run_all_baselines(
    graph: nx.Graph,
    cfg,
    n_trials: int = 10,
) -> Dict[str, float]:
    """Run all 4 Babaei et al. baselines, averaged over n_trials.

    Each trial samples different link weights (via env.reset()), which models
    the seller's uncertainty about exact edge weights.

    Args:
        graph: Social network graph.
        cfg: OmegaConf DictConfig.
        n_trials: Number of Monte Carlo trials over link weight sampling.

    Returns:
        Dict with keys: "ie_strategy", "mu_discount", "greedy_discount", "sigma_discount"
        each mapped to mean revenue over n_trials.
    """
    results = {
        "ie_strategy": [],
        "mu_discount": [],
        "greedy_discount": [],
        "sigma_discount": [],
    }

    for trial in range(n_trials):
        # Each trial uses a different seed for link weight sampling
        trial_cfg = _override_seed(cfg, cfg.project.seed + trial)
        results["ie_strategy"].append(ie_strategy(graph, trial_cfg))
        results["mu_discount"].append(mu_discount(graph, trial_cfg))
        results["greedy_discount"].append(greedy_discount(graph, trial_cfg))
        results["sigma_discount"].append(sigma_discount(graph, trial_cfg))

    return {k: float(np.mean(v)) for k, v in results.items()}


def _override_seed(cfg, new_seed: int):
    """Return a new config with overridden project.seed.

    Args:
        cfg: OmegaConf DictConfig.
        new_seed: Replacement seed value.

    Returns:
        OmegaConf DictConfig with project.seed = new_seed.
    """
    from omegaconf import OmegaConf
    override = OmegaConf.create({"project": {"seed": new_seed}})
    return OmegaConf.merge(cfg, override)


# ════════════════════════════════════════════════════════════════════════════════
# GROUP A — Additional hand-crafted baselines
# ════════════════════════════════════════════════════════════════════════════════

def random_baseline(graph: nx.Graph, cfg) -> float:
    """Floor baseline: offer every buyer a random discount in [0, 1].

    Buyers are visited in a random order.  Every buyer accepts (since
    offered_price = valuation * (1 - d) ≤ valuation always), but high
    discounts reduce revenue.  Expected revenue ≈ 0.5 × sum(valuations).

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        Total revenue collected.
    """
    rng = np.random.default_rng(cfg.project.seed)
    env = _make_env(graph, cfg)
    env.reset()

    nodes = list(env.nodes)
    rng.shuffle(nodes)
    total_revenue = 0.0

    for node in nodes:
        discount = float(rng.uniform(0.0, 1.0))
        valuation = env._compute_valuation(node)
        offered_price = valuation * (1.0 - discount)
        if valuation >= offered_price:
            env.S.add(node)
            env._influence_cache = {}
            total_revenue += offered_price
        env.offered.add(node)
        env.t += 1

    return total_revenue


def myopic_full_price(graph: nx.Graph, cfg) -> float:
    """Myopic full-price baseline: offer every buyer at their exact valuation.

    Discount is always 0.0.  Every buyer accepts (price = valuation = threshold).
    Revenue = sum of all valuations at the time of offer.  Buyers are visited in
    random order; cascade from accepted buyers grows valuations for later buyers.

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        Total revenue collected.
    """
    rng = np.random.default_rng(cfg.project.seed)
    env = _make_env(graph, cfg)
    env.reset()

    nodes = list(env.nodes)
    rng.shuffle(nodes)
    total_revenue = 0.0

    for node in nodes:
        valuation = env._compute_valuation(node)
        # No discount → price = valuation; buyer always accepts
        if valuation > 0:
            env.S.add(node)
            env._influence_cache = {}
            total_revenue += valuation
        env.offered.add(node)
        env.t += 1

    return total_revenue


def hill_climbing_baseline(graph: nx.Graph, cfg) -> float:
    """Hill-climbing seed selection + degree-based pricing (Babaei et al. 2013).

    Extends the IE-strategy "free seed / full price" dichotomy by applying
    greedy_discount-style pricing to non-seed buyers instead of pure myopic
    pricing.  This captures more revenue from non-seeds while still triggering
    the influence cascade from the k seeds.

    Phase 1 (k steps):  Greedy IM hill-climbing selects k seeds (free, discount=1.0).
    Phase 2 (n-k steps): Remaining buyers priced by degree region (6-level discount).

    This is the expert teacher for GAIL-RL-Rich — it produces both a strong
    seed policy AND a reasonable pricing policy for non-seeds.

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        Total revenue collected.
    """
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = env.nodes
    n = env.n
    degrees = dict(graph.degree())

    # Degree region boundaries for 6-level greedy pricing
    sorted_degrees = sorted(degrees.values(), reverse=True)
    region_boundaries = [sorted_degrees[min(int(i * n / 6), n - 1)]
                         for i in range(7)]
    region_boundaries[-1] = 0

    total_revenue = 0.0
    offered_set = set()

    # Phase 1: greedy IM → k free seeds
    seed_set = _greedy_seed_selection(graph, env, k)
    for node in seed_set:
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1
    # Seeds already in env.S via _greedy_seed_selection

    # Phase 2: remaining n-k buyers — greedy_discount pricing
    for step in range(n - len(offered_set)):
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break
        valuations = {v: env._compute_valuation(v) for v in remaining}
        target = max(remaining, key=lambda v: valuations[v])
        val = valuations[target]
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
            if offered_price > 0:
                total_revenue += offered_price

        offered_set.add(target)
        env.offered.add(target)
        env.t += 1

    return total_revenue


def hill_climbing_trajectory(graph: nx.Graph, cfg) -> List[Tuple]:
    """Generate expert trajectory from hill_climbing_baseline for GAIL.

    Same algorithm as hill_climbing_baseline() but records each step as
    (node_idx, discount, marginal_revenue) for use in imitation learning.

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        List of (node_idx, discount, marginal_revenue) tuples.
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

    # Phase 1: greedy IM seeds → free (discount=1.0, revenue=0)
    seed_set = _greedy_seed_selection(graph, env, k)
    for node in seed_set:
        idx = env.node_to_idx[node]
        trajectory.append((idx, 1.0, 0.0))
        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    # Phase 2: greedy_discount pricing for remaining buyers
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


def local_search_baseline(graph: nx.Graph, cfg) -> float:
    """Local-search seed refinement + greedy pricing (Babaei et al. 2013).

    Improves upon hill_climbing by trying to swap out seeds for better ones:
      1. Start from greedy IM seed set (k seeds).
      2. For N_SWAP iterations: randomly swap one seed with one non-seed.
         Keep the swap if total influence on non-seeds improves.
      3. Apply greedy_discount pricing to ALL n buyers with the refined seeds first.

    Local-search often outperforms hill-climbing on modular/clustered graphs
    because it can escape local optima in the seed set.

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        Total revenue collected.
    """
    rng = np.random.default_rng(cfg.project.seed)
    env = _make_env(graph, cfg)
    env.reset()

    k = cfg.budget.k
    nodes = list(graph.nodes())
    n = len(nodes)
    degrees = dict(graph.degree())

    # ── Phase 1: greedy IM seed initialisation ────────────────────────────────
    seed_set_nodes = set(_greedy_seed_selection(graph, env, k))

    # ── Phase 2: local swap refinement ───────────────────────────────────────
    def _compute_total_influence(seeds: set) -> float:
        """Proxy: sum influence on ALL non-seed nodes given this seed set."""
        env2 = _make_env(graph, cfg)
        env2.reset()
        env2.S = set(seeds)
        env2._influence_cache = {}
        return sum(env2.get_current_influence(v) for v in nodes if v not in seeds)

    n_swaps = max(10, k * 3)
    current_score = _compute_total_influence(seed_set_nodes)
    non_seeds = [v for v in nodes if v not in seed_set_nodes]

    for _ in range(n_swaps):
        if not non_seeds or not seed_set_nodes:
            break
        seed_out = rng.choice(list(seed_set_nodes))
        seed_in  = rng.choice(non_seeds)
        candidate = (seed_set_nodes - {seed_out}) | {seed_in}
        score = _compute_total_influence(candidate)
        if score > current_score:
            seed_set_nodes = candidate
            non_seeds = [v for v in nodes if v not in seed_set_nodes]
            current_score = score

    # ── Phase 3: greedy_discount pricing with refined seeds ───────────────────
    sorted_degrees = sorted(degrees.values(), reverse=True)
    region_boundaries = [sorted_degrees[min(int(i * n / 6), n - 1)]
                         for i in range(7)]
    region_boundaries[-1] = 0

    env2 = _make_env(graph, cfg)
    env2.reset()
    env2.S = seed_set_nodes
    env2._influence_cache = {}

    offered_set: set = set(seed_set_nodes)
    for node in seed_set_nodes:
        env2.offered.add(node)
        env2.t += 1

    total_revenue = 0.0
    for _ in range(n - len(offered_set)):
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break
        valuations = {v: env2._compute_valuation(v) for v in remaining}
        target = max(remaining, key=lambda v: valuations[v])
        val = valuations[target]
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
            env2.S.add(target)
            env2._influence_cache = {}
            if offered_price > 0:
                total_revenue += offered_price

        offered_set.add(target)
        env2.offered.add(target)
        env2.t += 1

    return total_revenue


# ════════════════════════════════════════════════════════════════════════════════
# GROUP C — PriSCa (TKDE 2025)   Joint pricing approximation
# ════════════════════════════════════════════════════════════════════════════════

def prisca_baseline(graph: nx.Graph, cfg) -> float:
    """PriSCa — simplified single-product joint pricing (TKDE 2025).

    Based on: "Multi-Grade Revenue Maximization for Promotional and
    Competitive Viral Marketing in Social Networks" (TKDE 2025).

    This implements the single-product adaptation of PriSCa:

    Algorithm:
      At each step t:
        1. For every not-yet-offered buyer i, compute:
             WRP(i) = current influence normalized in [0, 1]
                    = influence_from_S(i) / total_weight(i)
             val(i) = Rayleigh valuation at WRP(i)
        2. Select buyer i* = argmax_i  WRP(i) × val(i)
           (greedy selection: who gives most expected revenue NOW)
        3. Price  p(i*) = WRP(i*) × val(i*)
           (pricing formula: expected revenue = acceptance_prob × revenue)
        4. i* accepts iff p(i*) ≤ val(i*)  ← always true since WRP ≤ 1
        5. If accepted: add to S, update influence for neighbors
        6. Revenue += p(i*)

    Note: This is our single-product interpretation of PriSCa.
    We state this clearly in the paper.

    Args:
        graph: Social network graph.
        cfg:   OmegaConf DictConfig.

    Returns:
        Total revenue collected.
    """
    env = _make_env(graph, cfg)
    env.reset()

    nodes = env.nodes
    offered_set: set = set()
    total_revenue = 0.0

    for _ in range(env.n):
        remaining = [v for v in nodes if v not in offered_set]
        if not remaining:
            break

        # Step 1+2: greedy selection by WRP × valuation
        best_node = None
        best_score = -1.0
        for node in remaining:
            wrp = env.get_current_influence(node)   # ∈ [0, 1]
            val = env._compute_valuation(node)
            score = wrp * val
            if score > best_score:
                best_score = score
                best_node = node

        if best_node is None:
            break

        # Step 3+4: price = WRP × val; always accepted
        wrp_best = env.get_current_influence(best_node)
        val_best = env._compute_valuation(best_node)
        price = wrp_best * val_best

        # Step 5: add to S → cascade
        env.S.add(best_node)
        env._influence_cache = {}
        total_revenue += price

        offered_set.add(best_node)
        env.offered.add(best_node)
        env.t += 1

    return total_revenue


# ════════════════════════════════════════════════════════════════════════════════
# GROUP B — Decoupled GNN baselines
# ════════════════════════════════════════════════════════════════════════════════

def _apply_greedy_pricing_to_order(
    graph: nx.Graph,
    cfg,
    node_order: List,
) -> float:
    """Apply greedy_discount pricing to a fixed node visitation order.

    This is the "decoupled" evaluation: a GNN provides the ordering of nodes,
    and greedy_discount handles the pricing.  The GNN replaces the hand-crafted
    IM seed selection; the revenue mechanism is identical to greedy_discount.

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig.
        node_order: List of node identifiers (NetworkX node IDs) in visit order.
                    Must cover all n nodes exactly once.

    Returns:
        Total revenue when following node_order with greedy_discount pricing.
    """
    env = _make_env(graph, cfg)
    env.reset()

    nodes = env.nodes
    n = env.n
    degrees = dict(graph.degree())
    sorted_degrees = sorted(degrees.values(), reverse=True)
    region_boundaries = [sorted_degrees[min(int(i * n / 6), n - 1)]
                         for i in range(7)]
    region_boundaries[-1] = 0

    total_revenue = 0.0
    offered_set: set = set()

    for node in node_order:
        if node in offered_set:
            continue
        val = env._compute_valuation(node)
        deg = degrees.get(node, 0)

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
            env.S.add(node)
            env._influence_cache = {}
            if offered_price > 0:
                total_revenue += offered_price

        offered_set.add(node)
        env.offered.add(node)
        env.t += 1

    return total_revenue


def _graph_to_edgelist_file(graph: nx.Graph, path: str) -> None:
    """Write a NetworkX graph to a ToupleGDD-compatible edge-list text file.

    Format: one edge per line "src dst weight" (0-indexed integer node IDs).
    Node IDs are renumbered 0..n-1 so ToupleGDD's integer indexing works.

    Args:
        graph: NetworkX graph.
        path:  Output file path.
    """
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
    """Call ToupleGDD subprocess shim and return ordered seed list.

    Writes graph to temp file, runs `python touplegdd_seed_printer.py`
    from cfg.baselines.touple_gdd_dir, parses JSON {"seeds": [...]}.

    Args:
        graph:               Social network graph.
        cfg:                 OmegaConf DictConfig with baselines.touple_gdd_dir.
        model_name:          "S2V_DQN" or "Tripling".
        checkpoint_filename: Filename of .ckpt relative to touple_gdd_dir.

    Returns:
        Ordered list of node indices selected by the model, or None on error.
    """
    touple_dir = str(cfg.baselines.touple_gdd_dir)
    ckpt_path  = os.path.join(touple_dir, checkpoint_filename)
    shim_path  = os.path.join(touple_dir, "touplegdd_seed_printer.py")

    for path, label in [(touple_dir, "touple_gdd_dir"),
                        (ckpt_path,  f"{model_name} checkpoint"),
                        (shim_path,  "touplegdd_seed_printer.py")]:
        if not os.path.exists(path):
            warnings.warn(f"[Group-B] {label} not found: {path}. "
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
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            warnings.warn(f"[Group-B] {model_name} subprocess error: "
                          f"{result.stderr[:200]}. Skipping.", stacklevel=3)
            return None

        data = json.loads(result.stdout.strip())
        if "error" in data:
            warnings.warn(f"[Group-B] {model_name} returned error: "
                          f"{data['error']}. Skipping.", stacklevel=3)
            return None

        # seeds are 0-indexed ints → map back to graph node IDs
        seed_indices = data.get("seeds", [])
        seed_nodes   = [nodes[i] for i in seed_indices if i < len(nodes)]
        # Full traversal order: seeds first, then remaining nodes by degree
        non_seeds = [v for v in nodes if v not in set(seed_nodes)]
        deg = dict(graph.degree())
        non_seeds.sort(key=lambda v: deg[v], reverse=True)
        return seed_nodes + non_seeds

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        warnings.warn(f"[Group-B] {model_name} failed: {e}. Skipping.", stacklevel=3)
        return None
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def _run_decoupled_gnn_baseline(
    graph: nx.Graph,
    cfg,
    node_order: List,
    n_trials: int = 3,
) -> Dict[str, float]:
    """Apply greedy_discount pricing to a GNN-provided node order over n_trials.

    Each trial re-samples link weights (env.reset) but applies the SAME node
    order — testing how the GNN order performs under link-weight uncertainty.

    Args:
        graph:      Social network graph.
        cfg:        OmegaConf DictConfig.
        node_order: Node visit order from the GNN (length n).
        n_trials:   Number of link-weight sampling trials.

    Returns:
        Dict with "mean_revenue", "std_revenue".
    """
    revenues = []
    for trial in range(n_trials):
        trial_cfg = _override_seed(cfg, cfg.project.seed + trial)
        rev = _apply_greedy_pricing_to_order(graph, trial_cfg, node_order)
        revenues.append(rev)
    return {
        "mean_revenue": float(np.mean(revenues)),
        "std_revenue":  float(np.std(revenues)),
    }


def s2v_dqn_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[Dict[str, float]]:
    """S2V-DQN (Dai et al. 2017) node ordering + greedy_discount pricing.

    Loads the pretrained S2V-DQN checkpoint from cfg.baselines.touple_gdd_dir,
    runs the model to obtain the optimal seed ordering, then applies
    greedy_discount pricing to that sequence.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig (must have cfg.baselines.touple_gdd_dir
                  and cfg.baselines.s2v_dqn_checkpoint).
        n_trials: Number of pricing trials (link-weight samples).

    Returns:
        Dict {"mean_revenue": float, "std_revenue": float} or None if unavailable.
    """
    node_order = _call_touplegdd_shim(
        graph, cfg, "S2V_DQN", str(cfg.baselines.s2v_dqn_checkpoint))
    if node_order is None:
        return None
    return _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)


def touple_gdd_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[Dict[str, float]]:
    """ToupleGDD/Tripling (Chen et al. 2022) node ordering + greedy_discount pricing.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Number of pricing trials.

    Returns:
        Dict {"mean_revenue": float, "std_revenue": float} or None if unavailable.
    """
    node_order = _call_touplegdd_shim(
        graph, cfg, "Tripling", str(cfg.baselines.touple_gdd_checkpoint))
    if node_order is None:
        return None
    return _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)


def dgn_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[Dict[str, float]]:
    """DGN (Wang et al. 2024) node ordering + greedy_discount pricing.

    Stub — checkpoint not yet available.  Returns None with a warning.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig (cfg.baselines.dgn_checkpoint).
        n_trials: Number of pricing trials.

    Returns:
        None (checkpoint not found).
    """
    ckpt = str(cfg.baselines.dgn_checkpoint)
    if not os.path.exists(ckpt):
        warnings.warn(f"[Group-B] DGN checkpoint not found: {ckpt}. Skipping.")
        return None
    # Placeholder: implement DGN model loading here when checkpoint available
    warnings.warn("[Group-B] DGN model loading not yet implemented.")
    return None


def wsdm_gnn_im_rl_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[Dict[str, float]]:
    """WSDM GNN-IM-RL (Seyedin et al. 2027) node ordering + greedy_discount pricing.

    Loads the WSDM GNN-IM-RL JointPolicy checkpoint from cfg.baselines.wsdm_gnn_checkpoint,
    runs a greedy episode to obtain the node visit order, then applies
    greedy_discount pricing to that sequence (decoupled).

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Number of pricing trials.

    Returns:
        Dict {"mean_revenue": float, "std_revenue": float} or None if unavailable.
    """
    ckpt = str(cfg.baselines.wsdm_gnn_checkpoint)
    if not os.path.exists(ckpt):
        warnings.warn(f"[Group-B] WSDM GNN-IM-RL checkpoint not found: {ckpt}. Skipping.")
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
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        policy.eval()

        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())

        # Greedy rollout to extract node order (ignore discounts — decoupled)
        env = _make_env(graph, cfg)
        env.reset()
        node_order = []

        with torch.no_grad():
            for _ in range(n):
                available = env.available_nodes
                if not available:
                    break
                feats = compute_node_features(
                    graph=graph, static_features=static,
                    S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, n=n, k=n, env=env)
                data  = graph_to_pyg_data(graph, feats, device)
                mask  = get_available_mask(n, frozenset(env.offered), nodes, device)
                node_idx, _, _ = policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=True)
                if node_idx not in available:
                    node_idx = available[0]
                node_order.append(nodes[node_idx])
                env.offered.add(nodes[node_idx])
                env.t += 1

        return _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)

    except Exception as e:
        warnings.warn(f"[Group-B] WSDM GNN-IM-RL failed: {e}")
        return None


def wsdm_gail_rl_decoupled(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Optional[Dict[str, float]]:
    """WSDM GAIL-RL-Rich (Seyedin et al. 2027) node ordering + greedy_discount pricing.

    Same as wsdm_gnn_im_rl_decoupled but uses the GAIL-trained checkpoint.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Number of pricing trials.

    Returns:
        Dict {"mean_revenue": float, "std_revenue": float} or None if unavailable.
    """
    ckpt = str(cfg.baselines.wsdm_gail_checkpoint)
    if not os.path.exists(ckpt):
        warnings.warn(f"[Group-B] WSDM GAIL-RL-Rich checkpoint not found: {ckpt}. Skipping.")
        return None

    # Reuse the same loading logic as wsdm_gnn_im_rl_decoupled but with gail ckpt
    try:
        import torch
        from src.models.encoders.graphsage import GraphSAGEEncoder
        from src.models.policies.joint_policy import JointPolicy
        from src.utils.features import compute_static_features, compute_node_features
        from src.utils.helpers import graph_to_pyg_data, get_available_mask

        device = torch.device("cpu")
        enc = GraphSAGEEncoder(
            in_dim=cfg.features.dim, hidden_dim=cfg.encoder.hidden_dim,
            n_layers=cfg.encoder.n_layers, dropout=cfg.encoder.dropout,
        ).to(device)
        policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
        policy.load_state_dict(torch.load(ckpt, map_location="cpu"))
        policy.eval()

        static = compute_static_features(graph)
        n = graph.number_of_nodes()
        nodes = list(graph.nodes())
        env = _make_env(graph, cfg)
        env.reset()
        node_order = []

        with torch.no_grad():
            for _ in range(n):
                available = env.available_nodes
                if not available:
                    break
                feats = compute_node_features(
                    graph=graph, static_features=static,
                    S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, n=n, k=n, env=env)
                data  = graph_to_pyg_data(graph, feats, device)
                mask  = get_available_mask(n, frozenset(env.offered), nodes, device)
                node_idx, _, _ = policy.select_and_price(
                    data.x, data.edge_index, mask, greedy=True)
                if node_idx not in available:
                    node_idx = available[0]
                node_order.append(nodes[node_idx])
                env.offered.add(nodes[node_idx])
                env.t += 1

        return _run_decoupled_gnn_baseline(graph, cfg, node_order, n_trials)

    except Exception as e:
        warnings.warn(f"[Group-B] WSDM GAIL-RL-Rich failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════════════════════
# Comprehensive runners for all 3 groups
# ════════════════════════════════════════════════════════════════════════════════

def run_group_a_baselines(
    graph: nx.Graph,
    cfg,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Run all 8 Group A (hand-crafted) baselines and return mean revenues.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: MC trials over link-weight samples for each baseline.

    Returns:
        Dict mapping method name → mean revenue over n_trials.
    """
    methods = {
        "random":         random_baseline,
        "myopic_full":    myopic_full_price,
        "ie_strategy":    ie_strategy,
        "mu_discount":    mu_discount,
        "sigma_discount": sigma_discount,
        "greedy_discount": greedy_discount,
        "hill_climbing":  hill_climbing_baseline,
        "local_search":   local_search_baseline,
    }
    results: Dict[str, float] = {}
    for name, fn in methods.items():
        revenues = []
        for trial in range(n_trials):
            trial_cfg = _override_seed(cfg, cfg.project.seed + trial)
            revenues.append(fn(graph, trial_cfg))
        results[name] = float(np.mean(revenues))
    return results


def run_group_b_baselines(
    graph: nx.Graph,
    cfg,
    n_trials: int = 3,
) -> Dict[str, Optional[float]]:
    """Run all 5 Group B (decoupled GNN) baselines.

    Baselines that cannot run (checkpoint missing) return None.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: Pricing trials per method.

    Returns:
        Dict mapping method name → mean revenue (or None if unavailable).
    """
    results: Dict[str, Optional[float]] = {}
    for name, fn in [
        ("s2v_dqn_decoupled",         s2v_dqn_decoupled),
        ("touple_gdd_decoupled",       touple_gdd_decoupled),
        ("dgn_decoupled",              dgn_decoupled),
        ("wsdm_gnn_im_rl_decoupled",   wsdm_gnn_im_rl_decoupled),
        ("wsdm_gail_rl_decoupled",     wsdm_gail_rl_decoupled),
    ]:
        t0 = time.time()
        out = fn(graph, cfg, n_trials)
        elapsed = time.time() - t0
        if out is None:
            results[name] = None
        else:
            results[name] = out["mean_revenue"]
    return results


def run_group_c_baselines(
    graph: nx.Graph,
    cfg,
    n_trials: int = 5,
) -> Dict[str, float]:
    """Run Group C (approximation algorithm) baselines.

    Args:
        graph:    Social network graph.
        cfg:      OmegaConf DictConfig.
        n_trials: MC trials.

    Returns:
        Dict mapping method name → mean revenue.
    """
    revenues = []
    for trial in range(n_trials):
        trial_cfg = _override_seed(cfg, cfg.project.seed + trial)
        revenues.append(prisca_baseline(graph, trial_cfg))
    return {"prisca": float(np.mean(revenues))}
