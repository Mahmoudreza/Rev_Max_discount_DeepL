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
