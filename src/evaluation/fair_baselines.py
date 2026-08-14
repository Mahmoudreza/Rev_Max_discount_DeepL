"""src/evaluation/fair_baselines.py

Fair-Greedy baseline for revenue maximization with group fairness.
NEW FILE — does NOT modify any existing src/ file.

Fair-Greedy = Greedy-Discount pricing logic + maxmin adoption-rate node ordering.
Pricing logic (influence tiers, acceptance) is IDENTICAL to greedy_discount_trajectory;
only the node selection order changes.
"""
from __future__ import annotations
import sys
sys.path.insert(0, ".")

from typing import List, Dict
import numpy as np
import networkx as nx

# Import helpers from baselines (READ ONLY — no modification)
from src.evaluation.baselines import (
    _make_env,
    _compute_normalized_infl,
    _rayleigh_price,
)


def fair_greedy_discount_trajectory(
    graph: nx.Graph,
    labels: np.ndarray,
    cfg,
    seed: int = 0,
) -> List[Dict]:
    """Greedy-Discount pricing + maxmin group adoption-rate ordering.

    IDENTICAL to greedy_discount_trajectory in ALL of:
      - Influence-tier pricing: FREE / f(2/6) / f(4/6) at thresholds 2/6, 4/6
      - Estimated-vs-true valuation acceptance
      - One offer per buyer (n steps total)
    (calls the same _make_env, _compute_normalized_infl, _rayleigh_price helpers)

    ONLY the node ORDER changes:
      - Maintain per-group degree-descending queues (precomputed, static)
      - At each step: compute current adoption rates rho_g over ACCEPTED buyers
        rho_g = |accepted buyers in group g| / |g|
      - Offer to the highest-degree unoffered node of argmin_g rho_g
        Tie in rho → global degree order (higher degree first)
        A group with an empty (all-offered) queue yields to the other group

    Args:
        graph:  Social network graph (nodes 0..n-1).
        labels: np.array of 0/1 per node (1 = minority group B).
        cfg:    OmegaConf DictConfig (same as used in baselines.py).
        seed:   Random seed (passed to env; pricing itself is deterministic).

    Returns:
        List of dicts per step (length n), each with keys:
            node_idx, discount, marginal_gain, price, accepted.
        Same format as greedy_discount_trajectory — plug-in compatible.
    """
    env = _make_env(graph, cfg)
    env.reset()

    n   = env.n
    b   = float(cfg.influence.b)
    lw  = env._link_weights
    nodes = list(graph.nodes())

    # Pre-compute static degree ordering per group
    deg = dict(graph.degree())
    # Sort each group by degree descending (tie-break: node index ascending)
    group0_queue = sorted(
        [v for v in nodes if labels[env.node_to_idx[v]] == 0],
        key=lambda v: (-deg[v], v)
    )
    group1_queue = sorted(
        [v for v in nodes if labels[env.node_to_idx[v]] == 1],
        key=lambda v: (-deg[v], v)
    )

    offered_set: set = set()
    acc_per_group = {0: 0, 1: 0}   # accepted count per group
    nA = int((labels == 0).sum())
    nB = int((labels == 1).sum())
    group_sizes = {0: max(nA, 1), 1: max(nB, 1)}

    trajectory: List[Dict] = []

    for _ in range(n):
        # Compute current adoption rates
        rho = {g: acc_per_group[g] / group_sizes[g] for g in (0, 1)}

        # Find next target: argmin rho group, highest-degree unoffered node
        target = None
        for _ in range(2):   # at most 2 tries (one per group)
            # Candidate group ordering: lower rho first; tie → higher degree node
            if rho[0] <= rho[1]:
                grp_order = [0, 1]
            else:
                grp_order = [1, 0]

            for g in grp_order:
                queue = group0_queue if g == 0 else group1_queue
                for v in queue:
                    if v not in offered_set:
                        target = v
                        break
                if target is not None:
                    break
            if target is not None:
                break

        if target is None:
            break   # all nodes offered

        # ── Pricing: IDENTICAL to greedy_discount_trajectory ──────────────
        infl    = _compute_normalized_infl(graph, target, env.S, lw)
        if infl < 2.0 / 6.0:
            price = 0.0
        elif infl < 4.0 / 6.0:
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            price = _rayleigh_price(4.0 / 6.0, b)

        est_val  = env._estimate_valuation(target)
        true_val = env._true_valuation(target)
        node_idx = env.node_to_idx[target]
        grp_of_target = int(labels[node_idx])

        if price == 0.0:
            env.S.add(target)
            env._influence_cache = {}
            disc    = 1.0
            marginal = 0.0
            accepted = True
        elif true_val >= price:
            env.S.add(target)
            env._influence_cache = {}
            disc    = max(0.0, 1.0 - price / est_val) if est_val > 0 else 0.0
            marginal = price
            accepted = True
        else:
            disc    = max(0.0, 1.0 - price / est_val) if est_val > 0 else 0.0
            marginal = 0.0
            accepted = False

        if accepted and price > 0:   # only revenue-generating acceptances count
            acc_per_group[grp_of_target] += 1
        if accepted and price == 0:  # free seed also counts for adoption
            acc_per_group[grp_of_target] += 1

        trajectory.append({
            "node_idx": node_idx,
            "discount": disc,
            "marginal_gain": marginal,
            "price": price,
            "accepted": accepted,
            "est_val": est_val,      # extra field for fairness audit
        })
        offered_set.add(target)
        env.offered.add(target)
        env.t += 1

    return trajectory
