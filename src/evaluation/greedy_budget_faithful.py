"""greedy_budget_faithful.py — Faithful Greedy+Budget port.

Identical to the unconstrained greedy_discount (dynamic re-ranking by
estimated valuation, Rayleigh tier pricing) with ONE addition: the budget
feasibility rule from §3.2.

Difference from the existing greedy_discount_budget (static version):
  - Static: visits nodes in degree-descending order (fixed before loop).
  - Faithful (this file): re-ranks by estimated valuation at every step,
    same as the unconstrained greedy_discount.

Budget rule (SKIP-never-reprice, identical to static):
  If B - c + tier_price < 0 → skip that buyer, continue to next-best.
  Campaign ends when env.available_nodes is exhausted OR all remaining
  buyers are unaffordable.

Profit = R - c * |S_T|  (cost on accepted items only, same accounting).
"""
from __future__ import annotations
import math, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import networkx as nx

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
# Import the EXACT same influence + cache functions as unconstrained greedy_discount
from src.evaluation.baselines import _compute_normalized_infl, _invalidate_caches

C_DEFAULT   = 0.3
B_RAYLEIGH  = 1.0
TIER1_PRICE = math.sqrt(-2 * math.log(1 - 2.0/6.0)) * B_RAYLEIGH   # f(2/6) ≈ 0.534
TIER2_PRICE = math.sqrt(-2 * math.log(1 - 4.0/6.0)) * B_RAYLEIGH   # f(4/6) ≈ 0.548


def _make_env(graph, B, c, seed=0, weight_high=2.0, n_mc=200):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=seed,
                          weight_high=weight_high, n_mc_samples=n_mc)
    return BudgetRevenueEnv(graph, cfg)


def _tier_price(infl: float) -> float:
    if infl < 2.0 / 6.0:
        return 0.0
    elif infl < 4.0 / 6.0:
        return TIER1_PRICE
    else:
        return TIER2_PRICE


def greedy_discount_budget_faithful(
    graph: nx.Graph,
    B: float,
    c: float = C_DEFAULT,
    n_trials: int = 10,
    weight_high: float = 2.0,
    n_mc: int = 200,
) -> dict:
    """Faithful Greedy+Budget: dynamic est-val re-ranking + budget skip.

    Returns per-trial arrays for profit, revenue, |S_T|, n_below, n_skips.
    """
    profits=[]; revenues=[]; sts=[]; belows=[]; skips_list=[]

    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial,
                        weight_high=weight_high, n_mc=n_mc)
        env.reset()
        lw = env._link_weights   # same as unconstrained greedy_discount
        revenue=0.0; n_below=0; n_skips=0

        while True:
            remaining = [v for v in env.nodes if v not in env.offered]
            if not remaining:
                break
            if env._check_bankrupt():
                break

            # Dynamic re-rank: highest estimated valuation — identical to unconstrained
            target = max(remaining, key=lambda v: env._estimate_valuation(v))
            # SAME influence function as unconstrained greedy_discount
            infl  = _compute_normalized_infl(graph, target, env.S, lw)
            price = _tier_price(infl)

            # Budget feasibility: SKIP-never-reprice
            if env.B - c + price < -1e-9:
                env.offered.add(target)
                env.t += 1
                n_skips += 1
                if env._check_bankrupt():
                    break
                continue

            # Offer and evaluate
            true_val = env._true_valuation(target)
            env.offered.add(target)
            env.t += 1

            if price == 0.0:
                env.S.add(target)
                env.B -= c
                _invalidate_caches(env, target)   # same as unconstrained
            elif true_val >= price:
                env.S.add(target)
                env.B = env.B - c + price
                _invalidate_caches(env, target)
                revenue += price
                if price < c:
                    n_below += 1

        S_T = len(env.S)
        profits.append(revenue - c * S_T)
        revenues.append(revenue)
        sts.append(S_T)
        belows.append(n_below)
        skips_list.append(n_skips)

    def _s(arr):
        return {"mean": float(np.mean(arr)), "std": float(np.std(arr)),
                "all": [float(x) for x in arr]}

    return {
        "profit":   _s(profits),
        "revenue":  _s(revenues),
        "n_in_S":   _s(sts),
        "n_below":  _s(belows),
        "n_skips":  _s(skips_list),
    }
