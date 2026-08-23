"""ie_budget.py — IE-Strategy (Babaei et al.) under the §3.2 budget constraint.

Reuses the unconstrained IE helpers (seed selection + myopic pricing) from
src/evaluation/baselines.py unchanged.  The only addition is the feasibility
check and the skip rule identical to greedy_discount_budget.

k_seeds = 30  (cfg.budget.k from base_config.yaml — the Table-1 unconstrained value).

Design notes
------------
* _greedy_seed_selection(graph, env, k) ADDS seeds to env.S during selection.
  For seeds that fail the budget feasibility check (B - c < 0), we REMOVE them
  from env.S (undo the selection's mutation) and clear all caches.
* Phase 2 only offers nodes with est_val > 0, matching the unconstrained IE
  protocol (nodes with zero valuation are not offered).
* Cache clearing uses full dict-clear after each Phase-2 acceptance, matching
  the unconstrained ie_strategy exactly.
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import networkx as nx

import numpy as np
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.baselines import _greedy_seed_selection   # reused unchanged

# ── k_seeds for the IE influence phase (matches unconstrained Table-1 eval) ──
IE_K_SEEDS: int = 30


def _make_env(graph, B, c, seed=0, weight_high=2.0):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=seed,
                          weight_high=weight_high)
    return BudgetRevenueEnv(graph, cfg)


def _aggregate(results: list) -> dict:
    """Aggregate a list of per-trial dicts — same schema as budget_baselines._aggregate."""
    if not results:
        return {}
    agg: dict = {}
    for k in results[0].keys():
        vals = [r[k] for r in results if k in r]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "all": vals}
        elif isinstance(vals[0], list):
            agg[k] = vals           # keep all trajectory lists
        else:
            agg[k] = vals[0]
    return agg


def ie_strategy_budget(
    graph: nx.Graph,
    B: float,
    c: float,
    k_seeds: int = IE_K_SEEDS,
    n_trials: int = 5,
    weight_high: float = 2.0,
) -> dict:
    """IE-Strategy (Babaei et al.) under the feasibility rule of §3.2.

    Phase 1 (influence): _greedy_seed_selection adds seeds to env.S during
      selection.  We accept a seed (keep in S, deduct c) iff B - c >= 0.
      Unaffordable seeds are REMOVED from env.S and all caches cleared.
    Phase 2 (exploit): remaining buyers in env.nodes order, offer only if
      p = est_val(i) > 0.  Execute iff B - c + p >= 0; otherwise SKIP.
      On acceptance: add to S, B <- B - c + p, clear all caches (matching
      unconstrained ie_strategy exactly).

    Returns the same aggregated structure as greedy_discount_budget.
    """
    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        revenue         = 0.0
        n_paid_accepted = 0
        n_subsidized    = 0

        # ── Phase 1: influence seeds ───────────────────────────────────────
        # _greedy_seed_selection adds ALL k_seeds nodes to env.S during selection.
        seed_list = _greedy_seed_selection(graph, env, k_seeds)

        for node in seed_list:
            if env.B - c >= -1e-9:
                # Affordable: node already in S (from selection); deduct budget
                env.B -= c
            else:
                # Unaffordable: undo S addition, clear all caches
                env.S.discard(node)
                env._influence_cache.clear()
                env._true_val_cache.clear()
                env._est_val_cache.clear()
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)

        # ── Phase 2: exploit remaining buyers at myopic price ──────────────
        for node in env.nodes:
            if node in env.offered:
                continue
            if env._check_bankrupt():
                break

            p = env._estimate_valuation(node)   # myopic price (no discount)
            if p <= 0.0:
                # Zero-valuation: skip (matches unconstrained IE "if est_val > 0")
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            if p < c:
                n_subsidized += 1

            # feasibility check — NEVER reprice
            if env.B - c + p < -1e-9:
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            true_val = env._true_valuation(node)
            if true_val >= p:
                env.S.add(node)
                env.B = env.B - c + p
                # Full cache clear — matches unconstrained ie_strategy exactly
                env._influence_cache.clear()
                env._true_val_cache.clear()
                env._est_val_cache.clear()
                revenue         += p
                n_paid_accepted += 1
            # rejection: no budget change

            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)

        results.append({
            "revenue":           revenue,
            "n_accepted":        n_paid_accepted,
            "n_in_S":            len(env.S),
            "n_offered":         len(env.offered),
            "n_subsidized":      n_subsidized,
            "min_budget":        min(env.budget_history) if env.budget_history else env.B,
            "final_budget":      env.B,
            "bankrupt":          env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
        })

    return _aggregate(results)


def ie_strategy_budget_aware(
    graph: nx.Graph,
    B: float,
    c: float,
    k_seeds_max: int = IE_K_SEEDS,
    n_trials: int = 5,
    weight_high: float = 2.0,
) -> dict:
    """IE-Strategy with budget-aware seed count.

    k_seeds = min(k_seeds_max, floor(B / c))
    so Phase-1 never attempts more seeds than the budget can fund.
    Everything else identical to ie_strategy_budget.
    """
    import math
    k = min(k_seeds_max, math.floor(B / c))
    return ie_strategy_budget(
        graph, B, c,
        k_seeds=k,
        n_trials=n_trials,
        weight_high=weight_high,
    )
