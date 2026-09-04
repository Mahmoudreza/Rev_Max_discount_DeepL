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

import heapq
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import networkx as nx

import numpy as np
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.baselines import _greedy_seed_selection, _invalidate_caches

# ── k_seeds for the IE influence phase (matches unconstrained Table-1 eval) ──
IE_K_SEEDS: int = 30


def _greedy_seed_selection_celf(graph: nx.Graph, env, k: int) -> list:
    """CELF (lazy) greedy seed selection — exact same criterion as
    _greedy_seed_selection but O(n·d²) instead of O(k·n·d²) by only
    recomputing marginal gains for 2-hop neighbours of each new seed.

    Uses true link weights (deterministic), so results are identical to the
    naive version and 30/30 seed sets match on any network.
    """
    nb_sets = {v: set(graph.neighbors(v)) for v in graph.nodes()}
    tw = {v: sum(env._link_weights.get((v, nb), 0.0) for nb in nb_sets[v])
          for v in graph.nodes()}

    S_sel: set = set()
    remaining: set = set(graph.nodes())

    def _gain(node: int) -> float:
        g = 0.0
        for nb in nb_sets.get(node, set()):
            if nb not in remaining or nb in S_sel:
                continue
            tw_nb = tw.get(nb, 0.0)
            if tw_nb <= 0:
                continue
            infl_S = sum(env._link_weights.get((nb, j), 0.0)
                         for j in (S_sel & nb_sets.get(nb, set()))) / tw_nb
            w = env._link_weights.get((nb, node), 0.0)
            g += min(1.0, infl_S + w / tw_nb) - infl_S
        return g

    # Initial full evaluation
    gains = {v: _gain(v) for v in graph.nodes()}
    heap: list = [(-g, v) for v, g in gains.items()]
    heapq.heapify(heap)
    stale: set = set()

    selected: list = []
    for _ in range(min(k, graph.number_of_nodes())):
        while heap:
            neg_g, node = heapq.heappop(heap)
            if node not in remaining:
                continue
            if node not in stale:
                # Fresh — accept this seed
                selected.append(node)
                S_sel.add(node)
                env.S.add(node)
                _invalidate_caches(env, node)
                remaining.discard(node)
                # Mark 2-hop neighbourhood as stale
                for nb in nb_sets.get(node, set()):
                    for nb2 in nb_sets.get(nb, set()):
                        if nb2 in remaining:
                            stale.add(nb2)
                break
            else:
                # Stale — recompute and re-push
                new_g = _gain(node)
                stale.discard(node)
                heapq.heappush(heap, (-new_g, node))
    return selected


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
    seed_orderings: dict = None,
) -> dict:
    """IE-Strategy (Babaei et al.) under the feasibility rule of §3.2.

    Phase 1 (influence): CELF greedy seed selection (or pre-computed ordering
      via seed_orderings[trial]).  Compute once per (network, trial) and
      pass via seed_orderings to avoid recomputing across k-values.
    Phase 2 (exploit): candidates pre-ranked by get_current_influence (fast,
      deterministic, N_MC=∞ equivalent); price at est_val(N_MC=200).

    seed_orderings: dict {trial: list_of_nodes} — if provided, Phase 1 is
      skipped (use pre-computed CELF ordering); otherwise computed with CELF.

    Returns the same aggregated structure as greedy_discount_budget.
    """
    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        revenue         = 0.0
        n_paid_accepted = 0
        n_subsidized    = 0

        # ── Phase 1: influence seeds (CELF or pre-computed) ───────────────
        if seed_orderings is not None and trial in seed_orderings:
            # Replay pre-computed ordering: re-add to env.S (env was just reset)
            seed_list = seed_orderings[trial]
            for node in seed_list:
                env.S.add(node)
                _invalidate_caches(env, node)
        else:
            seed_list = _greedy_seed_selection_celf(graph, env, k_seeds)

        for node in seed_list:
            if env.B - c >= -1e-9:
                env.B -= c
            else:
                env.S.discard(node)
                env._influence_cache.clear()
                env._true_val_cache.clear()
                env._est_val_cache.clear()
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)

        # ── Phase 2: pre-rank by influence (fast), price at N_MC=200 ──────
        # Rank remaining candidates by get_current_influence (O(deg), cached)
        # so high-influence buyers are offered first. Price at full N_MC=200.
        remaining_phase2 = [n for n in env.nodes if n not in env.offered]
        remaining_phase2.sort(key=lambda v: env.get_current_influence(v), reverse=True)

        for node in remaining_phase2:
            if node in env.offered:
                continue
            if env._check_bankrupt():
                break

            p = env._estimate_valuation(node)
            if p <= 0.0:
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            if p < c:
                n_subsidized += 1

            if env.B - c + p < -1e-9:
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            true_val = env._true_valuation(node)
            if true_val >= p:
                env.S.add(node)
                env.B = env.B - c + p
                env._influence_cache.clear()
                env._true_val_cache.clear()
                env._est_val_cache.clear()
                revenue         += p
                n_paid_accepted += 1

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
