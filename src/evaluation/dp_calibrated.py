"""src/evaluation/dp_calibrated.py — Degree-calibrated DP baselines for Idea 3.

Provides three DP variants that improve over the naive two_phase_dp_budget in
budget_baselines.py by using graph-aware valuation curves instead of a
position-average approximation.

Methods (all NEW files, nothing in budget_baselines.py is modified):
  calibrate_valuation_curves  — offline calibration step, cached to disk
  dp_calibrated_budget        — calibrated DP (replaces avg_val with v_curve)
  dp_receding_budget          — receding-horizon re-solve every N steps
  dp_upper_bound              — oracle bound with true valuations, forced acceptance
"""

from __future__ import annotations

import hashlib
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.baselines import _rayleigh_price

_CACHE_DIR = "results/logs"


# ── Graph hash ─────────────────────────────────────────────────────────────────

def _graph_hash(graph) -> str:
    """Deterministic 12-char hex hash of graph topology (n, m, first-100 degrees)."""
    deg_str = "_".join(str(d) for _, d in sorted(graph.degree())[:100])
    key = f"{graph.number_of_nodes()}_{graph.number_of_edges()}_{deg_str}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _deg_class(degree: int, boundaries: np.ndarray) -> int:
    """Return the degree-class index in [0, n_classes-1] for a given degree."""
    for i in range(len(boundaries) - 2, 0, -1):
        if degree >= boundaries[i]:
            return i
    return 0


# ── Step 1.1: Calibration ──────────────────────────────────────────────────────

def calibrate_valuation_curves(
    graph,
    cfg: BudgetEnvConfig,
    n_classes: int = 5,
    n_sims: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Measure per-class valuation curves on the actual graph.

    Runs n_sims greedy-discount simulations on the graph (large B so budget is
    not binding), records estimated valuation at each position, and aggregates
    into per-degree-class curves. Smoothed with a rolling window and cached.

    Args:
        graph:     NetworkX graph.
        cfg:       BudgetEnvConfig (weight_high used for link-weight sampling).
        n_classes: Number of degree-class buckets.
        n_sims:    Number of simulation passes.
        seed:      Base RNG seed; each sim uses seed+i.

    Returns:
        v_curve:            np.ndarray (n_classes, n) — mean est_val per class/pos.
        class_of_position:  np.ndarray (n,) int — degree class of k-th node in
                            degree-descending order.
        class_boundaries:   np.ndarray (n_classes+1,) — degree quantile edges.
    """
    n = graph.number_of_nodes()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    gh = _graph_hash(graph)
    cache_path = os.path.join(
        _CACHE_DIR, f"dp_calibration_{gh}_nc{n_classes}_ns{n_sims}.npz"
    )

    if os.path.exists(cache_path):
        data = np.load(cache_path, allow_pickle=True)
        return data["v_curve"], data["class_of_position"], data["class_boundaries"]

    # Degree-descending order (deterministic)
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg   = np.array([graph.degree(v) for v in ordering], dtype=float)

    # Compute quantile boundaries over degree values (in desc order, so quantiles
    # are taken over the distribution of degrees, not the ordering index).
    boundaries = np.quantile(all_deg, np.linspace(0.0, 1.0, n_classes + 1))
    boundaries[-1] += 1.0  # ensure last node is included in last class

    class_of_position = np.array(
        [_deg_class(int(all_deg[k]), boundaries) for k in range(n)], dtype=np.int32
    )

    val_sums   = np.zeros((n_classes, n), dtype=np.float64)
    val_counts = np.zeros((n_classes, n), dtype=np.float64)

    tier1 = _rayleigh_price(2.0 / 6.0)
    tier2 = _rayleigh_price(4.0 / 6.0)

    for sim in range(n_sims):
        # Use a very large budget so budget never limits the simulation.
        env = _make_env(
            graph, B=1e9, c=cfg.production_cost,
            seed=seed + sim, weight_high=cfg.weight_high,
        )
        env.reset()

        for k, node in enumerate(ordering):
            if node in env.offered:
                continue

            # Record estimated valuation BEFORE offering (seller's estimate)
            est_val = env._estimate_valuation(node)
            cls = int(class_of_position[k])
            val_sums[cls, k]   += est_val
            val_counts[cls, k] += 1

            # Simulate acceptance with greedy tier pricing (Babaei 2013 tiers)
            # so that S grows realistically and later nodes see real influence.
            infl = env.get_current_influence(node)
            if infl < 2.0 / 6.0:
                price = 0.0         # free seed
            elif infl < 4.0 / 6.0:
                price = tier1
            else:
                price = tier2

            true_val = env._true_valuation(node)
            if price == 0.0 or true_val >= price:
                env.S.add(node)
                for nb in env.graph.neighbors(node):
                    env._influence_cache.pop(nb, None)
                    env._true_val_cache.pop(nb, None)
                    env._est_val_cache.pop(nb, None)

            env.offered.add(node)
            env.t += 1

    # Mean curves, avoiding division by zero
    v_curve = np.where(
        val_counts > 0,
        val_sums / np.maximum(val_counts, 1.0),
        0.0,
    )

    # Smooth each class row with a rolling mean (window=25)
    window = 25
    kernel = np.ones(window) / window
    for c_idx in range(n_classes):
        smoothed = np.convolve(v_curve[c_idx], kernel, mode="same")
        # Fix boundary: fill leading near-zero region with first valid value
        first_pos = np.where(smoothed > 1e-6)[0]
        if len(first_pos) > 0:
            smoothed[:first_pos[0]] = smoothed[first_pos[0]]
        v_curve[c_idx] = smoothed

    np.savez(
        cache_path,
        v_curve=v_curve,
        class_of_position=class_of_position,
        class_boundaries=boundaries,
    )
    return v_curve, class_of_position, boundaries


# ── Internal: DP solve ─────────────────────────────────────────────────────────

def _plan_dp(
    n_total: int,
    v_curve: np.ndarray,
    class_of_position: np.ndarray,
    B: float,
    c: float,
    tiers: tuple,
    delta: float,
    k_start: int = 0,
    rate_corr: float = 1.0,
) -> List[Optional[float]]:
    """Phase-1 DP: plan optimal tier sequence.

    Args:
        n_total:           Total number of nodes in the graph.
        v_curve:           (n_classes, n_total) mean valuation curve.
        class_of_position: (n_total,) degree class per position.
        B:                 Current (remaining) budget.
        c:                 Production cost.
        tiers:             Tuple of discount fractions (e.g. 1.0=free, 0.0=full).
        delta:             Budget discretisation step.
        k_start:           First position to plan (for receding horizon).
        rate_corr:         Acceptance-rate correction multiplier for v_curve.

    Returns:
        plan: list of length (n_total - k_start); each entry is a discount
              fraction or None (skip).
    """
    n_remain = n_total - k_start
    b_steps  = max(1, int(B / delta) + 1)
    tiers_list = list(tiers)

    # dp[b_idx][k_rel] = max expected revenue from position k_start+k_rel
    #                    with budget b_idx * delta
    dp          = [[0.0] * (n_remain + 1) for _ in range(b_steps + 1)]
    tier_choice = [[-1]  * (n_remain + 1) for _ in range(b_steps + 1)]

    for k_rel in range(n_remain - 1, -1, -1):
        k_abs   = k_start + k_rel
        cls     = int(class_of_position[k_abs])
        avg_val = float(v_curve[cls, k_abs]) * rate_corr

        for b_idx in range(b_steps + 1):
            b_curr = b_idx * delta
            # Default: skip this position
            best_rev = dp[b_idx][k_rel + 1]
            best_t   = -1

            for t_idx, t_disc in enumerate(tiers_list):
                price = avg_val * (1.0 - t_disc)
                if b_curr - c + price < -1e-9:
                    continue  # unaffordable
                new_b_raw  = b_curr - c + price
                new_b_idx  = min(int(new_b_raw / delta), b_steps)
                ev = price + dp[new_b_idx][k_rel + 1]
                if ev > best_rev:
                    best_rev = ev
                    best_t   = t_idx

            dp[b_idx][k_rel]          = best_rev
            tier_choice[b_idx][k_rel] = best_t

    # Extract the optimal plan by following the greedy path from b_start
    plan  = [None] * n_remain
    b_idx = min(int(B / delta), b_steps)
    for k_rel in range(n_remain):
        k_abs = k_start + k_rel
        cls   = int(class_of_position[k_abs])
        t_i   = tier_choice[min(b_idx, b_steps)][k_rel]
        if t_i >= 0:
            disc         = tiers_list[t_i]
            plan[k_rel]  = disc
            avg_val      = float(v_curve[cls, k_abs]) * rate_corr
            price        = avg_val * (1.0 - disc)
            new_b_raw    = b_idx * delta - c + price
            b_idx        = min(max(int(new_b_raw / delta), 0), b_steps)

    return plan


# ── Internal: Plan execution ───────────────────────────────────────────────────

def _execute_plan(
    env: BudgetRevenueEnv,
    ordering: list,
    plan: List[Optional[float]],
    c: float,
    k_start: int = 0,
) -> Tuple[float, int, int]:
    """Execute a DP tier plan on a BudgetRevenueEnv.

    Walks the degree-descending ordering from k_start, applying the planned
    discount at each position. NEVER reprices: if unaffordable, SKIP.

    Args:
        env:      BudgetRevenueEnv (already reset).
        ordering: Degree-descending node list.
        plan:     Output of _plan_dp.
        c:        Production cost.
        k_start:  Offset into ordering.

    Returns:
        (revenue, n_accepted, n_subsidized)
    """
    revenue      = 0.0
    n_accepted   = 0
    n_subsidized = 0
    node_ptr     = k_start

    for disc in plan:
        if env._check_bankrupt() or len(env.offered) >= env.n:
            break

        # Advance to next un-offered node
        while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
            node_ptr += 1
        if node_ptr >= len(ordering):
            break

        node = ordering[node_ptr]
        node_ptr += 1

        if disc is None:
            # DP says skip this position
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)
            continue

        # Affordability check with actual est_val — SKIP if unaffordable
        est_val = env._estimate_valuation(node)
        price   = est_val * (1.0 - disc)
        if env.B - c + price < -1e-9:
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)
            continue

        _, reward, done, info = env.step(env.node_to_idx[node], disc)
        revenue += reward
        if info.get("accepted", False):
            n_accepted += 1
        if info.get("offered_price", c) < c:
            n_subsidized += 1
        if done:
            break

    return revenue, n_accepted, n_subsidized


# ── Step 1.2: Calibrated DP ────────────────────────────────────────────────────

def dp_calibrated_budget(
    graph,
    cfg: BudgetEnvConfig,
    B: float,
    c: float,
    n_trials: int = 5,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
    n_classes: int = 5,
    n_sims: int = 30,
    seed_frac: float = 0.15,
) -> dict:
    """Calibrated DP: same structure as two_phase_dp_budget, better valuation inputs.

    Replaces the naive position-average valuation with v_curve[class][position]
    from the offline calibration step, which captures both the seeding-delay
    effect and per-degree-class heterogeneity.

    Structure matches two_phase_dp_budget:
      Phase 2a — Seed top nodes for free (same seed_frac as DP-naive).
      Phase 2b — Execute DP plan on remaining positions with reduced budget.

    Key fix vs two_phase_dp_budget Phase 2: we NEVER reprice unaffordable
    positions (only SKIP), which is the correct budget-constrained behaviour.

    Args:
        graph:      NetworkX graph.
        cfg:        BudgetEnvConfig (weight_high, production_cost, etc.).
        B:          Initial budget.
        c:          Production cost.
        n_trials:   MC trials.
        tiers:      Discount fractions to consider.
        delta:      Budget DP discretisation step.
        n_classes:  Number of degree-class buckets for calibration.
        n_sims:     Number of calibration simulations.
        seed_frac:  Fraction of n to seed for free before DP phase (mirrors DP-naive).

    Returns:
        Aggregated dict (same schema as greedy_discount_budget / two_phase_dp_budget).
    """
    n = graph.number_of_nodes()
    v_curve, class_of_position, _ = calibrate_valuation_curves(
        graph, cfg, n_classes=n_classes, n_sims=n_sims
    )
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

    # Seeding budget: same formula as two_phase_dp_budget
    n_seed = max(0, min(int(n * seed_frac), int(B / c) // 3))
    # DP plan starts from position n_seed with budget reduced by seeding cost
    B_eff  = B - n_seed * c
    plan   = _plan_dp(n, v_curve, class_of_position, max(B_eff, 0.0), c, tiers, delta,
                      k_start=n_seed)

    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=cfg.weight_high)
        env.reset()
        revenue = 0.0
        n_sub   = 0

        # ── Phase 2a: Seeding pass (free offers to top-degree nodes) ────────
        for nd in ordering[:n_seed]:
            if nd in env.offered or env._check_bankrupt():
                break
            _, r_s, done_s, info_s = env.step(env.node_to_idx[nd], 1.0)
            revenue += r_s
            if info_s.get("offered_price", c) < c:
                n_sub += 1
            if done_s:
                break

        # ── Phase 2b: Execute DP plan (from position n_seed) ─────────────────
        r_dp, n_acc, n_sub_dp = _execute_plan(env, ordering, plan, c, k_start=n_seed)
        revenue += r_dp
        n_sub   += n_sub_dp

        identity_err = abs(env.B - (B - c * len(env.S) + revenue))

        results.append({
            "revenue":           revenue,
            "n_accepted":        n_acc,
            "n_offered":         len(env.offered),
            "n_subsidized":      n_sub,
            "min_budget":        min(env.budget_history) if env.budget_history else B,
            "final_budget":      env.B,
            "bankrupt":          env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
            "accounting_err":    identity_err,
        })

    return _aggregate(results)


# ── Step 1.3: Receding-Horizon DP ──────────────────────────────────────────────

def dp_receding_budget(
    graph,
    cfg: BudgetEnvConfig,
    B: float,
    c: float,
    n_trials: int = 5,
    resolve_every: int = 50,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
    n_classes: int = 5,
    n_sims: int = 30,
    **kw,
) -> dict:
    """Receding-horizon DP: re-solve Phase-1 every resolve_every steps.

    Each re-solve uses:
      - Current remaining budget (env.B).
      - Positions remaining (n - step).
      - An acceptance-rate correction: v_curve *= clip(actual/planned, 0.5, 1.5)
        so persistent optimism/pessimism self-corrects.

    The O(|B|·n·|T|) re-solve is cheap (~ms per call).

    Args:
        graph:         NetworkX graph.
        cfg:           BudgetEnvConfig.
        B:             Initial budget.
        c:             Production cost.
        n_trials:      MC trials.
        resolve_every: Re-solve interval (steps).
        tiers/delta/n_classes/n_sims: passed to calibration + DP.

    Returns:
        Aggregated dict (same schema as dp_calibrated_budget).
    """
    n = graph.number_of_nodes()
    v_curve, class_of_position, _ = calibrate_valuation_curves(
        graph, cfg, n_classes=n_classes, n_sims=n_sims
    )
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

    # Seeding bootstrap (same formula as dp_calibrated_budget)
    _seed_frac = 0.15
    n_seed = max(0, min(int(n * _seed_frac), int(B / c) // 3))
    B_eff  = max(B - n_seed * c, 0.0)

    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=cfg.weight_high)
        env.reset()

        revenue      = 0.0
        n_accepted   = 0
        n_subsidized = 0

        # ── Seeding pass (mirrors dp_calibrated_budget Phase 2a) ────────────
        for nd in ordering[:n_seed]:
            if nd in env.offered or env._check_bankrupt():
                break
            _, r_s, done_s, info_s = env.step(env.node_to_idx[nd], 1.0)
            revenue += r_s
            if info_s.get("offered_price", c) < c:
                n_subsidized += 1
            if done_s:
                break

        node_ptr = n_seed

        # Initial plan from position n_seed with remaining budget
        plan = _plan_dp(n, v_curve, class_of_position, env.B, c, tiers, delta,
                        k_start=n_seed)
        # Count how many paid-tier slots the plan expects to accept
        planned_paid = sum(1 for d in plan if d is not None and d < 0.999)

        for step in range(n_seed, n):
            if env._check_bankrupt() or len(env.offered) >= n:
                break

            # Re-plan at resolve_every intervals (except immediately after seeding)
            if step > n_seed and (step - n_seed) % resolve_every == 0:
                planned_rate = planned_paid / max(step, 1)
                actual_rate  = n_accepted / max(step, 1)
                correction   = float(
                    np.clip(actual_rate / max(planned_rate, 1e-3), 0.5, 1.5)
                )
                new_plan_rest = _plan_dp(
                    n, v_curve, class_of_position, env.B, c, tiers, delta,
                    k_start=step, rate_corr=correction,
                )
                # Splice new plan into positions [step - n_seed:]
                idx = step - n_seed
                plan[idx:] = new_plan_rest
                planned_paid = sum(1 for d in new_plan_rest if d is not None and d < 0.999)

            plan_idx = step - n_seed
            if plan_idx >= len(plan):
                break
            disc = plan[plan_idx]

            # Advance node_ptr
            while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
                node_ptr += 1
            if node_ptr >= len(ordering):
                break

            node = ordering[node_ptr]
            node_ptr += 1

            if disc is None:
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            est_val = env._estimate_valuation(node)
            price   = est_val * (1.0 - disc)
            if env.B - c + price < -1e-9:
                # SKIP — never reprice
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            _, reward, done, info = env.step(env.node_to_idx[node], disc)
            revenue += reward
            if info.get("accepted", False):
                n_accepted += 1
            if info.get("offered_price", c) < c:
                n_subsidized += 1
            if done:
                break

        identity_err = abs(env.B - (B - c * len(env.S) + revenue))
        results.append({
            "revenue":           revenue,
            "n_accepted":        n_accepted,
            "n_offered":         len(env.offered),
            "n_subsidized":      n_subsidized,
            "min_budget":        min(env.budget_history) if env.budget_history else B,
            "final_budget":      env.B,
            "bankrupt":          env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
            "accounting_err":    identity_err,
        })

    return _aggregate(results)


# ── Step 1.4: Oracle Upper Bound ───────────────────────────────────────────────

def dp_upper_bound(
    graph,
    cfg: BudgetEnvConfig,
    B: float,
    c: float,
    n_trials: int = 5,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
    **kw,
) -> dict:
    """Oracle upper bound: DP planned and executed with TRUE valuations.

    Not achievable by any real seller (requires knowing exact link weights).
    Reports 'DP-Oracle (upper bound)' in papers.
    Goal: compute the gap to oracle → quote 'method X achieves Y% of oracle'.

    Algorithm per trial:
      Pass 1 — oracle info collection:
        Reset env (fixes link weights). Walk degree-desc ordering, force all
        preceding nodes into S to maximise influence, record true_val[k].
      DP planning on v_true (single class = all nodes).
      Pass 2 — forced-acceptance execution (same seed = same link weights):
        Walk ordering, apply planned tier, compute price = true_val*(1-disc).
        If affordable: force accept (bypass est_val), update budget.

    Args:
        graph:    NetworkX graph.
        cfg:      BudgetEnvConfig.
        B:        Initial budget.
        c:        Production cost.
        n_trials: MC trials.
        tiers:    Discount fractions.
        delta:    Budget DP step.

    Returns:
        Aggregated dict (same schema as other DP baselines).
    """
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

    results = []
    for trial in range(n_trials):
        # ── Pass 1: collect oracle true valuations ──────────────────────────
        env_pass1 = _make_env(graph, B=1e9, c=c, seed=trial, weight_high=cfg.weight_high)
        env_pass1.reset()

        v_true = np.zeros(n, dtype=np.float64)
        for k, node in enumerate(ordering):
            v_true[k] = env_pass1._true_valuation(node)
            # Force all preceding into S so subsequent nodes see full influence
            env_pass1.S.add(node)
            for nb in env_pass1.graph.neighbors(node):
                env_pass1._influence_cache.pop(nb, None)
                env_pass1._true_val_cache.pop(nb, None)
                env_pass1._est_val_cache.pop(nb, None)

        # ── DP planning on v_true (single class) ────────────────────────────
        v_true_curve    = v_true[np.newaxis, :]      # shape (1, n)
        class_of_pos_oracle = np.zeros(n, dtype=np.int32)
        plan = _plan_dp(n, v_true_curve, class_of_pos_oracle, B, c, tiers, delta)

        # ── Pass 2: forced-acceptance execution (same seed) ─────────────────
        env = _make_env(graph, B, c, seed=trial, weight_high=cfg.weight_high)
        env.reset()

        revenue      = 0.0
        n_acc        = 0
        n_sub        = 0
        node_ptr     = 0

        for k_idx, disc in enumerate(plan):
            if env._check_bankrupt() or len(env.offered) >= n:
                break

            while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
                node_ptr += 1
            if node_ptr >= len(ordering):
                break

            node = ordering[node_ptr]
            node_ptr += 1

            if disc is None:
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            # Use TRUE valuation for price (oracle has perfect information)
            true_val = env._true_valuation(node)
            price    = true_val * (1.0 - disc)

            if env.B - c + price < -1e-9:
                # Even the oracle cannot afford this — skip
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            # Force acceptance: price <= true_val always (disc ∈ [0,1])
            env.S.add(node)
            env.B = env.B - c + price
            for nb in env.graph.neighbors(node):
                env._influence_cache.pop(nb, None)
                env._true_val_cache.pop(nb, None)
                env._est_val_cache.pop(nb, None)
            revenue += price
            n_acc   += 1
            if price < c:
                n_sub += 1

            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)

        # Verify accounting identity: B_final = B0 - c*n_acc + revenue
        identity_err = abs(env.B - (B - c * n_acc + revenue))

        results.append({
            "revenue":           revenue,
            "n_accepted":        n_acc,
            "n_offered":         len(env.offered),
            "n_subsidized":      n_sub,
            "min_budget":        min(env.budget_history) if env.budget_history else B,
            "final_budget":      env.B,
            "bankrupt":          env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
            "accounting_err":    identity_err,
        })

    return _aggregate(results)
