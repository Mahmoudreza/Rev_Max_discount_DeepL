"""src/evaluation/dp_calibrated_v2.py — DP-Calibrated-v2: influence-indexed calibration.

Extends v1 by keying the calibration on (degree_class, influence_bucket) instead
of (degree_class, position), and using live estimated valuation at execution time
(closed-loop tier lookup).

Validation gate: v2 >= v1 on FF n=200 (checked in _validate_v2).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.baselines import _rayleigh_price
from src.evaluation.dp_calibrated import (
    _graph_hash,
    _deg_class,
    calibrate_valuation_curves,   # re-used for v_curve anchor
    _plan_dp,
    _execute_plan,
    dp_calibrated_budget as dp_calibrated_v1,  # for comparison in gate
)

_CACHE_DIR = "results/logs"
_N_BUCKETS = 10     # influence buckets
_N_CLASSES = 5      # degree classes


# ── 1. Calibration ─────────────────────────────────────────────────────────────

def calibrate_v2_table(
    graph,
    cfg: BudgetEnvConfig,
    n_classes: int = _N_CLASSES,
    n_buckets: int = _N_BUCKETS,
    n_sims: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate V[d][ib], A[d][ib][t_idx], P[k][ib].

    Args:
        graph:     NetworkX graph.
        cfg:       BudgetEnvConfig.
        n_classes: Number of degree buckets.
        n_buckets: Number of influence buckets.
        n_sims:    Calibration simulations.
        seed:      Base RNG seed.

    Returns:
        V:                  (n_classes, n_buckets) mean est_val per cell.
        A:                  (n_classes, n_buckets, 5) acceptance rate per cell/tier.
        P:                  (n, n_buckets) prob(influence_bucket=ib) at position k.
        class_boundaries:   (n_classes+1,) degree quantile boundaries.
        infl_boundaries:    (n_buckets+1,) influence boundaries.
    """
    n = graph.number_of_nodes()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    gh = _graph_hash(graph)
    cache = os.path.join(_CACHE_DIR, f"dp_calibration_v2_{gh}_nc{n_classes}_nb{n_buckets}_ns{n_sims}.npz")

    if os.path.exists(cache):
        dat = np.load(cache, allow_pickle=True)
        return (dat["V"], dat["A"], dat["P"],
                dat["class_boundaries"], dat["infl_boundaries"])

    # Degree ordering + class boundaries
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg = np.array([graph.degree(v) for v in ordering], dtype=float)
    class_boundaries = np.quantile(all_deg, np.linspace(0.0, 1.0, n_classes + 1))
    class_boundaries[-1] += 1.0

    class_of_pos = np.array([_deg_class(int(all_deg[k]), class_boundaries)
                              for k in range(n)], dtype=np.int32)

    tiers_list = [1.0, 0.8, 0.5, 0.2, 0.0]
    n_tiers = len(tiers_list)

    # Collect (cls, infl, est_val, true_val) per step across sims
    records_cls  = []
    records_infl = []
    records_ev   = []
    records_tv   = []

    for sim in range(n_sims):
        env = _make_env(graph, B=1e9, c=cfg.production_cost,
                        seed=seed + sim, weight_high=cfg.weight_high)
        env.reset()
        for k, node in enumerate(ordering):
            if node in env.offered:
                continue
            est_val = env._estimate_valuation(node)
            true_val = env._true_valuation(node)
            infl = env.get_current_influence(node)
            cls = int(class_of_pos[k])
            records_cls.append(cls)
            records_infl.append(float(infl))
            records_ev.append(float(est_val))
            records_tv.append(float(true_val))

            # Simulate a mid-tier offer to grow S realistically
            price = _rayleigh_price(2.0 / 6.0) if infl >= 2.0 / 6.0 else 0.0
            if price == 0.0 or true_val >= price:
                env.S.add(node)
                for nb in env.graph.neighbors(node):
                    env._influence_cache.pop(nb, None)
                    env._true_val_cache.pop(nb, None)
                    env._est_val_cache.pop(nb, None)
            env.offered.add(node)
            env.t += 1

    records_cls  = np.array(records_cls,  dtype=np.int32)
    records_infl = np.array(records_infl, dtype=np.float64)
    records_ev   = np.array(records_ev,   dtype=np.float64)
    records_tv   = np.array(records_tv,   dtype=np.float64)

    # Influence bucket boundaries from empirical quantiles (clip outliers)
    infl_q = np.quantile(records_infl, np.linspace(0.0, 1.0, n_buckets + 1))
    infl_q[-1] += 1e-9
    infl_boundaries = infl_q

    def _infl_bucket(x: float) -> int:
        for i in range(n_buckets - 1, 0, -1):
            if x >= infl_boundaries[i]:
                return i
        return 0

    ib_arr = np.array([_infl_bucket(x) for x in records_infl], dtype=np.int32)

    # V[d][ib] = mean est_val
    V_sum   = np.zeros((n_classes, n_buckets))
    V_cnt   = np.zeros((n_classes, n_buckets))
    for i in range(len(records_cls)):
        V_sum[records_cls[i], ib_arr[i]] += records_ev[i]
        V_cnt[records_cls[i], ib_arr[i]] += 1
    V = np.where(V_cnt > 0, V_sum / np.maximum(V_cnt, 1), 0.0)

    # Fill zero cells: interpolate along ib axis then cls axis
    for d_idx in range(n_classes):
        prev = None
        for ib in range(n_buckets):
            if V[d_idx, ib] > 1e-12:
                prev = V[d_idx, ib]
            elif prev is not None:
                V[d_idx, ib] = prev
        # forward fill from last valid
        for ib in range(n_buckets - 2, -1, -1):
            if V[d_idx, ib] < 1e-12 and V[d_idx, ib + 1] > 1e-12:
                V[d_idx, ib] = V[d_idx, ib + 1]

    # A[d][ib][t] = P(accept) = P(true_val >= price)
    A_num = np.zeros((n_classes, n_buckets, n_tiers))
    A_den = np.zeros((n_classes, n_buckets, n_tiers))
    for i in range(len(records_cls)):
        d, ib = records_cls[i], ib_arr[i]
        ev_i, tv_i = records_ev[i], records_tv[i]
        for t_idx, tier in enumerate(tiers_list):
            price = ev_i * (1.0 - tier)
            A_num[d, ib, t_idx] += 1.0 if tv_i >= price else 0.0
            A_den[d, ib, t_idx] += 1.0
    A = np.where(A_den > 0, A_num / np.maximum(A_den, 1), 0.5)

    # P[k][ib] = fraction of sims where node at position k had influence bucket ib
    # Re-run sims tracking per-position influence
    P_cnt = np.zeros((n, n_buckets))
    P_tot = np.zeros(n)
    for sim in range(n_sims):
        env = _make_env(graph, B=1e9, c=cfg.production_cost,
                        seed=seed + sim, weight_high=cfg.weight_high)
        env.reset()
        for k, node in enumerate(ordering):
            if node in env.offered:
                continue
            infl = env.get_current_influence(node)
            ib = _infl_bucket(float(infl))
            P_cnt[k, ib] += 1
            P_tot[k] += 1
            # Same mid-tier simulate as above
            est_val = env._estimate_valuation(node)
            true_val = env._true_valuation(node)
            price = _rayleigh_price(2.0 / 6.0) if infl >= 2.0 / 6.0 else 0.0
            if price == 0.0 or true_val >= price:
                env.S.add(node)
                for nb in env.graph.neighbors(node):
                    env._influence_cache.pop(nb, None)
                    env._true_val_cache.pop(nb, None)
                    env._est_val_cache.pop(nb, None)
            env.offered.add(node)
            env.t += 1
    P = P_cnt / np.maximum(P_tot[:, np.newaxis], 1)

    np.savez(cache, V=V, A=A, P=P,
             class_boundaries=class_boundaries,
             infl_boundaries=infl_boundaries)
    return V, A, P, class_boundaries, infl_boundaries


# ── 2. Planning ────────────────────────────────────────────────────────────────

def _plan_dp_v2(
    n_total: int,
    V: np.ndarray,
    A: np.ndarray,
    P: np.ndarray,
    class_of_pos: np.ndarray,
    B: float,
    c: float,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
) -> List[Optional[float]]:
    """DP using E[val at k] = sum_ib P[k][ib] * V[cls][ib].

    Same backward-induction structure as _plan_dp (v1) but valuation driven
    by influence-bucket prior instead of position-average curve.
    """
    n_remain = n_total
    b_steps = max(1, int(B / delta) + 1)
    tiers_list = list(tiers)

    dp          = [[0.0] * (n_remain + 1) for _ in range(b_steps + 1)]
    tier_choice = [[-1]  * (n_remain + 1) for _ in range(b_steps + 1)]

    for k_rel in range(n_remain - 1, -1, -1):
        cls = int(class_of_pos[k_rel])
        # Expected valuation at position k_rel
        avg_val = float(np.dot(P[k_rel], V[cls]))
        if avg_val <= 1e-9:
            avg_val = float(V[cls].max())

        for b_idx in range(b_steps + 1):
            b_curr = b_idx * delta
            best_rev = dp[b_idx][k_rel + 1]
            best_t   = -1

            for t_idx, t_disc in enumerate(tiers_list):
                price = avg_val * (1.0 - t_disc)
                if b_curr - c + price < -1e-9:
                    continue
                new_b_raw  = b_curr - c + price
                new_b_idx  = min(int(new_b_raw / delta), b_steps)
                # Expected-accept: use A[cls][ib_mode][t] as p_acc estimate
                ib_mode = int(np.argmax(P[k_rel]))
                p_acc = float(A[cls, ib_mode, t_idx])
                ev = p_acc * price + dp[new_b_idx][k_rel + 1]
                if ev > best_rev:
                    best_rev = ev
                    best_t   = t_idx

            dp[b_idx][k_rel]          = best_rev
            tier_choice[b_idx][k_rel] = best_t

    # Extract plan
    plan  = [None] * n_remain
    b_idx = min(int(B / delta), b_steps)
    for k_rel in range(n_remain):
        cls = int(class_of_pos[k_rel])
        t_i = tier_choice[min(b_idx, b_steps)][k_rel]
        if t_i >= 0:
            disc         = tiers_list[t_i]
            plan[k_rel]  = disc
            avg_val      = float(np.dot(P[k_rel], V[cls]))
            if avg_val <= 1e-9:
                avg_val = float(V[cls].max())
            price        = avg_val * (1.0 - disc)
            new_b_raw    = b_idx * delta - c + price
            b_idx        = min(max(int(new_b_raw / delta), 0), b_steps)
    return plan


# ── 3. Closed-loop execution ───────────────────────────────────────────────────

def _execute_v2(
    env: BudgetRevenueEnv,
    ordering: list,
    plan: List[Optional[float]],
    V: np.ndarray,
    A: np.ndarray,
    class_boundaries: np.ndarray,
    infl_boundaries: np.ndarray,
    c: float,
    class_of_pos: np.ndarray,
    dp_table: list,
    b_steps: int,
    delta: float,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
) -> Tuple[float, int, int]:
    """Closed-loop v2 execution.

    At each step: use LIVE est_val and influence bucket to pick the tier that
    maximizes p_acc*(price + continuation).  Falls back to plan tier if dp_table
    not provided.

    Returns: (revenue, n_accepted, n_subsidized)
    """
    tiers_list = list(tiers)
    revenue      = 0.0
    n_accepted   = 0
    n_subsidized = 0
    node_ptr     = 0

    def _infl_bucket(x: float) -> int:
        for i in range(len(infl_boundaries) - 2, 0, -1):
            if x >= infl_boundaries[i]:
                return i
        return 0

    for k_rel, disc_plan in enumerate(plan):
        if env._check_bankrupt() or len(env.offered) >= env.n:
            break
        while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
            node_ptr += 1
        if node_ptr >= len(ordering):
            break

        node = ordering[node_ptr]
        node_ptr += 1

        if disc_plan is None:
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)
            continue

        # Live valuation and influence bucket
        est_val = env._estimate_valuation(node)
        infl    = env.get_current_influence(node)
        b_curr  = env.B
        ib      = _infl_bucket(float(infl))
        cls     = int(_deg_class(int(env.graph.degree(node)), class_boundaries))

        # Closed-loop: choose tier maximising p_acc*(live_price + continuation)
        best_val  = -1e18
        best_disc = disc_plan  # fallback

        b_idx_curr = min(int(b_curr / delta), b_steps)
        remain     = len(plan) - k_rel - 1  # remaining positions after this

        for t_idx, t_disc in enumerate(tiers_list):
            price = est_val * (1.0 - t_disc)
            if b_curr - c + price < -1e-9:
                continue
            p_acc = float(A[cls, ib, t_idx]) if ib < A.shape[1] else 0.5
            b_next = b_curr - c + price
            b_next_idx = min(max(int(b_next / delta), 0), b_steps)
            # Continuation from dp_table[b_next_idx] at next position
            # dp_table is indexed [b_idx][k_rel_from_end] — use remaining steps
            k_cont = min(remain, len(dp_table[0]) - 1)
            continuation = dp_table[b_next_idx][max(len(dp_table[0]) - 1 - k_cont, 0)]
            total = p_acc * (price + continuation) + (1 - p_acc) * dp_table[b_idx_curr][max(len(dp_table[0]) - 1 - k_cont, 0)]
            if total > best_val:
                best_val  = total
                best_disc = t_disc

        price = est_val * (1.0 - best_disc)
        if b_curr - c + price < -1e-9:
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)
            continue

        _, reward, done, info = env.step(env.node_to_idx[node], best_disc)
        revenue += reward
        if info.get("accepted", False):
            n_accepted += 1
        if info.get("offered_price", c) < c:
            n_subsidized += 1
        if done:
            break

    return revenue, n_accepted, n_subsidized


# ── 4. Public API ──────────────────────────────────────────────────────────────

def dp_calibrated_v2_budget(
    graph,
    cfg: BudgetEnvConfig,
    B: float,
    c: float,
    n_trials: int = 5,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
    n_classes: int = _N_CLASSES,
    n_buckets: int = _N_BUCKETS,
    n_sims: int = 30,
    seed_frac: float = 0.0,  # no-op: planner uses tier-1.0 action for seeding
) -> dict:
    """DP-Calibrated-v2: influence-indexed calibration + closed-loop execution.

    No separate free-seed phase — the DP planner handles seeding organically
    by choosing tier=1.0 (full discount = free) when that maximises E[revenue].
    This removes the confound present in v1 and keeps the comparison clean.

    Args:
        graph:      NetworkX graph.
        cfg:        BudgetEnvConfig.
        B:          Budget per episode.
        c:          Production cost.
        n_trials:   Number of evaluation trials.
        tiers:      Discount tiers (1.0 = free, used by planner as needed).
        delta:      Budget discretisation step.
        n_classes:  Degree-class buckets.
        n_buckets:  Influence buckets.
        n_sims:     Calibration simulations.
        seed_frac:  Ignored (kept for API compatibility).

    Returns:
        Aggregated results dict (same schema as dp_calibrated_budget).
    """
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    # Calibrate
    V, A, P, class_boundaries, infl_boundaries = calibrate_v2_table(
        graph, cfg, n_classes=n_classes, n_buckets=n_buckets,
        n_sims=n_sims, seed=0,
    )
    class_of_pos = np.array([_deg_class(int(all_deg[k]), class_boundaries)
                              for k in range(n)], dtype=np.int32)

    # DP plan over ALL n positions (planner uses tier=1.0 for seeding when optimal)
    b_steps = max(1, int(B / delta) + 1)
    plan = _plan_dp_v2(
        n_total=n,
        V=V, A=A, P=P,
        class_of_pos=class_of_pos,
        B=B, c=c, tiers=tiers, delta=delta,
    )
    dp_table_simple = [[0.0] * (n + 1) for _ in range(b_steps + 1)]

    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B=B, c=c, seed=trial, weight_high=cfg.weight_high)
        env.reset()

        rev, n_acc, n_sub = _execute_v2(
            env=env,
            ordering=ordering,
            plan=plan,
            V=V, A=A,
            class_boundaries=class_boundaries,
            infl_boundaries=infl_boundaries,
            c=c,
            class_of_pos=class_of_pos,
            dp_table=dp_table_simple,
            b_steps=b_steps,
            delta=delta,
            tiers=tiers,
        )

        results.append({
            "revenue":       rev,
            "n_accepted":    n_acc,
            "n_offered":     int(env.t),
            "n_subsidized":  n_sub,
            "min_budget":    float(min(env.budget_history)) if env.budget_history else 0.0,
            "final_budget":  float(env.B),
            "bankrupt":      bool(env._check_bankrupt()),
            "accounting_err": abs(env.B - (B - c * env.t + rev)),
        })

    return _aggregate(results)


# ── 5. Validation gate ─────────────────────────────────────────────────────────

def validate_v2(graph, cfg: BudgetEnvConfig, B: float, c: float,
                n_sims: int = 10, k_list=(10, 30)) -> bool:
    """Gate: v2 >= v1 at both budget levels in k_list. Returns True if PASS."""
    import sys
    all_pass = True
    for k in k_list:
        B_k = k * c  # budget = k * c
        r_v1 = dp_calibrated_v1(graph, cfg, B=B_k, c=c,
                                 n_trials=n_sims, n_sims=10)["revenue"]["mean"]
        r_v2 = dp_calibrated_v2_budget(graph, cfg, B=B_k, c=c,
                                       n_trials=n_sims, n_sims=10)["revenue"]["mean"]
        ok = r_v2 >= r_v1 - 1e-6
        print(f"  Gate k={k}: v1={r_v1:.3f}  v2={r_v2:.3f}  {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_pass = False
    return all_pass
