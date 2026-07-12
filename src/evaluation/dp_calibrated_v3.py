"""src/evaluation/dp_calibrated_v3.py — DP-Calibrated-v3: cascade-size-aware DP.

State space: (b_idx, k, sb) where sb = bucket(|S|, n_s_buckets).
Combines v2's live-value lookahead with cascade-aware continuation values.

Gate: v3 >= v1 at k=10,30; v3 > v2 by >= 10 at k=40 → promoted to paper.
"""

from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.baselines import _rayleigh_price
from src.evaluation.dp_calibrated import _graph_hash, _deg_class
from src.evaluation.dp_calibrated_v2 import calibrate_v2_table

_CACHE_DIR   = "results/logs"
_N_CLASSES   = 5
_N_BUCKETS   = 10    # influence buckets (reuse v2 A table)
_N_S_BUCKETS = 20    # cascade-size buckets


# ── 1. Calibration (cascade-indexed) ──────────────────────────────────────────

def calibrate_v3_table(
    graph,
    cfg: BudgetEnvConfig,
    n_classes: int = _N_CLASSES,
    n_s_buckets: int = _N_S_BUCKETS,
    n_buckets: int = _N_BUCKETS,
    n_sims: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate V3[d][sb], A3[d][sb][t], T[k][sb].

    Returns:
        V3:               (n_classes, n_s_buckets) mean est_val per class/cascade-bucket.
        A3:               (n_classes, n_s_buckets, 5) acceptance rates.
        T:                (n, n_s_buckets) P(sb at position k).
        class_boundaries: (n_classes+1,).
        sb_size:          nodes per cascade bucket (= n/n_s_buckets).
    """
    n = graph.number_of_nodes()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    gh = _graph_hash(graph)
    cache = os.path.join(_CACHE_DIR,
                         f"dp_calibration_v3_{gh}_nc{n_classes}_ns{n_s_buckets}_{n_sims}.npz")

    if os.path.exists(cache):
        dat = np.load(cache, allow_pickle=True)
        return (dat["V3"], dat["A3"], dat["T"],
                dat["class_boundaries"], float(dat["sb_size"]))

    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)
    class_boundaries = np.quantile(all_deg, np.linspace(0.0, 1.0, n_classes + 1))
    class_boundaries[-1] += 1.0
    class_of_pos = np.array([_deg_class(int(all_deg[k]), class_boundaries)
                              for k in range(n)], dtype=np.int32)

    sb_size = max(1, n / n_s_buckets)

    def _sb(s_size: int) -> int:
        return min(int(s_size / sb_size), n_s_buckets - 1)

    tiers_list = [1.0, 0.8, 0.5, 0.2, 0.0]
    n_tiers = len(tiers_list)

    V3_sum = np.zeros((n_classes, n_s_buckets))
    V3_cnt = np.zeros((n_classes, n_s_buckets))
    A3_num = np.zeros((n_classes, n_s_buckets, n_tiers))
    A3_den = np.zeros((n_classes, n_s_buckets, n_tiers))
    T_cnt  = np.zeros((n, n_s_buckets))
    T_tot  = np.zeros(n)

    for sim in range(n_sims):
        env = _make_env(graph, B=1e9, c=cfg.production_cost,
                        seed=seed + sim, weight_high=cfg.weight_high)
        env.reset()
        for k, node in enumerate(ordering):
            if node in env.offered:
                continue
            est_val  = env._estimate_valuation(node)
            true_val = env._true_valuation(node)
            infl     = env.get_current_influence(node)
            cls      = int(class_of_pos[k])
            s_size   = len(env.S)
            sb       = _sb(s_size)

            V3_sum[cls, sb] += est_val
            V3_cnt[cls, sb] += 1
            T_cnt[k, sb]    += 1
            T_tot[k]        += 1

            for t_idx, tier in enumerate(tiers_list):
                price = est_val * (1.0 - tier)
                A3_num[cls, sb, t_idx] += 1.0 if true_val >= price else 0.0
                A3_den[cls, sb, t_idx] += 1.0

            # Simulate mid-tier acceptance to grow S
            price_sim = _rayleigh_price(2.0 / 6.0) if infl >= 2.0 / 6.0 else 0.0
            if price_sim == 0.0 or true_val >= price_sim:
                env.S.add(node)
                for nb in env.graph.neighbors(node):
                    env._influence_cache.pop(nb, None)
                    env._true_val_cache.pop(nb, None)
                    env._est_val_cache.pop(nb, None)
            env.offered.add(node)
            env.t += 1

    V3 = np.where(V3_cnt > 0, V3_sum / np.maximum(V3_cnt, 1), 0.0)
    # Fill zeros by nearest neighbour along sb axis
    for d_idx in range(n_classes):
        prev = None
        for sb in range(n_s_buckets):
            if V3[d_idx, sb] > 1e-12:
                prev = V3[d_idx, sb]
            elif prev is not None:
                V3[d_idx, sb] = prev
        for sb in range(n_s_buckets - 2, -1, -1):
            if V3[d_idx, sb] < 1e-12 and V3[d_idx, sb + 1] > 1e-12:
                V3[d_idx, sb] = V3[d_idx, sb + 1]

    A3 = np.where(A3_den > 0, A3_num / np.maximum(A3_den, 1), 0.5)
    T  = T_cnt / np.maximum(T_tot[:, np.newaxis], 1)

    np.savez(cache, V3=V3, A3=A3, T=T,
             class_boundaries=class_boundaries, sb_size=np.array(sb_size))
    return V3, A3, T, class_boundaries, sb_size


# ── 2. DP with state (b_idx, sb) swept over all k ─────────────────────────────

def _plan_dp_v3(
    n_total: int,
    V3: np.ndarray,
    A3: np.ndarray,
    T: np.ndarray,
    class_of_pos: np.ndarray,
    B: float,
    c: float,
    sb_size: float,
    n_s_buckets: int = _N_S_BUCKETS,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-induction DP over state (b_idx, k, sb).

    Vectorised over b_idx for each (k, sb).

    Returns:
        dp3:   (b_steps+1, n_s_buckets) value table at k=0 for each (b,sb).
        tier3: (b_steps+1, n_total, n_s_buckets) optimal tier index (-1=skip).
    """
    b_steps    = max(1, int(B / delta) + 1)
    tiers_list = list(tiers)
    n_tiers    = len(tiers_list)

    # dp[b_idx, sb] = max expected revenue from current position onward
    dp   = np.zeros((b_steps + 1, n_s_buckets))
    tier3 = np.full((b_steps + 1, n_total, n_s_buckets), -1, dtype=np.int8)

    b_vals = np.arange(b_steps + 1) * delta  # shape (b_steps+1,)

    # Precompute sb_next: after one acceptance mid-bucket, what sb do we get?
    # Midpoint of sb bucket: sb * sb_size + sb_size/2
    sb_next = np.array([
        min(int(((sb * sb_size + sb_size / 2.0) + 1) / sb_size), n_s_buckets - 1)
        for sb in range(n_s_buckets)
    ], dtype=np.int32)

    for k_rel in range(n_total - 1, -1, -1):
        cls     = int(class_of_pos[k_rel])
        new_dp  = dp.copy()  # skip is baseline

        for sb in range(n_s_buckets):
            # Expected value at this (cls, sb)
            avg_val = float(V3[cls, sb])
            if avg_val <= 1e-9:
                avg_val = float(V3[cls].max())

            sbn = int(sb_next[sb])

            for t_idx, t_disc in enumerate(tiers_list):
                price     = avg_val * (1.0 - t_disc)
                b_accept  = b_vals - c + price           # (b_steps+1,)
                afford    = b_accept >= -1e-9

                p_acc = float(A3[cls, sb, t_idx])
                b_a_clipped = np.clip((b_accept / delta).round().astype(int),
                                      0, b_steps)

                # Continuation: accept → dp[b_a_idx, sb_next]; reject → dp[b_idx, sb]
                dp_acc = dp[b_a_clipped, sbn]    # shape (b_steps+1,)
                dp_rej = dp[:, sb]                # shape (b_steps+1,)

                value_t = (p_acc * (price + dp_acc) +
                           (1.0 - p_acc) * dp_rej)
                value_t = np.where(afford, value_t, -np.inf)

                improved = value_t > new_dp[:, sb]
                new_dp[:, sb]           = np.where(improved, value_t,   new_dp[:, sb])
                tier3[:, k_rel, sb]     = np.where(improved, t_idx,     tier3[:, k_rel, sb])

        dp = new_dp

    return dp, tier3


# ── 3. Closed-loop execution ───────────────────────────────────────────────────

def _execute_v3(
    env: BudgetRevenueEnv,
    ordering: list,
    dp3: np.ndarray,
    tier3: np.ndarray,
    V3: np.ndarray,
    A3: np.ndarray,
    class_boundaries: np.ndarray,
    sb_size: float,
    c: float,
    class_of_pos: np.ndarray,
    n_s_buckets: int,
    b_steps: int,
    delta: float,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    log_steps: int = 10,
) -> Tuple[float, int, int, list]:
    """Execute v3 policy (v2's live-value + v3's cascade-aware continuation).

    The tier is chosen as:
      t* = argmax_t { p_acc(d,sb,t) * (live_price + dp3[b_next,sb_next])
                      + (1-p_acc) * dp3[b_curr,sb] }

    Returns: (revenue, n_accepted, n_subsidized, trace[:log_steps])
    """
    tiers_list = list(tiers)
    revenue      = 0.0
    n_accepted   = 0
    n_subsidized = 0
    node_ptr     = 0
    trace        = []

    def _sb(s: int) -> int:
        return min(int(s / sb_size), n_s_buckets - 1)

    def _sb_next(sb: int) -> int:
        mid = sb * sb_size + sb_size / 2.0
        return min(int((mid + 1) / sb_size), n_s_buckets - 1)

    n_positions = tier3.shape[1]

    for k_rel in range(n_positions):
        if env._check_bankrupt() or len(env.offered) >= env.n:
            break
        while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
            node_ptr += 1
        if node_ptr >= len(ordering):
            break

        node   = ordering[node_ptr]
        node_ptr += 1

        est_val = env._estimate_valuation(node)
        b_curr  = env.B
        s_curr  = len(env.S)
        sb      = _sb(s_curr)
        sbn     = _sb_next(sb)
        cls     = int(_deg_class(int(env.graph.degree(node)), class_boundaries))

        b_idx_curr = min(int(b_curr / delta), b_steps)

        # Closed-loop tier selection
        best_val  = -1e18
        best_disc = None
        best_t    = -1

        for t_idx, t_disc in enumerate(tiers_list):
            price = est_val * (1.0 - t_disc)
            if b_curr - c + price < -1e-9:
                continue
            p_acc = float(A3[cls, sb, t_idx])
            b_next     = b_curr - c + price
            b_next_idx = min(max(int(b_next / delta), 0), b_steps)
            dp_acc = float(dp3[b_next_idx, sbn])
            dp_rej = float(dp3[b_idx_curr, sb])
            total  = p_acc * (price + dp_acc) + (1.0 - p_acc) * dp_rej
            if total > best_val:
                best_val  = total
                best_disc = t_disc
                best_t    = t_idx

        if best_disc is None:
            # Nothing affordable — skip
            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)
            continue

        if k_rel < log_steps:
            trace.append({
                "k":        k_rel,
                "node":     int(node),
                "cls":      cls,
                "S_size":   s_curr,
                "sb":       sb,
                "est_val":  round(est_val, 4),
                "tier":     tiers_list[best_t],
                "price":    round(est_val * (1 - tiers_list[best_t]), 4),
            })

        _, reward, done, info = env.step(env.node_to_idx[node], best_disc)
        revenue += reward
        if info.get("accepted", False):
            n_accepted += 1
        if info.get("offered_price", c) < c:
            n_subsidized += 1
        if done:
            break

    return revenue, n_accepted, n_subsidized, trace


# ── 4. Public API ──────────────────────────────────────────────────────────────

def dp_calibrated_v3_budget(
    graph,
    cfg: BudgetEnvConfig,
    B: float,
    c: float,
    n_trials: int = 5,
    tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05,
    n_classes: int = _N_CLASSES,
    n_s_buckets: int = _N_S_BUCKETS,
    n_sims: int = 30,
    seed_frac: float = 0.0,  # no-op: planner uses tier-1.0 action for seeding
    return_trace: bool = False,
) -> dict:
    """DP-Calibrated-v3: cascade-size-aware DP state + closed-loop execution.

    No separate free-seed phase — the DP planner handles seeding organically
    by choosing tier=1.0 (full discount = free) when that maximises E[revenue].
    This removes the confound present in v1 and keeps the comparison clean.

    Args:
        graph:        NetworkX graph.
        cfg:          BudgetEnvConfig.
        B:            Budget per episode.
        c:            Production cost.
        n_trials:     Evaluation trials.
        tiers:        Discount tiers (1.0 = free, used by planner as needed).
        delta:        Budget discretisation step.
        n_classes:    Degree-class buckets.
        n_s_buckets:  Cascade-size buckets (DP state dimension).
        n_sims:       Calibration sims.
        seed_frac:    Ignored (kept for API compatibility).
        return_trace: If True, include 10-step execution trace.

    Returns:
        Aggregated results dict (same schema as v1/v2 + optional 'trace').
    """
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    # Calibrate
    V3, A3, T, class_boundaries, sb_size = calibrate_v3_table(
        graph, cfg,
        n_classes=n_classes, n_s_buckets=n_s_buckets,
        n_sims=n_sims, seed=0,
    )
    class_of_pos = np.array([_deg_class(int(all_deg[k]), class_boundaries)
                              for k in range(n)], dtype=np.int32)

    # DP table over ALL n positions (planner uses tier=1.0 for seeding when optimal)
    b_steps = max(1, int(B / delta) + 1)
    dp3, tier3 = _plan_dp_v3(
        n_total=n,
        V3=V3, A3=A3, T=T,
        class_of_pos=class_of_pos,
        B=B, c=c,
        sb_size=sb_size, n_s_buckets=n_s_buckets,
        tiers=tiers, delta=delta,
    )

    results  = []
    traces   = []

    for trial in range(n_trials):
        env = _make_env(graph, B=B, c=c, seed=trial, weight_high=cfg.weight_high)
        env.reset()

        # Cascade-aware closed-loop over all positions
        rev, n_acc, n_sub, trace = _execute_v3(
            env=env,
            ordering=ordering,
            dp3=dp3, tier3=tier3,
            V3=V3, A3=A3,
            class_boundaries=class_boundaries,
            sb_size=sb_size, c=c,
            class_of_pos=class_of_pos,
            n_s_buckets=n_s_buckets,
            b_steps=b_steps, delta=delta,
            tiers=tiers,
            log_steps=10,
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
        if trial == 0:
            traces = trace

    out = _aggregate(results)
    if return_trace:
        out["trace"] = traces
    return out
