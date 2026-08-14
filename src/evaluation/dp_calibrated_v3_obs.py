"""src/evaluation/dp_calibrated_v3_obs.py — Cal-DP v3 (observation-only calibration).

Identical to dp_calibrated_v3.py EXCEPT the calibration loop never calls
_true_valuation.  Acceptance is observed from env.step() only.

Observation matching: 5 passes × n_sims campaigns, one dedicated pass per
discount tier (tiers=[1.0,0.8,0.5,0.2,0.0]).  Total offers = 5×n_sims×n,
identical to the oracle version's 5 labels per buyer per campaign.

Empty cells filled by monotone interpolation along the tier axis (A is
non-increasing in price), then nearest-neighbour fill along cls/sb axes.
No uninformative 0.5 prior.

Planning and execution are imported directly from dp_calibrated_v3.
"""

from __future__ import annotations

import os
from typing import Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.dp_calibrated import _graph_hash, _deg_class
from src.evaluation.dp_calibrated_v3 import (
    _plan_dp_v3,
    _execute_v3,
    _N_CLASSES,
    _N_BUCKETS,
    _N_S_BUCKETS,
)

_CACHE_DIR = "results/logs"


# ── Observation-only calibration ──────────────────────────────────────────────

def calibrate_v3_obs_table(
    graph,
    cfg: BudgetEnvConfig,
    n_classes: int = _N_CLASSES,
    n_s_buckets: int = _N_S_BUCKETS,
    n_buckets: int = _N_BUCKETS,
    n_sims: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calibrate V3[d][sb], A3[d][sb][t] using env.step() observations only.

    No call to _true_valuation is made.  Acceptance is observed from
    env.step(node_idx, tier_disc) return value (info["accepted"]).

    Returns: (V3, A3, T, class_boundaries, sb_size) — same schema as v3.
    """
    n = graph.number_of_nodes()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    gh = _graph_hash(graph)
    # Cache key "_5pass": 5 dedicated tier passes × n_sims (150k total offers)
    cache = os.path.join(
        _CACHE_DIR,
        f"dp_calibration_v3_obs5_{gh}_nc{n_classes}_ns{n_s_buckets}_{n_sims}.npz",
    )
    if os.path.exists(cache):
        dat = np.load(cache, allow_pickle=True)
        return (dat["V3"], dat["A3"], dat["T"],
                dat["class_boundaries"], float(dat["sb_size"]))

    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)
    class_boundaries = np.quantile(all_deg, np.linspace(0.0, 1.0, n_classes + 1))
    class_boundaries[-1] += 1.0
    class_of_pos = np.array(
        [_deg_class(int(all_deg[k]), class_boundaries) for k in range(n)],
        dtype=np.int32,
    )

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

    # 5 passes × n_sims: pass t_pass is dedicated to tier tiers_list[t_pass].
    # Total offers = 5 × n_sims × n  (matches oracle's 5 labels per buyer).
    for t_pass in range(n_tiers):
        t_disc = tiers_list[t_pass]
        for sim in range(n_sims):
            env = _make_env(graph, B=1e9, c=cfg.production_cost,
                            seed=seed + t_pass * n_sims + sim,
                            weight_high=cfg.weight_high)
            env.reset()
            node_to_idx = {v: i for i, v in enumerate(env.nodes)}

            for k, node in enumerate(ordering):
                if node in env.offered:
                    continue
                if env._check_bankrupt():
                    break

                est_val = env._estimate_valuation(node)
                cls     = int(class_of_pos[k])
                s_size  = len(env.S)
                sb      = _sb(s_size)

                # V and T: collect only on pass 0 (mid tier) to avoid bias
                if t_pass == 2:   # tier=0.5 pass — balanced acceptance
                    V3_sum[cls, sb] += est_val
                    V3_cnt[cls, sb] += 1
                    T_cnt[k, sb]    += 1
                    T_tot[k]        += 1

                # OBSERVABLE acceptance via env.step()
                nidx = node_to_idx[node]
                _, _, done, info = env.step(nidx, t_disc)
                accepted = bool(info.get("accepted", False))

                A3_num[cls, sb, t_pass] += 1.0 if accepted else 0.0
                A3_den[cls, sb, t_pass] += 1.0

                if done:
                    break

    # V3: mean est_val from mid-tier pass (tier=0.5), fill zeros
    V3 = np.where(V3_cnt > 0, V3_sum / np.maximum(V3_cnt, 1), 0.0)
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

    # A3 raw estimates
    A3 = np.where(A3_den > 0, A3_num / np.maximum(A3_den, 1), np.nan)

    # Monotone interpolation:
    # Step 1: along tier axis, enforce non-increasing A3[d,sb,*] (lower price→higher acc)
    #   tiers_list = [1.0, 0.8, 0.5, 0.2, 0.0] → t_idx=0 cheapest (free) → A should decrease
    # Step 2: nearest-neighbour fill remaining NaN along sb then d axes
    for d_idx in range(n_classes):
        for sb in range(n_s_buckets):
            row = A3[d_idx, sb, :]  # shape (n_tiers,)
            # Forward fill (t=0 to t=4): A3[t] <= A3[t-1] (monotone non-increasing)
            observed = ~np.isnan(row)
            if observed.any():
                # Interpolate NaNs between observed points
                xs = np.where(observed)[0]
                ys = row[observed]
                if len(xs) > 1:
                    interp = np.interp(np.arange(n_tiers), xs, ys)
                else:
                    interp = np.full(n_tiers, ys[0])
                # Enforce monotone non-increasing
                for t in range(1, n_tiers):
                    if interp[t] > interp[t - 1]:
                        interp[t] = interp[t - 1]
                A3[d_idx, sb, :] = interp
    # Fill any fully-NaN (cls, sb) cells from nearest sb neighbour
    for d_idx in range(n_classes):
        prev_row = None
        for sb in range(n_s_buckets):
            if not np.isnan(A3[d_idx, sb, 0]):
                prev_row = A3[d_idx, sb, :]
            elif prev_row is not None:
                A3[d_idx, sb, :] = prev_row
        for sb in range(n_s_buckets - 2, -1, -1):
            if np.isnan(A3[d_idx, sb, 0]) and not np.isnan(A3[d_idx, sb + 1, 0]):
                A3[d_idx, sb, :] = A3[d_idx, sb + 1, :]
    # Final fallback: replace any remaining NaN with 0.5
    A3 = np.where(np.isnan(A3), 0.5, A3)

    T  = T_cnt / np.maximum(T_tot[:, np.newaxis], 1)

    np.savez(cache, V3=V3, A3=A3, T=T,
             class_boundaries=class_boundaries, sb_size=np.array(sb_size))
    return V3, A3, T, class_boundaries, sb_size


# ── Public API (same signature as dp_calibrated_v3_budget) ────────────────────

def dp_calibrated_v3_obs_budget(
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
    return_trace: bool = False,
) -> dict:
    """Cal-DP v3 with observation-only calibration (no _true_valuation)."""
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    V3, A3, T, class_boundaries, sb_size = calibrate_v3_obs_table(
        graph, cfg,
        n_classes=n_classes, n_s_buckets=n_s_buckets,
        n_sims=n_sims, seed=0,
    )
    class_of_pos = np.array(
        [_deg_class(int(all_deg[k]), class_boundaries) for k in range(n)],
        dtype=np.int32,
    )

    b_steps = max(1, int(B / delta) + 1)
    dp3, tier3 = _plan_dp_v3(
        n_total=n, V3=V3, A3=A3, T=T,
        class_of_pos=class_of_pos, B=B, c=c,
        sb_size=sb_size, n_s_buckets=n_s_buckets,
        tiers=tiers, delta=delta,
    )

    results = []
    traces  = []
    for trial in range(n_trials):
        env = _make_env(graph, B=B, c=c, seed=trial, weight_high=cfg.weight_high)
        env.reset()
        rev, n_acc, n_sub, trace = _execute_v3(
            env=env, ordering=ordering,
            dp3=dp3, tier3=tier3, V3=V3, A3=A3,
            class_boundaries=class_boundaries,
            sb_size=sb_size, c=c,
            class_of_pos=class_of_pos,
            n_s_buckets=n_s_buckets,
            b_steps=b_steps, delta=delta,
            tiers=tiers, log_steps=10,
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
