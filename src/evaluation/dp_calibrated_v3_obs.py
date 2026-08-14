"""src/evaluation/dp_calibrated_v3_obs.py — Cal-DP v3 (observation-only calibration).

Identical to dp_calibrated_v3.py EXCEPT the calibration loop never calls
_true_valuation.  Acceptance is observed from env.step() only, exactly as a
real seller would observe it.

Tier rotation: node k in sim s uses tier index (s*n+k) % n_tiers, giving
~uniform coverage of all tiers over 30 sims.  A cells are updated only for
the tier that was actually simulated.  V cells (est_val) are unchanged.

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
    cache = os.path.join(
        _CACHE_DIR,
        f"dp_calibration_v3_obs_{gh}_nc{n_classes}_ns{n_s_buckets}_{n_sims}.npz",
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

    for sim in range(n_sims):
        env = _make_env(graph, B=1e9, c=cfg.production_cost,
                        seed=seed + sim, weight_high=cfg.weight_high)
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

            # Accumulate V (uses est_val only — no true_val)
            V3_sum[cls, sb] += est_val
            V3_cnt[cls, sb] += 1
            T_cnt[k, sb]    += 1
            T_tot[k]        += 1

            # Choose ONE tier for this (sim, position) pair — round-robin
            t_obs   = (sim * n + k) % n_tiers
            t_disc  = tiers_list[t_obs]

            # OBSERVABLE acceptance: call env.step(), record info["accepted"]
            nidx = node_to_idx[node]
            _, _, done, info = env.step(nidx, t_disc)
            accepted = bool(info.get("accepted", False))

            A3_num[cls, sb, t_obs] += 1.0 if accepted else 0.0
            A3_den[cls, sb, t_obs] += 1.0
            # Other tiers: not updated this step (prior 0.5 used for empty cells)

            if done:
                break

    # V3: mean est_val (fill zeros by nearest-neighbour)
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

    # A3: fill unobserved cells with prior 0.5
    A3 = np.where(A3_den > 0, A3_num / np.maximum(A3_den, 1), 0.5)
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
