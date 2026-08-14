"""src/evaluation/dp_calibrated_v2_obs.py — Cal-DP v2 (observation-only calibration).

Identical to dp_calibrated_v2.py EXCEPT _true_valuation is never called.
Acceptance rates are estimated purely from env.step() observable outcomes.
Planning and execution imported from dp_calibrated_v2.
"""
from __future__ import annotations
import os
from typing import Tuple
import numpy as np

from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.dp_calibrated import _graph_hash, _deg_class
from src.evaluation.dp_calibrated_v2 import (
    _plan_dp_v2, _execute_v2, dp_calibrated_v2_budget,
    _N_BUCKETS, _N_CLASSES,
)

_CACHE_DIR = "results/logs"


def calibrate_v2_obs_table(
    graph,
    cfg: BudgetEnvConfig,
    n_classes: int = _N_CLASSES,
    n_buckets: int = _N_BUCKETS,
    n_sims: int = 30,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """V[d][ib], A[d][ib][t], P[k][ib] — observation-only (no _true_valuation)."""
    n = graph.number_of_nodes()
    os.makedirs(_CACHE_DIR, exist_ok=True)
    gh = _graph_hash(graph)
    cache = os.path.join(
        _CACHE_DIR,
        f"dp_calibration_v2_obs_{gh}_nc{n_classes}_nb{n_buckets}_{n_sims}.npz",
    )
    if os.path.exists(cache):
        dat = np.load(cache, allow_pickle=True)
        return (dat["V"], dat["A"], dat["P"],
                dat["class_boundaries"], dat["infl_boundaries"])

    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)
    class_boundaries = np.quantile(all_deg, np.linspace(0.0, 1.0, n_classes + 1))
    class_boundaries[-1] += 1.0
    class_of_pos = np.array(
        [_deg_class(int(all_deg[k]), class_boundaries) for k in range(n)],
        dtype=np.int32,
    )
    tiers_list = [1.0, 0.8, 0.5, 0.2, 0.0]
    n_tiers = len(tiers_list)

    # First pass: collect (cls, infl, est_val) and sample acceptance via env.step()
    records_cls  = []
    records_infl = []
    records_ev   = []
    records_acc  = []   # (t_obs, accepted) pairs
    records_tobs = []

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
            infl    = env.get_current_influence(node)
            cls     = int(class_of_pos[k])
            records_cls.append(cls)
            records_infl.append(float(infl))
            records_ev.append(float(est_val))

            t_obs  = (sim * n + k) % n_tiers
            nidx   = node_to_idx[node]
            _, _, done, info = env.step(nidx, tiers_list[t_obs])
            records_acc.append(bool(info.get("accepted", False)))
            records_tobs.append(t_obs)
            if done:
                break

    records_cls   = np.array(records_cls,   dtype=np.int32)
    records_infl  = np.array(records_infl,  dtype=np.float64)
    records_ev    = np.array(records_ev,    dtype=np.float64)
    records_acc   = np.array(records_acc,   dtype=bool)
    records_tobs  = np.array(records_tobs,  dtype=np.int32)

    infl_q = np.quantile(records_infl, np.linspace(0.0, 1.0, n_buckets + 1))
    infl_q[-1] += 1e-9
    infl_boundaries = infl_q

    def _infl_bucket(x: float) -> int:
        for i in range(n_buckets - 1, 0, -1):
            if x >= infl_boundaries[i]:
                return i
        return 0

    ib_arr = np.array([_infl_bucket(x) for x in records_infl], dtype=np.int32)

    V_sum = np.zeros((n_classes, n_buckets))
    V_cnt = np.zeros((n_classes, n_buckets))
    for i in range(len(records_cls)):
        V_sum[records_cls[i], ib_arr[i]] += records_ev[i]
        V_cnt[records_cls[i], ib_arr[i]] += 1
    V = np.where(V_cnt > 0, V_sum / np.maximum(V_cnt, 1), 0.0)
    for d_idx in range(n_classes):
        prev = None
        for ib in range(n_buckets):
            if V[d_idx, ib] > 1e-12:
                prev = V[d_idx, ib]
            elif prev is not None:
                V[d_idx, ib] = prev
        for ib in range(n_buckets - 2, -1, -1):
            if V[d_idx, ib] < 1e-12 and V[d_idx, ib + 1] > 1e-12:
                V[d_idx, ib] = V[d_idx, ib + 1]

    # A[d][ib][t] from observable outcomes only
    A_num = np.zeros((n_classes, n_buckets, n_tiers))
    A_den = np.zeros((n_classes, n_buckets, n_tiers))
    for i in range(len(records_cls)):
        d, ib, t = records_cls[i], ib_arr[i], records_tobs[i]
        A_num[d, ib, t] += 1.0 if records_acc[i] else 0.0
        A_den[d, ib, t] += 1.0
    A = np.where(A_den > 0, A_num / np.maximum(A_den, 1), 0.5)

    # P[k][ib]: second pass — also uses env.step() for consistent S growth
    P_cnt = np.zeros((n, n_buckets))
    P_tot = np.zeros(n)
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
            infl = env.get_current_influence(node)
            ib   = _infl_bucket(float(infl))
            P_cnt[k, ib] += 1
            P_tot[k]     += 1
            t_obs = (sim * n + k) % n_tiers
            nidx  = node_to_idx[node]
            _, _, done, _ = env.step(nidx, tiers_list[t_obs])
            if done:
                break
    P = P_cnt / np.maximum(P_tot[:, np.newaxis], 1)

    np.savez(cache, V=V, A=A, P=P,
             class_boundaries=class_boundaries, infl_boundaries=infl_boundaries)
    return V, A, P, class_boundaries, infl_boundaries


def dp_calibrated_v2_obs_budget(
    graph, cfg: BudgetEnvConfig, B: float, c: float,
    n_trials: int = 5, tiers: tuple = (1.0, 0.8, 0.5, 0.2, 0.0),
    delta: float = 0.05, n_classes: int = _N_CLASSES,
    n_buckets: int = _N_BUCKETS, n_sims: int = 30,
) -> dict:
    """Cal-DP v2 with observation-only calibration (no _true_valuation)."""
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    V, A, P, class_boundaries, infl_boundaries = calibrate_v2_obs_table(
        graph, cfg, n_classes=n_classes, n_buckets=n_buckets, n_sims=n_sims, seed=0,
    )
    class_of_pos = np.array(
        [_deg_class(int(all_deg[k]), class_boundaries) for k in range(n)],
        dtype=np.int32,
    )
    b_steps = max(1, int(B / delta) + 1)
    plan = _plan_dp_v2(
        n_total=n, V=V, A=A, P=P, class_of_pos=class_of_pos,
        B=B, c=c, tiers=tiers, delta=delta,
    )
    dp_table_simple = [[0.0] * (n + 1) for _ in range(b_steps + 1)]
    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B=B, c=c, seed=trial, weight_high=cfg.weight_high)
        env.reset()
        rev, n_acc, n_sub = _execute_v2(
            env=env, ordering=ordering, plan=plan,
            V=V, A=A, class_boundaries=class_boundaries,
            infl_boundaries=infl_boundaries, c=c,
            class_of_pos=class_of_pos, dp_table=dp_table_simple,
            b_steps=b_steps, delta=delta, tiers=tiers,
        )
        results.append({
            "revenue": rev, "n_accepted": n_acc,
            "n_offered": int(env.t), "n_subsidized": n_sub,
            "min_budget": float(min(env.budget_history)) if env.budget_history else 0.0,
            "final_budget": float(env.B),
            "bankrupt": bool(env._check_bankrupt()),
            "accounting_err": abs(env.B - (B - c * env.t + rev)),
        })
    return _aggregate(results)
