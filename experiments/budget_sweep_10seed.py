#!/usr/bin/env python3
"""
experiments/budget_sweep_10seed.py  — Block A (reviewer M2)
===========================================================
Budget sweep with 10 weight seeds (0..9) for 5 methods:
  IE+Budget, Greedy+Budget, Cal-DP (obs-v2/v3 composite),
  CGS (arm3, lambda=1.0), Rev-GNN-LSTM (arm_b sha=0b549f93 FF+BA)

Protocol: BudgetRevenueEnv, W_HIGH=2.0 (Def 2.1), c=0.3, B=k*c,
  N_MC=200, k in [5,10,15,20,30,40], 5 networks.
Per-seed arrays: revenue, profit (rev - c*|S_T|), n_in_S, n_below_cost.

Writes: results/logs/budget_10s_{NET}.json
Merge + paired tests: python experiments/merge_budget_10seed.py
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.budget_baselines import greedy_discount_budget, _make_env
from src.evaluation.dp_calibrated import _deg_class
from src.evaluation.dp_calibrated_v2 import _plan_dp_v2, _execute_v2
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated_v3 import _plan_dp_v3, _execute_v3, _N_S_BUCKETS
from src.evaluation.dp_calibrated_v3_obs import calibrate_v3_obs_table
from _cal_episode_utils import calibrate, arm3_episode

try:
    from src.evaluation.ie_budget import (
        ie_strategy_budget, _greedy_seed_selection_celf, IE_K_SEEDS as _IE_K_SEEDS,
    )
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget
    _greedy_seed_selection_celf = None
    _IE_K_SEEDS = 30

from _arm_b_utils import load_arm_b, make_ei, eval_arm_b_k_full, ARM_B_SHA

# ── Protocol ──────────────────────────────────────────────────────────────────
K_VALUES     = [5, 10, 15, 20, 30, 40]
C            = 0.3
N_TRIALS     = 10
SEEDS        = list(range(N_TRIALS))
N_SIMS       = 5
W_HIGH       = 2.0    # Uniform(0, W_HIGH) per Definition 2.1
N_MC         = 200    # MC samples for valuation estimate
CGS_LAM      = 1.0
TIERS        = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA        = 0.05
NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]

assert W_HIGH == 2.0, f"W_HIGH must be 2.0 per Def 2.1; got {W_HIGH}"


def load_graph(net: str):
    if net == "polblogs":     return load_polblogs()
    if net == "FF_1000":      return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":      return load_rice_facebook()
    if net == "Modular_FF":   return generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    if net == "FF_2000":      return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def _raw_key(r, key, n_trials):
    """Extract per-seed array for 'key' from result dict."""
    sub = r.get(key, {})
    if isinstance(sub, dict):
        arr = sub.get("all", None)
        if arr is not None: return arr[:n_trials]
        m = sub.get("mean", 0.0)
        return [m] * n_trials
    if isinstance(sub, list): return sub[:n_trials]
    return [0.0] * n_trials


def _raw(r, n_trials):
    """Extract per-seed revenue array."""
    for key in ("total_revenue", "revenue"):
        arr = _raw_key(r, key, n_trials)
        if any(v != 0.0 for v in arr): return arr
    return [0.0] * n_trials


def _profit_from_dict(r, n_trials):
    """Extract profit if available, else compute from revenue - C*n_in_S."""
    p_arr = _raw_key(r, "profit", n_trials)
    if any(v != 0.0 for v in p_arr): return p_arr
    rev = _raw(r, n_trials)
    s_t = _raw_key(r, "n_in_S", n_trials)
    if any(v != 0.0 for v in s_t):
        return [rv - C * st for rv, st in zip(rev, s_t)]
    return [float("nan")] * n_trials


def _stats(vals):
    a = np.array([v for v in vals if not np.isnan(v)], dtype=float)
    if len(a) == 0:
        return {"mean": float("nan"), "std": float("nan"), "all": list(vals)}
    return {"mean": round(float(a.mean()), 3), "std": round(float(a.std()), 3),
            "all": [round(float(v), 3) for v in vals]}


def _caldp_execute(graph, cfg, V2, A2, P2, cb2, ib2, V3, A3, T3, cb3, sb3, k):
    """Plan + execute Cal-DP v2 and v3 for one k. Returns (v2_revs, v3_revs)."""
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)
    cpos     = np.array([_deg_class(int(all_deg[i]), cb2) for i in range(n)], dtype=np.int32)
    B = k * C; b_steps = max(1, int(B / DELTA) + 1)

    plan2 = _plan_dp_v2(n_total=n, V=V2, A=A2, P=P2, class_of_pos=cpos,
                        B=B, c=C, tiers=TIERS, delta=DELTA)
    dp3, tier3 = _plan_dp_v3(n_total=n, V3=V3, A3=A3, T=T3, class_of_pos=cpos,
                              B=B, c=C, sb_size=sb3, n_s_buckets=_N_S_BUCKETS,
                              tiers=TIERS, delta=DELTA)
    v2_revs, v3_revs = [], []
    for seed in range(N_TRIALS):
        env2 = _make_env(graph, B=B, c=C, seed=seed, weight_high=cfg.weight_high)
        env2.reset()
        rev2, _, _ = _execute_v2(env=env2, ordering=ordering, plan=plan2,
                                  V=V2, A=A2, class_boundaries=cb2, infl_boundaries=ib2,
                                  c=C, class_of_pos=cpos,
                                  dp_table=[[0.0]*(n+1) for _ in range(b_steps+1)],
                                  b_steps=b_steps, delta=DELTA, tiers=TIERS)
        env3 = _make_env(graph, B=B, c=C, seed=seed, weight_high=cfg.weight_high)
        env3.reset()
        rev3, _, _, _ = _execute_v3(env=env3, ordering=ordering, dp3=dp3, tier3=tier3,
                                     V3=V3, A3=A3, class_boundaries=cb3, sb_size=sb3,
                                     c=C, class_of_pos=cpos, n_s_buckets=_N_S_BUCKETS,
                                     b_steps=b_steps, delta=DELTA, tiers=TIERS, log_steps=0)
        v2_revs.append(float(rev2)); v3_revs.append(float(rev3))
    return v2_revs, v3_revs


def run_network(net: str, out_path: str, device):
    graph = load_graph(net)
    cfg   = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH, n_mc_samples=N_MC)
    assert cfg.weight_high == 2.0, f"env W_HIGH={cfg.weight_high}"
    print(f"\n=== {net}  W_HIGH={cfg.weight_high}  N_MC={cfg.n_mc_samples} ===", flush=True)

    arm_b = load_arm_b(device)
    ei, cache = make_ei(graph, device)

    # ── All calibrations ONCE per network ─────────────────────────────────────
    t_cal = time.time()
    print(f"  [1/3] CGS calibrate...", flush=True)
    V, A, P, cb, ib = calibrate(graph, cfg)
    t_cgs = time.time() - t_cal; print(f"        CGS done  {t_cgs:.0f}s", flush=True)

    print(f"  [2/3] Cal-DP v2 calibrate (n_sims={N_SIMS})...", flush=True)
    V2, A2, P2, cb2, ib2 = calibrate_v2_obs_table(graph, cfg, n_sims=N_SIMS, seed=0)
    t_v2 = time.time()-t_cal-t_cgs; print(f"        v2 done  {t_v2:.0f}s", flush=True)

    print(f"  [3/3] Cal-DP v3 calibrate (n_sims={N_SIMS})...", flush=True)
    V3, A3, T3, cb3, sb3 = calibrate_v3_obs_table(graph, cfg, n_sims=N_SIMS, seed=0)
    t_v3 = time.time()-t_cal-t_cgs-t_v2; print(f"        v3 done  {t_v3:.0f}s", flush=True)
    print(f"  Total calibration: {time.time()-t_cal:.0f}s  "
          f"(CGS={t_cgs:.0f}s  v2={t_v2:.0f}s  v3={t_v3:.0f}s)", flush=True)

    # ── IE seed orderings: compute ONCE per (network, trial) via CELF ─────────
    ie_seed_orderings = None
    if _greedy_seed_selection_celf is not None:
        t_ie_pre = time.time()
        ie_seed_orderings = {}
        for trial in range(N_TRIALS):
            env_tmp = _make_env(graph, B=float("inf"), c=C, seed=trial,
                                weight_high=W_HIGH)
            env_tmp.reset()
            ie_seed_orderings[trial] = _greedy_seed_selection_celf(
                graph, env_tmp, _IE_K_SEEDS)
        print(f"  IE CELF precomputed ({N_TRIALS} trials) in "
              f"{time.time()-t_ie_pre:.1f}s", flush=True)

    results = {}
    for k in K_VALUES:
        B  = k * C
        t0 = time.time()

        # ── IE + Greedy (fast, no calibration, W_HIGH explicit) ───────────────
        t_ie = time.time()
        r_ie  = ie_strategy_budget(graph, B, C, n_trials=N_TRIALS, weight_high=W_HIGH,
                                   seed_orderings=ie_seed_orderings)
        r_gd  = greedy_discount_budget(graph, B, C, n_trials=N_TRIALS, weight_high=W_HIGH)
        ie_raw = _raw(r_ie, N_TRIALS); gd_raw = _raw(r_gd, N_TRIALS)
        ie_prof = _profit_from_dict(r_ie, N_TRIALS); gd_prof = _profit_from_dict(r_gd, N_TRIALS)
        ie_s_t  = _raw_key(r_ie, "n_in_S", N_TRIALS); gd_s_t = _raw_key(r_gd, "n_in_S", N_TRIALS)
        t_ie_done = time.time()-t_ie

        # ── Cal-DP: plan+execute only (calibration already done above) ────────
        t_cdp = time.time()
        v2_raw, v3_raw = _caldp_execute(graph, cfg, V2, A2, P2, cb2, ib2, V3, A3, T3, cb3, sb3, k)
        cdp_raw = [max(a, b) for a, b in zip(v2_raw, v3_raw)]
        t_cdp_done = time.time()-t_cdp

        # ── CGS (arm3, lambda=1.0) ────────────────────────────────────────────
        # arm3_episode returns (revenue, n_sk) where n_sk is int (|S_T|)
        cgs_revs, cgs_profs, cgs_s_ts, cgs_bcs = [], [], [], []
        for seed in SEEDS:
            env = _make_env(graph, B=B, c=C, seed=seed, weight_high=cfg.weight_high)
            env.reset()
            # DIAGNOSTIC: print realized edge weights for seed=0, first k only
            if seed == 0 and k == K_VALUES[0]:
                _w = np.array(list(env._link_weights.values()))
                print(f"  [DIAG net={net} k={k} seed=0] "
                      f"cfg.W_HIGH={cfg.weight_high:.1f}  "
                      f"link_mean={_w.mean():.4f}  link_max={_w.max():.4f}",
                      flush=True)
            try:
                rev, n_sk = arm3_episode(env, graph, A, V, cb, ib, CGS_LAM)
                # n_sk may be int or dict (future-proof)
                if isinstance(n_sk, dict):
                    n_s = n_sk.get("n_in_S", 0); bc = n_sk.get("n_below_cost", 0)
                else:
                    n_s = int(n_sk); bc = 0
                cgs_revs.append(float(rev)); cgs_s_ts.append(n_s); cgs_bcs.append(bc)
                cgs_profs.append(float(rev) - C * n_s)
            except Exception as e:
                print(f"  CGS err seed={seed}: {e}")
                cgs_revs.append(float("nan")); cgs_profs.append(float("nan"))
                cgs_s_ts.append(0); cgs_bcs.append(0)

        # ── arm_b (LSTM, sha=0b549f93) ────────────────────────────────────────
        ab_revs, ab_s_ts, ab_bcs, ab_profs = eval_arm_b_k_full(
            arm_b, graph, cache, ei, B, N_TRIALS, device)

        elapsed = time.time() - t0
        results[k] = {
            "IE+Budget":     {**_stats(ie_raw),  "profit": _stats(ie_prof),  "n_in_S": _stats(ie_s_t)},
            "Greedy+Budget": {**_stats(gd_raw),  "profit": _stats(gd_prof),  "n_in_S": _stats(gd_s_t)},
            "Cal-DP":        _stats(cdp_raw),
            "CGS":           {**_stats(cgs_revs), "profit": _stats(cgs_profs),
                              "n_in_S": _stats(cgs_s_ts), "n_below": _stats(cgs_bcs)},
            "Rev-GNN-LSTM":  {**_stats(ab_revs), "profit": _stats(ab_profs),
                              "n_in_S": _stats(ab_s_ts), "n_below": _stats(ab_bcs)},
        }
        print(f"  k={k:2d}  IE={_stats(ie_raw)['mean']:.1f}"
              f"  GD={_stats(gd_raw)['mean']:.1f}"
              f"  CDP={_stats(cdp_raw)['mean']:.1f}"
              f"  CGS={_stats(cgs_revs)['mean']:.1f}"
              f"  GNN={_stats(ab_revs)['mean']:.1f}"
              f"  ({elapsed:.0f}s)", flush=True)

    shard = {"network": net, "n_trials": N_TRIALS, "seeds": SEEDS,
             "W_HIGH": W_HIGH, "N_MC": N_MC,
             "shas": {"arm_b": ARM_B_SHA}, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(shard, f, indent=2)
    print(f"Saved → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS_ALL)
    ap.add_argument("--out-dir",  default="results/logs")
    ap.add_argument("--gpu",      type=int, default=0)
    ap.add_argument("--force",    action="store_true", help="Overwrite existing shards")
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[budget_sweep_10seed] device={device}  W_HIGH={W_HIGH}  N_MC={N_MC}"
          f"  n_trials={N_TRIALS}  sha={ARM_B_SHA}")

    for net in args.networks:
        out = os.path.join(args.out_dir, f"budget_10s_{net}.json")
        if os.path.exists(out) and not args.force:
            print(f"SKIP {net}: {out} already exists (use --force to overwrite)"); continue
        run_network(net, out, device)


if __name__ == "__main__":
    main()
