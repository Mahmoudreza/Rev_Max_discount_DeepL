#!/usr/bin/env python3
"""
experiments/budget_sweep_10seed.py  — Block A (reviewer M2)
===========================================================
Budget sweep with 10 weight seeds (0..9) for 4 methods:
  IE+Budget, Greedy+Budget, Cal-DP (obs-v2/v3 composite), Rev-GNN-LSTM (arm_b FF+BA)

Protocol: BudgetRevenueEnv, c=0.3, B=k*c, k in [5,10,15,20,30,40], 5 networks.
Single learned policy: rev_gnn_lstm_densemix.pt (FF+BA), asserts sha=0b549f93.
One shard per network; safe for parallel execution.

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
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget

try:
    from src.evaluation.ie_budget import ie_strategy_budget
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget

from _arm_b_utils import load_arm_b, make_ei, eval_arm_b_k, ARM_B_SHA

# ── Protocol ──────────────────────────────────────────────────────────────────
K_VALUES     = [5, 10, 15, 20, 30, 40]
C            = 0.3
N_TRIALS     = 10
SEEDS        = list(range(N_TRIALS))
N_SIMS       = 5
W_HIGH       = 1.0
NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]


def load_graph(net: str):
    if net == "polblogs":     return load_polblogs()
    if net == "FF_1000":      return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":      return load_rice_facebook()
    if net == "Modular_FF":   return generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    if net == "FF_2000":      return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def _raw(r, n_trials):
    return r.get("total_revenue", r.get("revenue", {})).get("all", [
        r.get("total_revenue", r.get("revenue", {0: 0})).get("mean", 0.0)
    ] * n_trials)


def run_network(net: str, out_path: str, device):
    graph = load_graph(net)
    cfg   = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH)
    print(f"\n=== {net} ===", flush=True)
    arm_b = load_arm_b(device)
    ei, cache = make_ei(graph, device)

    results = {}
    for k in K_VALUES:
        B  = k * C
        t0 = time.time()

        r_ie  = ie_strategy_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        r_gd  = greedy_discount_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        r2    = dp_calibrated_v2_obs_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)
        r3    = dp_calibrated_v3_obs_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)

        ie_raw  = _raw(r_ie, N_TRIALS)
        gd_raw  = _raw(r_gd, N_TRIALS)
        v2_raw  = r2.get("revenue", {}).get("all", [r2.get("revenue",{}).get("mean",0.)]*N_TRIALS)
        v3_raw  = r3.get("revenue", {}).get("all", [r3.get("revenue",{}).get("mean",0.)]*N_TRIALS)
        cdp_raw = [max(a, b) for a, b in zip(v2_raw, v3_raw)]
        ab_raw  = eval_arm_b_k(arm_b, graph, cache, ei, B, N_TRIALS, device)

        def _stats(vals):
            a = np.array(vals, dtype=float)
            return {"mean": round(float(a.mean()),3), "std": round(float(a.std()),3),
                    "all": [round(v,3) for v in vals]}

        results[k] = {
            "IE+Budget":     _stats(ie_raw[:N_TRIALS]),
            "Greedy+Budget": _stats(gd_raw[:N_TRIALS]),
            "Cal-DP":        _stats(cdp_raw),
            "Rev-GNN-LSTM":  _stats(ab_raw),
        }
        elapsed = time.time() - t0
        print(f"  k={k:2d}  IE={results[k]['IE+Budget']['mean']:.1f}"
              f"  GD={results[k]['Greedy+Budget']['mean']:.1f}"
              f"  CDP={results[k]['Cal-DP']['mean']:.1f}"
              f"  GNN={results[k]['Rev-GNN-LSTM']['mean']:.1f}"
              f"  ({elapsed:.0f}s)", flush=True)

    shard = {"network": net, "n_trials": N_TRIALS, "seeds": SEEDS,
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
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[budget_sweep_10seed] device={device}  n_trials={N_TRIALS}  sha={ARM_B_SHA}")

    for net in args.networks:
        out = os.path.join(args.out_dir, f"budget_10s_{net}.json")
        if os.path.exists(out):
            print(f"SKIP {net}: {out} already exists"); continue
        run_network(net, out, device)


if __name__ == "__main__":
    main()
