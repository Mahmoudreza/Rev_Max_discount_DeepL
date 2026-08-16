#!/usr/bin/env python3
"""
experiments/caldp_offgraph.py — Block D2
=========================================
Cal-DP information-parity test:
  (on-graph)  calibrate on the TARGET graph (standard)
  (off-graph) calibrate on a DIFFERENT graph drawn from the same family
              (same generator + params, different seed) then evaluate on target.

Also includes B4: reports fraction of A and P calibration cells filled by
direct observation vs interpolated, for each network at n_sims=5 (25k offers).

Protocol: BudgetRevenueEnv, c=0.3, B=k*c, k=[5,10,15,20,30,40], 10 seeds.
Writes: results/logs/caldp_offgraph_{NET}.json
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.dp_calibrated_v2_obs import (
    dp_calibrated_v2_obs_budget, calibrate_v2_obs_table,
)
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget

K_VALUES = [5, 10, 15, 20, 30, 40]
C = 0.3; N_TRIALS = 10; N_SIMS = 5; W_HIGH = 1.0
NETWORKS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]

# Off-graph: different seed for same generator
OFFGRAPH_SEEDS = {"FF_1000": 99, "FF_2000": 99, "Modular_FF": 99,
                  "polblogs": None, "Rice_FB": None}  # real-world: no alternative


def load_graph(net, seed=None):
    s = seed
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000,0.37,0.32, seed=s if s else 0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250],0.37,0.32,0.05,
                                                                  seed=s if s else 0)
    if net == "FF_2000":    return generate_forest_fire(2000,0.37,0.32, seed=s if s else 1)
    raise ValueError(net)


def _composite(r2, r3):
    v2 = r2.get("revenue",{}).get("all", [r2.get("revenue",{}).get("mean",0.)]*N_TRIALS)
    v3 = r3.get("revenue",{}).get("all", [r3.get("revenue",{}).get("mean",0.)]*N_TRIALS)
    return [max(a,b) for a,b in zip(v2,v3)]


def calib_fill_fraction(A, P):
    """Fraction of cells with at least one direct observation (non-zero count)."""
    a_fill = float((A > 0).any(axis=2).mean())
    p_fill = float((P > 0).any(axis=2).mean()) if P is not None else float("nan")
    return a_fill, p_fill


def run_network(net, out_dir):
    graph_tgt  = load_graph(net)
    cfg        = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH)
    off_seed   = OFFGRAPH_SEEDS.get(net)
    has_offgraph = (off_seed is not None)
    graph_off  = load_graph(net, seed=off_seed) if has_offgraph else None
    print(f"\n=== {net} | off_graph={'yes' if has_offgraph else 'N/A'} ===", flush=True)

    # B4: fill fraction
    V, A, P, cb, ib = calibrate_v2_obs_table(graph_tgt, cfg, n_sims=N_SIMS, seed=0)
    af, pf = calib_fill_fraction(A, P)
    print(f"  B4 fill fractions: A={af:.2%}  P={pf:.2%}", flush=True)

    results = {}
    for k in K_VALUES:
        B = k * C; t0 = time.time()
        # On-graph (standard)
        r2o = dp_calibrated_v2_obs_budget(graph_tgt, cfg, B=B, c=C,
                                           n_trials=N_TRIALS, n_sims=N_SIMS)
        r3o = dp_calibrated_v3_obs_budget(graph_tgt, cfg, B=B, c=C,
                                           n_trials=N_TRIALS, n_sims=N_SIMS)
        on_raw = _composite(r2o, r3o)
        on_mean = round(float(np.mean(on_raw)),2)
        on_std  = round(float(np.std(on_raw)),2)

        # Off-graph (calibrate on different realisation)
        if has_offgraph:
            r2x = dp_calibrated_v2_obs_budget(graph_off, cfg, B=B, c=C,
                                               n_trials=N_TRIALS, n_sims=N_SIMS,
                                               eval_graph=graph_tgt)
            r3x = dp_calibrated_v3_obs_budget(graph_off, cfg, B=B, c=C,
                                               n_trials=N_TRIALS, n_sims=N_SIMS,
                                               eval_graph=graph_tgt)
            off_raw = _composite(r2x, r3x)
            off_mean = round(float(np.mean(off_raw)),2)
            off_std  = round(float(np.std(off_raw)),2)
        else:
            off_raw = None; off_mean = None; off_std = None

        results[k] = {
            "Cal-DP_on_graph":  {"mean": on_mean,  "std": on_std,
                                  "all": [round(v,2) for v in on_raw]},
            "Cal-DP_off_graph": {"mean": off_mean, "std": off_std,
                                  "all": [round(v,2) for v in off_raw] if off_raw else None},
        }
        print(f"  k={k:2d}  on={on_mean:.1f}±{on_std:.1f}"
              f"  off={'%.1f±%.1f' % (off_mean,off_std) if has_offgraph else 'N/A'}"
              f"  ({time.time()-t0:.0f}s)", flush=True)

    shard = {"network": net, "n_trials": N_TRIALS, "n_sims": N_SIMS,
             "B4_fill": {"A": round(af,4), "P": round(pf,4)},
             "off_graph_seed": off_seed, "results": results}
    out = os.path.join(out_dir, f"caldp_offgraph_{net}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f: json.dump(shard, f, indent=2)
    print(f"Saved → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS)
    ap.add_argument("--out-dir", default="results/logs")
    args = ap.parse_args()
    for net in args.networks:
        run_network(net, args.out_dir)


if __name__ == "__main__":
    main()
