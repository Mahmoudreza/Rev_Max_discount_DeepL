"""
experiments/resweep_caldp_obs.py
Re-sweep Cal-DP with observation-only calibration (frozen budget: 5x5=25k).
Per-k composite = max(v2_obs, v3_obs) mean revenue.
One shard per network; safe for parallel execution.

Usage:
    python experiments/resweep_caldp_obs.py [--networks NET ...] \
        [--k-values K ...] [--out PATH]
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget

# ── frozen constants ─────────────────────────────────────────────────────────
NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
K_VALUES_ALL = [5, 10, 15, 20, 30, 40]
C            = 0.3
N_SIMS       = 5          # 5×5 = 25,000 observed offers (commit 328cf31)
N_TRIALS     = 3          # seeds 0,1,2 — consistent with oracle Cal-DP baseline
CALIB_DIR    = "results/logs"


def load_graph(net):
    if net == "polblogs":
        return load_polblogs()
    elif net == "FF_1000":
        return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    elif net == "Rice_FB":
        return load_rice_facebook()
    elif net == "Modular_FF":
        return generate_modular_forest_fire([250, 250], 0.37, 0.32, 0.05, seed=0)
    elif net == "FF_2000":
        return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    else:
        raise ValueError(f"Unknown network: {net}")


def get_outpath(net, out_arg):
    if out_arg:
        return out_arg
    os.makedirs(CALIB_DIR, exist_ok=True)
    return os.path.join(CALIB_DIR, f"caldp_obs_{net}.json")


def run_network(net, k_values, out_path):
    print(f"\n=== {net} | k={k_values} ===", flush=True)
    cfg   = BudgetEnvConfig(production_cost=C, weight_high=1.0)
    graph = load_graph(net)

    results = {}
    for k in k_values:
        B = k * C
        t0 = time.time()
        r2 = dp_calibrated_v2_obs_budget(graph, cfg, B=B, c=C,
                                          n_trials=N_TRIALS, n_sims=N_SIMS)
        r3 = dp_calibrated_v3_obs_budget(graph, cfg, B=B, c=C,
                                          n_trials=N_TRIALS, n_sims=N_SIMS)
        v2  = float(r2["revenue"])
        v3  = float(r3["revenue"])
        comp = max(v2, v3)
        elapsed = time.time() - t0
        print(f"  k={k:2d}  v2={v2:.2f}  v3={v3:.2f}  comp={comp:.2f}  ({elapsed:.0f}s)",
              flush=True)
        results[k] = {
            "v2_obs":    round(v2,   4),
            "v3_obs":    round(v3,   4),
            "composite": round(comp, 4),
            "n_trials":  N_TRIALS,
            "n_sims":    N_SIMS,
        }

    shard = {"network": net, "calib_budget": "5x5=25000",
             "c": C, "n_trials": N_TRIALS, "n_sims": N_SIMS,
             "results": results}
    with open(out_path, "w") as f:
        json.dump(shard, f, indent=2)
    print(f"  [written] {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks",  nargs="+", default=NETWORKS_ALL)
    ap.add_argument("--k-values",  nargs="+", type=int, default=K_VALUES_ALL)
    ap.add_argument("--out",       default=None)
    args = ap.parse_args()

    for net in args.networks:
        out = get_outpath(net, args.out if len(args.networks) == 1 else None)
        run_network(net, args.k_values, out)


if __name__ == "__main__":
    main()
