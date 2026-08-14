"""
Re-sweep Cal-DP with observation-only calibration (frozen budget: 5x5=25k).
Per-k value = composite max(v2_obs, v3_obs).
Writes one shard per network; safe for parallel execution.

Usage:
    python experiments/resweep_caldp_obs.py [--networks NET ...] \
        [--k-values K ...] [--out results/logs/caldp_obs_<net>.json]
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.evaluation.dp_calibrated_v2_obs import (
    build_graph_cache, calibrate_v2_obs_table, eval_v2_obs_revenue,
)
from src.evaluation.dp_calibrated_v3_obs import (
    calibrate_v3_obs_table, eval_v3_obs_revenue,
)

# ── frozen constants ─────────────────────────────────────────────────────────
NETWORKS_ALL  = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
K_VALUES_ALL  = [5, 10, 15, 20, 30, 40]
SEEDS         = [42, 123, 7]
C             = 0.3
N_PASSES      = 5
N_SIMS        = 5          # 5×5 = 25,000 observed offers (commit 328cf31)
CALIB_DIR     = "results/logs"

def get_outpath(net, out_arg):
    if out_arg:
        return out_arg
    os.makedirs(CALIB_DIR, exist_ok=True)
    return os.path.join(CALIB_DIR, f"caldp_obs_{net}.json")

def load_calib(net, graph_cache, v):
    calib_path = os.path.join(CALIB_DIR, f"caldp_obs_calib_{net}.npz")
    if os.path.exists(calib_path):
        d = np.load(calib_path, allow_pickle=True)
        if v == 2:
            return d["v2_table"].item()
        return d["v3_table"].item()
    return None

def save_calib(net, v2_table, v3_table):
    os.makedirs(CALIB_DIR, exist_ok=True)
    calib_path = os.path.join(CALIB_DIR, f"caldp_obs_calib_{net}.npz")
    np.savez(calib_path, v2_table=np.array(v2_table, dtype=object),
             v3_table=np.array(v3_table, dtype=object))
    print(f"  [calib saved] {calib_path}")

def run_network(net, k_values, out_path):
    print(f"\n=== {net} | k={k_values} ===")
    graph_cache = build_graph_cache(net)
    n = graph_cache["n"]

    calib_path = os.path.join(CALIB_DIR, f"caldp_obs_calib_{net}.npz")
    if os.path.exists(calib_path):
        d = np.load(calib_path, allow_pickle=True)
        v2_table = d["v2_table"].item()
        v3_table = d["v3_table"].item()
        print(f"  [calib loaded from cache]")
    else:
        t0 = time.time()
        print(f"  calibrating v2_obs (5x5)...")
        v2_table = calibrate_v2_obs_table(graph_cache, n_sims=N_SIMS,
                                          n_passes=N_PASSES, seed_offset=0)
        print(f"  calibrating v3_obs (5x5)...")
        v3_table = calibrate_v3_obs_table(graph_cache, n_sims=N_SIMS,
                                          n_passes=N_PASSES, seed_offset=0)
        print(f"  calibration done in {time.time()-t0:.1f}s")
        save_calib(net, v2_table, v3_table)

    results = {}
    for k in k_values:
        budget = k * C
        revs_v2, revs_v3, revs_comp = [], [], []
        for seed in SEEDS:
            r2 = eval_v2_obs_revenue(graph_cache, v2_table, k=k,
                                     budget=budget, seed=seed,
                                     reprice_mode="skip")
            r3 = eval_v3_obs_revenue(graph_cache, v3_table, k=k,
                                     budget=budget, seed=seed,
                                     reprice_mode="skip")
            revs_v2.append(r2)
            revs_v3.append(r3)
            revs_comp.append(max(r2, r3))
        mean_v2   = float(np.mean(revs_v2))
        mean_v3   = float(np.mean(revs_v3))
        mean_comp = float(np.mean(revs_comp))
        print(f"  k={k:2d}  v2={mean_v2:.2f}  v3={mean_v3:.2f}  comp={mean_comp:.2f}")
        results[k] = {
            "v2_obs": round(mean_v2, 4),
            "v3_obs": round(mean_v3, 4),
            "composite": round(mean_comp, 4),
            "seeds": {str(s): {"v2": round(v2, 4), "v3": round(v3, 4)}
                      for s, v2, v3 in zip(SEEDS, revs_v2, revs_v3)},
        }

    shard = {"network": net, "calib_budget": "5x5=25000",
             "c": C, "seeds": SEEDS, "results": results}
    with open(out_path, "w") as f:
        json.dump(shard, f, indent=2)
    print(f"  [written] {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS_ALL)
    ap.add_argument("--k-values", nargs="+", type=int, default=K_VALUES_ALL)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    for net in args.networks:
        out = get_outpath(net, args.out if len(args.networks) == 1 else None)
        run_network(net, args.k_values, out)

if __name__ == "__main__":
    main()
