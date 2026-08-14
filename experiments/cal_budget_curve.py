"""experiments/cal_budget_curve.py
Calibration-budget sweep: revenue vs #observed offers.
Usage: python cal_budget_curve.py <n_sims> <out_json>
Runs Cal-DP v3_obs composite for FF_1000 seeds [42,123,7],
k=[5,10,20,40], c=0.3, and writes a single JSON result line.
"""
import sys, os, time, json
sys.path.insert(0, "/Users/reza/Desktop/revmax-aaai2027")
os.chdir("/Users/reza/Desktop/revmax-aaai2027")

import numpy as np
from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.dp_calibrated import _graph_hash, _deg_class
from src.evaluation.budget_baselines import _make_env
from src.evaluation.dp_calibrated_v3 import (
    _plan_dp_v3, _execute_v3, _N_CLASSES, _N_BUCKETS, _N_S_BUCKETS,
)
from src.evaluation.dp_calibrated_v3_obs import (
    calibrate_v3_obs_table, dp_calibrated_v3_obs_budget,
)

n_sims = int(sys.argv[1])
out_json = sys.argv[2]

cfg   = BudgetEnvConfig(production_cost=0.3, weight_high=1.0)
c     = 0.3
seeds = [42, 123, 7]
ks    = [5, 10, 20, 40]
n_tiers = 5
n = 1000
total_offers = n_tiers * n_sims * n  # per-graph-seed

# ── calibrate + track cell fill ──────────────────────────────────────────
def calibrate_with_stats(g, cfg, n_sims):
    """Returns (V3, A3, T, cb, sb_size, pct_observed)."""
    n_classes, n_s_buckets = _N_CLASSES, _N_S_BUCKETS
    gh = _graph_hash(g)
    cache = os.path.join(
        "results/logs",
        f"dp_calibration_v3_obs5_{gh}_nc{n_classes}_ns{n_s_buckets}_{n_sims}.npz",
    )
    # Use existing calibrate function (caches result)
    V3, A3_interp, T, cb, sb = calibrate_v3_obs_table(
        g, cfg, n_classes=n_classes, n_s_buckets=n_s_buckets,
        n_sims=n_sims, seed=0,
    )
    # Recompute raw fill fraction: count cells where direct obs exist.
    # Since the cache stores the INTERPOLATED A3, we reconstruct fill
    # by checking whether the interp was needed. Proxy: cells where
    # A3_interp[d,sb,t] equals a neighbour value are interpolated.
    # Simple proxy: load raw num/den arrays if cache exists with them,
    # else estimate from uniqueness.
    # Faster proxy: any (cls,sb) row that has at least 1 unique value per tier
    # is considered "observed". We count via variance across tiers.
    A3_obs_mask = np.var(A3_interp, axis=2) > 1e-6  # (n_classes, n_s_buckets)
    pct_obs = float(A3_obs_mask.mean())
    return V3, A3_interp, T, cb, sb, pct_obs

# ── run sweep ────────────────────────────────────────────────────────────
graphs = {s: generate_forest_fire(n, p=0.37, pb=0.32, seed=s) for s in seeds}

# Calibrate each graph (timed)
t0 = time.time()
caches = {}
pct_obs_list = []
for s in seeds:
    g = graphs[s]
    V3, A3, T, cb, sb, pct_obs = calibrate_with_stats(g, cfg, n_sims)
    caches[s] = (V3, A3, T, cb, sb)
    pct_obs_list.append(pct_obs)
cal_secs = time.time() - t0
pct_obs_mean = float(np.mean(pct_obs_list))

# Evaluate k=[5,10,20,40]
k_revs = {}
for k in ks:
    B = k * c
    revs = []
    for s in seeds:
        g = graphs[s]
        r = dp_calibrated_v3_obs_budget(
            g, cfg, B=B, c=c, n_trials=5, n_sims=n_sims,
        )
        revs.append(r["revenue"]["mean"])
    k_revs[k] = float(np.mean(revs))

result = {
    "n_sims": n_sims,
    "total_offers": total_offers,
    "passes_x_sims": f"5x{n_sims}",
    "k5": k_revs[5], "k10": k_revs[10],
    "k20": k_revs[20], "k40": k_revs[40],
    "mean_rev": float(np.mean(list(k_revs.values()))),
    "cal_secs": round(cal_secs, 1),
    "pct_cells_observed": round(pct_obs_mean * 100, 1),
}

with open(out_json, "w") as f:
    json.dump(result, f)
print(json.dumps(result))
