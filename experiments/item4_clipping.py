#!/usr/bin/env python3
"""
experiments/item4_clipping.py — ITEM 4: G5 Clipping Fractions.
===============================================================
For FF_1000 and Rice_FB at k=5,20,40, over 10 Cal-DP episodes:
  - fraction of steps where env.B > clip_threshold (= b_steps * delta)
  - maximum balance reached, vs B0
  - mean revenue (current grid)

Extended-grid variant is NOT run: recalibrating with larger budget range
requires re-running calibrate_v2_obs_table with a different B parameter,
then re-running all 10-seed episodes; estimated ~2h for both networks.
Only clipping fractions and max balance are reported.

Grid-index line (dp_calibrated_v2.py:339):
    b_idx_curr = min(int(b_curr / delta), b_steps)
B_max line (dp_calibrated_v2.py:221):
    b_steps = max(1, int(B / delta) + 1)
Balance increases on profitable sale (budget_revenue_env.py:169):
    self.B = self.B - self.production_cost + info["offered_price"]

Writes: results/logs/item4_clipping.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated import _deg_class

C = 0.3; W_HIGH = 1.0; DELTA = 0.05
SEEDS = list(range(10))
KS    = [5, 20, 40]
NETS  = {
    "FF_1000":  lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":  load_rice_facebook,
}
TIERS = [1.0, 0.8, 0.5, 0.2, 0.0]
OUT   = "results/logs/item4_clipping.json"


def _infl_bucket(x, ib_arr):
    for i in range(len(ib_arr)-2, 0, -1):
        if x >= ib_arr[i]: return i
    return 0


def caldp_episode_with_tracking(graph, cfg, V, A, class_bnd, infl_bnd, clip_threshold):
    """Run Cal-DP episode and track budget at every step."""
    B0 = cfg.budget_B
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    total = 0.0
    budget_trace = []    # env.B after each step

    for node in ordering:
        if node in getattr(env, "offered", set()): continue
        if env._check_bankrupt(): break
        ni = env.node_to_idx.get(node)
        if ni is None: continue
        if env.B < C - 1e-9: break

        cls = int(_deg_class(int(graph.degree(node)), class_bnd))
        try: infl = float(env.get_current_influence(node))
        except: infl = 0.0
        ib = min(_infl_bucket(infl, infl_bnd), A.shape[1]-1)

        best_val = -1e18; best_disc = 0.0
        for ti, d in enumerate(TIERS):
            est = float(env._estimate_valuation(node))
            price = est * (1.0 - d)
            if env.B - C + price < -1e-9: continue
            p_acc = float(A[cls, ib, ti])
            val = p_acc * price + float(V[cls, ib])
            if val > best_val:
                best_val = val; best_disc = d

        _, reward, done, _ = env.step(ni, best_disc)
        total += reward
        budget_trace.append(float(env.B))  # record AFTER step
        if done: break

    n_steps = len(budget_trace)
    n_clipped = sum(1 for b in budget_trace if b > clip_threshold + 1e-9)
    max_b = max(budget_trace) if budget_trace else B0
    return {
        "revenue":     round(total, 3),
        "n_steps":     n_steps,
        "n_clipped":   n_clipped,
        "clip_frac":   round(n_clipped / max(n_steps, 1), 4),
        "max_balance": round(max_b, 4),
        "B0":          round(B0, 4),
        "max_over_B0": round(max_b - B0, 4),
    }


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT}"); return

    results = {
        "clip_formula": "b_steps=max(1,int(B0/delta)+1); clip_threshold=b_steps*delta; clipped iff env.B > clip_threshold",
        "delta":  DELTA,
        "extended_grid": "NOT RUN — requires recalibration (~2h); only clipping fractions reported",
    }

    for net, loader in NETS.items():
        graph = loader()
        print(f"\n=== {net} ===")
        cal_cfg = BudgetEnvConfig(budget_B=1.5, production_cost=C, seed=0, weight_high=W_HIGH)
        V, A, P, class_bnd, infl_bnd = calibrate_v2_obs_table(graph, cal_cfg, n_sims=30, seed=0)

        results[net] = {}
        for k in KS:
            B0 = k * C
            b_steps = max(1, int(B0 / DELTA) + 1)
            clip_threshold = b_steps * DELTA

            ep_results = []
            for s in SEEDS:
                cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=s, weight_high=W_HIGH)
                r = caldp_episode_with_tracking(graph, cfg, V, A, class_bnd, infl_bnd, clip_threshold)
                ep_results.append(r)

            clip_fracs = [r["clip_frac"] for r in ep_results]
            max_balances = [r["max_balance"] for r in ep_results]
            revenues = [r["revenue"] for r in ep_results]

            agg = {
                "B0":              round(B0, 3),
                "clip_threshold":  round(clip_threshold, 3),
                "clip_frac_mean":  round(float(np.mean(clip_fracs)), 4),
                "clip_frac_max":   round(float(np.max(clip_fracs)), 4),
                "max_balance_mean":round(float(np.mean(max_balances)), 3),
                "max_balance_max": round(float(np.max(max_balances)), 3),
                "max_over_B0_mean":round(float(np.mean([r["max_over_B0"] for r in ep_results])),3),
                "revenue_mean":    round(float(np.mean(revenues)), 2),
                "revenue_std":     round(float(np.std(revenues)), 2),
            }
            results[net][str(k)] = agg
            print(f"  k={k:2d}  B0={B0:.2f}  clip_thr={clip_threshold:.2f}  "
                  f"clip_frac={agg['clip_frac_mean']:.4f}  "
                  f"max_bal={agg['max_balance_max']:.3f}  "
                  f"rev={agg['revenue_mean']:.1f}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")

if __name__ == "__main__":
    main()
