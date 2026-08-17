#!/usr/bin/env python3
"""
experiments/item3_myopic_v2.py — ITEM 3: C2 Myopic Control (full k range).
===========================================================================
Cal-DP with lookahead REMOVED: at each step select
    tau* = argmax_tau  A[d][m][tau] * price(tau)
using the SAME calibrated tables. No V lookup. No capital planning.
Feasibility (B >= c) and skip rules unchanged.

Same calibrated tables; only the selection criterion changes.
NEW FILE — does not modify c2_myopic_caldp.py or the Cal-DP implementation.

k = [5,10,15,20,30,40]. All 5 networks. 10 seeds.
Reports paired difference, 95% CI, p-value vs full Cal-DP (from budget shards).
Writes: results/logs/item3_myopic_v2.json
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.env.polblogs_loader import load_polblogs
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated import _deg_class

C = 0.3; W_HIGH = 1.0
SEEDS = list(range(10))
KS    = [5, 10, 15, 20, 30, 40]
TIERS = [1.0, 0.8, 0.5, 0.2, 0.0]
NETS  = {
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "Rice_FB":    load_rice_facebook,
    "polblogs":   load_polblogs,
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
}
OUT      = "results/logs/item3_myopic_v2.json"
IN_TMPL  = "results/logs/budget_10s_{net}.json"
T_CRIT   = stats.t.ppf(0.975, df=9)


def _infl_bucket(x, ib_arr):
    for i in range(len(ib_arr)-2, 0, -1):
        if x >= ib_arr[i]: return i
    return 0


def myopic_episode(graph, cfg, V, A, class_bnd, infl_bnd):
    """One episode: myopic tier selection (no V, pure argmax A*price)."""
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    total = 0.0
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
            val = p_acc * price   # NO V LOOKUP — myopic only
            if val > best_val:
                best_val = val; best_disc = d

        _, reward, done, _ = env.step(ni, best_disc)
        total += reward
        if done: break
    return total


def _ci(d: np.ndarray):
    se = d.std(ddof=1) / math.sqrt(len(d))
    m  = d.mean()
    return {"mean_diff": round(float(m),2),
            "ci95": [round(float(m - T_CRIT*se),2), round(float(m + T_CRIT*se),2)],
            "pval": round(float(stats.ttest_1samp(d, 0).pvalue),4),
            "sig":  bool(abs(m) - T_CRIT*se > 0)}


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT}"); return

    results = {"note": "myopic = argmax(A*price), no V; full CalDP from budget shards"}

    for net, loader in NETS.items():
        graph = loader()
        print(f"\n=== {net}: calibrating... ===")
        cal_cfg = BudgetEnvConfig(budget_B=1.5, production_cost=C, seed=0, weight_high=W_HIGH)
        V, A, P, class_bnd, infl_bnd = calibrate_v2_obs_table(graph, cal_cfg, n_sims=30, seed=0)

        # Load full Cal-DP per-seed numbers from budget shard
        shard_path = IN_TMPL.format(net=net)
        cdp_shard = None
        if os.path.exists(shard_path):
            cdp_shard = json.load(open(shard_path))

        results[net] = {}
        for k in KS:
            B = k * C
            myopic_v = []
            for s in SEEDS:
                cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s, weight_high=W_HIGH)
                myopic_v.append(myopic_episode(graph, cfg, V, A, class_bnd, infl_bnd))

            m_arr = np.array(myopic_v)
            cell  = {"myopic_mean": round(float(m_arr.mean()),2),
                     "myopic_std":  round(float(m_arr.std()),2),
                     "myopic_all":  [round(x,2) for x in myopic_v]}

            # Paired comparison vs full Cal-DP
            if cdp_shard and str(k) in cdp_shard.get("results",{}):
                cdp_all = cdp_shard["results"][str(k)].get("Cal-DP",{}).get("all",[])
                if len(cdp_all) == 10:
                    diff = m_arr - np.array(cdp_all, dtype=float)
                    cell["caldp_mean"] = round(float(np.mean(cdp_all)),2)
                    cell["paired_myopic_minus_caldp"] = _ci(diff)
            else:
                cell["caldp_note"] = "shard not available"

            results[net][str(k)] = cell
            sig = cell.get("paired_myopic_minus_caldp",{}).get("sig","?")
            cdp_m = cell.get("caldp_mean","?")
            print(f"  k={k:2d}  myopic={m_arr.mean():.1f}  caldp={cdp_m}  "
                  f"diff={cell.get('paired_myopic_minus_caldp',{}).get('mean_diff','?')}  sig={sig}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT,"w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")

if __name__ == "__main__":
    main()
