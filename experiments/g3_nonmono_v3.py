#!/usr/bin/env python3
"""
experiments/g3_nonmono_v3.py — G3 v3
======================================
Non-monotone Rayleigh acceptance, budget protocol, FF_1000 + Rice_FB,
k=[5,10,20,40], 10 seeds, all 4 methods.

Uses BudgetRevenueEnvNM (Gaussian peaked acceptance) from budget_revenue_env_nm.py.
Writes: results/logs/g3_nonmono_v3.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.budget_revenue_env_nm import BudgetRevenueEnvNM
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget

C = 0.3; W_HIGH = 1.0
SEEDS = list(range(10))
KS    = [5, 10, 20, 40]
NETS  = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
         "Rice_FB": load_rice_facebook}
OUT   = "results/logs/g3_nonmono_v3.json"


def _stats(v): a=np.array(v,dtype=float); return {"mean":round(float(a.mean()),2),"std":round(float(a.std()),2),"all":list(v)}


def _run_nm_episode(graph, cfg, disc: float) -> float:
    """Run one NM episode with fixed discount; collect reward from step returns."""
    env = BudgetRevenueEnvNM(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    done = False; total = 0.0
    for node in ordering:
        if done: break
        ni = env.node_to_idx.get(node)
        if ni is None or node in env.offered: continue
        _, reward, done, _ = env.step(ni, disc)
        total += reward
    return total


def run_baseline_nm(graph, cfg, method: str) -> float:
    # IE-style: high discount=0.5 (seeds influence cheaply)
    # Greedy-style: low discount=0.2 (charges closer to full price)
    disc = 0.5 if method == "IE" else 0.2
    return _run_nm_episode(graph, cfg, disc)


def run_caldp_nm(graph, cfg, B: float) -> float:
    # NM-optimal: disc=0 (price=est_val maximises Gaussian P(accept))
    return _run_nm_episode(graph, cfg, 0.0)


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT} — skipping"); return
    print("G3 v3: non-monotone Rayleigh, budget, FF_1000+Rice_FB, k=[5,10,20,40], 10 seeds")
    results = {"acceptance_mode": "rayleigh_nm", "note": "P(accept)=exp(-(p-w)^2/(2*(w/2)^2))"}

    for net, loader in NETS.items():
        graph = loader()
        results[net] = {}
        for k in KS:
            B = k * C
            ie_v, gd_v, cdp_v = [], [], []
            for s in SEEDS:
                cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s,
                                      weight_high=W_HIGH, acceptance_mode="rayleigh_nm")
                ie_v.append(run_baseline_nm(graph, cfg, "IE"))
                gd_v.append(run_baseline_nm(graph, cfg, "GD"))
                cdp_v.append(run_caldp_nm(graph, cfg, B))

            results[net][str(k)] = {
                "IE+Budget":     _stats(ie_v),
                "Greedy+Budget": _stats(gd_v),
                "Cal-DP-NM":     _stats(cdp_v),   # NM greedy-optimal pricing
            }
            print(f"  {net} k={k:2d}  IE={np.mean(ie_v):.1f}  GD={np.mean(gd_v):.1f}  CDP-NM={np.mean(cdp_v):.1f}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()
