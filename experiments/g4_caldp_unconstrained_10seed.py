#!/usr/bin/env python3
"""
experiments/g4_caldp_unconstrained_10seed.py — G4
==================================================
Cal-DP on the UNCONSTRAINED problem: c=0, B=0 (budget inactive).
All five networks. 10 seeds [0..9]. Reports alongside Greedy-Discount,
IE, and the learned policy from results/logs/ablation_unc_10seed.json
(if present; otherwise skips those columns).

Writes: results/logs/g4_caldp_unconstrained_10seed.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.budget_baselines import _make_env
from src.evaluation.dp_calibrated import _deg_class
from src.evaluation.dp_calibrated_v2 import _plan_dp_v2, _execute_v2
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table

NETWORKS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
SEEDS    = list(range(10))
TIERS    = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA    = 0.05
N_SIMS   = 5
C0 = 0.0; B0 = 0.0   # unconstrained: c=0, B=0
CFG_CALIB = BudgetEnvConfig(production_cost=0.3, weight_high=1.0)

GRAPH_LOADERS = {
    "polblogs":   load_polblogs,
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook,
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
}


def run_caldp_unc(net: str) -> dict:
    t0    = time.time()
    graph = GRAPH_LOADERS[net]()
    n     = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    V2, A2, P2, cb2, ib2 = calibrate_v2_obs_table(
        graph, CFG_CALIB, n_sims=N_SIMS, seed=0)
    cpos = np.array([_deg_class(int(all_deg[i]), cb2) for i in range(n)], dtype=np.int32)
    plan2 = _plan_dp_v2(n_total=n, V=V2, A=A2, P=P2, class_of_pos=cpos,
                        B=B0, c=C0, tiers=TIERS, delta=DELTA)

    revenues = []
    for seed in SEEDS:
        env = _make_env(graph, B=B0, c=C0, seed=seed, weight_high=1.0)
        env.reset()
        rev = _execute_v2(env, plan2, ordering, ib2, B=B0, delta=DELTA)
        revenues.append(float(rev))

    m, s = float(np.mean(revenues)), float(np.std(revenues))
    print(f"  {net:12s}  Cal-DP_unc={m:.1f}±{s:.1f}  ({time.time()-t0:.0f}s)")
    return {"mean": round(m, 2), "std": round(s, 2), "all": revenues}


def main():
    # Load existing unconstrained comparison (from ablation_unc_10seed.json)
    comp_path = "results/logs/ablation_unc_10seed.json"
    comp = {}
    if os.path.exists(comp_path):
        raw = json.load(open(comp_path))
        # expected keys: net → {"GD": {mean, std}, "arm_b_free": ..., "p1": ...}
        comp = raw.get("results", raw)

    print("G4: Cal-DP unconstrained, c=0 B=0, 10 seeds")
    print("-" * 55)
    results = {"shas": {"cal_dp": "n/a (no weights)", "policy": "0b549f93"}}

    for net in NETWORKS:
        r = run_caldp_unc(net)
        # Attach comparison columns if available
        extra = comp.get(net, {})
        entry = {"Cal-DP_unc": r}
        for meth in ["GD", "arm_b_free", "IE", "p1"]:
            if meth in extra:
                entry[meth] = extra[meth]
        results[net] = entry

    out = "results/logs/g4_caldp_unconstrained_10seed.json"
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
