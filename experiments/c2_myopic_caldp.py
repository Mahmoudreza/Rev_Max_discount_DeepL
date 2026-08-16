#!/usr/bin/env python3
"""
experiments/c2_myopic_caldp.py — C2: Myopic Cal-DP control
============================================================
Cal-DP with LOOKAHEAD REMOVED: at each step pick tau maximising
A[cls][ib][tau] * price using the same calibrated tables, no value function.

Budget protocol, all 5 networks, k=[5,10,20,40], 10 seeds.
Writes: results/logs/c2_myopic_caldp.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook)
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated import _graph_hash, _deg_class

C = 0.3; W_HIGH = 1.0; DELTA = 0.05
SEEDS = list(range(10))
KS    = [5, 10, 20, 40]
TIERS = (1.0, 0.8, 0.5, 0.2, 0.0)   # disc fractions (1.0=free, 0.0=full price)
NETS  = {
    "polblogs":   load_polblogs,
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook,
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
}
OUT = "results/logs/c2_myopic_caldp.json"


def _infl_bucket(x, ib_arr):
    for i in range(len(ib_arr)-2, 0, -1):
        if x >= ib_arr[i]: return i
    return 0


def myopic_episode(graph, env, V, A, P, class_bnd, infl_bnd, ordering, seed):
    """One episode: greedy-myopic tier selection with Cal-DP calibrated tables."""
    env.reset()
    n_classes = A.shape[0]; n_bkts = A.shape[1]
    all_degs  = np.array([graph.degree(v) for v in ordering], dtype=float)
    class_of  = [int(_deg_class(int(graph.degree(v)), class_bnd)) for v in ordering]

    revenue = 0.0; node_ptr = 0
    for k_rel in range(len(ordering)):
        if env._check_bankrupt() or len(env.offered) >= env.n: break
        while node_ptr < len(ordering) and ordering[node_ptr] in env.offered:
            node_ptr += 1
        if node_ptr >= len(ordering): break

        node = ordering[node_ptr]; node_ptr += 1
        est_val = env._estimate_valuation(node)
        try: infl = env.get_current_influence(node)
        except: infl = 0.0

        cls = class_of[k_rel] if k_rel < len(class_of) else 0
        ib  = _infl_bucket(float(infl), infl_bnd)
        ib  = min(ib, A.shape[1]-1)

        # Myopic: pick tier maximising A[cls][ib][t] * price, NO continuation
        best_score = -1e18; best_disc = 0.0
        for t_idx, disc in enumerate(TIERS):
            price = est_val * (1.0 - disc)
            if env.B - C + price < -1e-9: continue
            score = float(A[cls, ib, t_idx]) * price
            if score > best_score:
                best_score = score; best_disc = disc

        ni = env.node_to_idx.get(node)
        if ni is None: continue
        _, reward, done, info = env.step(ni, best_disc)
        revenue += reward
        if done: break

    return revenue


def run_net(net: str, graph) -> dict:
    cfg0 = BudgetEnvConfig(budget_B=1.5, production_cost=C, seed=0, weight_high=W_HIGH)
    V, A, P, cb, ib_arr = calibrate_v2_obs_table(graph, cfg0, n_sims=30, seed=0)
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    out = {}
    for k in KS:
        B = k * C
        revs = []
        for s in SEEDS:
            cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s, weight_high=W_HIGH)
            env = BudgetRevenueEnv(graph, cfg)
            r = myopic_episode(graph, env, V, A, P, cb, ib_arr, ordering, s)
            revs.append(r)
        a = np.array(revs)
        out[str(k)] = {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2), "all": revs}
        print(f"  {net} k={k:2d}  myopic={a.mean():.1f}")
    return out


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT} — skipping"); return
    print("C2: Myopic Cal-DP (no lookahead), all 5 nets, k=[5,10,20,40], 10 seeds")
    results = {"note": "greedy: argmax A[cls][ib][tau]*price, no dp continuation"}
    for net, loader in NETS.items():
        print(f"\n=== {net} ===")
        graph = loader()
        results[net] = run_net(net, graph)
    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")

if __name__ == "__main__":
    main()
