#!/usr/bin/env python3
"""eval_baselines_budget_k50.py — Greedy+Budget baseline at k=50 on all 5 networks.

Contribution 2 (budget-constrained) baseline comparison.
Protocol: k=50, B=15 (=50*0.3), C=0.3, 1 seed, BudgetRevenueEnv, weight_high=2.0

Run on server:
    python -u experiments/eval_baselines_budget_k50.py 2>&1 | tee /tmp/baselines_k50.log

Output: results/logs/baselines_budget_k50.json
"""
from __future__ import annotations
import json, os, sys, time
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Approximate betweenness (k=200 pivots) to avoid O(n^3) on large graphs
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.evaluation.budget_baselines import greedy_discount_budget
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook

# ── Protocol constants (MUST match arm eval) ───────────────────────────────────
K      = 50        # seed set size / max offers
C      = 0.3       # production cost per item
B      = K * C     # budget = 15.0
b_RAY  = 1.0       # Rayleigh shape parameter
N_TRIALS = 1       # 1 seed (fast; enough for table entry)
W_HIGH   = 2.0     # weight_high for link weights Uniform(0, W_HIGH)

LOG_OUT = "results/logs/baselines_budget_k50.json"

# arm_b reference values for comparison table
EP80_REF = {
    "polblogs":   662.9,
    "FF_1000":    446.8,
    "Rice_FB":    216.6,
    "Modular_FF": 221.2,
    "FF_2000":    872.5,
}
# ep160 slots — fill in after server eval; placeholders here
EP160_REF = {
    "polblogs":   "?",
    "FF_1000":    "?",
    "Rice_FB":    "?",
    "Modular_FF": "?",
    "FF_2000":    "?",
}


def main():
    t_start = time.time()
    print(f"[baselines-k50] k={K}  B={B:.1f}  C={C}  N_TRIALS={N_TRIALS}", flush=True)

    # Load networks (same seed/params as arm eval)
    print("\n[baselines-k50] Loading networks...", flush=True)
    networks = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250, 250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }
    for name, G in networks.items():
        print(f"  {name}: n={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)

    # Run Greedy+Budget on each network
    results = {}
    for name, G in networks.items():
        t0 = time.time()
        print(f"\n[baselines-k50] === {name} ===", flush=True)
        res = greedy_discount_budget(G, B=B, c=C, b=b_RAY,
                                     n_trials=N_TRIALS, weight_high=W_HIGH)
        rev = res["revenue"]["mean"]
        results[name] = {"greedy_budget_rev": rev, "raw": res}
        print(f"  Greedy+Budget rev={rev:.1f}  ({time.time()-t0:.0f}s)", flush=True)

    # Print comparison table
    wall = time.time() - t_start
    print(f"\n{'─'*72}")
    print(f"{'network':<14} | {'Greedy+Budget':>13} | {'arm_b ep80':>10} | {'arm_b ep160':>11} | {'vs Greedy':>9}")
    print(f"{'─'*72}")
    for name in ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]:
        gb   = results[name]["greedy_budget_rev"]
        ep80 = EP80_REF[name]
        ep160 = EP160_REF[name]
        diff = ep80 - gb  # ep80 vs greedy (ep160 not yet available)
        ep160_str = f"{ep160:>11}" if isinstance(ep160, float) else f"{'?':>11}"
        print(f"{name:<14} | {gb:>13.1f} | {ep80:>10.1f} | {ep160_str} | {diff:>+9.1f}")
    print(f"{'─'*72}")
    print(f"\nWall time: {wall:.0f}s  ({wall/60:.1f} min)")

    # Save
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "protocol": {"k": K, "B": B, "C": C, "b_rayleigh": b_RAY,
                     "n_trials": N_TRIALS, "weight_high": W_HIGH},
        "results": {name: {"greedy_budget_rev": v["greedy_budget_rev"]}
                    for name, v in results.items()},
        "ep80_ref": EP80_REF,
        "wall_s": wall,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {LOG_OUT}")


if __name__ == "__main__":
    main()
