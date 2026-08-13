#!/usr/bin/env python3
"""eval_ie_budget_ksweep.py — IE-Strategy+Budget k-sweep on all 5 networks.

Protocol identical to eval_all_methods_ksweep.py:
  BudgetRevenueEnv, c=0.3, B0=k*c, seeds [0,1,2], weight_high=2.0, n_trials=3.

Outputs per shard:
  results/logs/budget_sweep_IE_{net}_{krange}.json

Usage:
  # All networks, all k:
  python -u experiments/eval_ie_budget_ksweep.py

  # Single network, subset k (for parallel launch):
  python -u experiments/eval_ie_budget_ksweep.py --networks FF_1000
  python -u experiments/eval_ie_budget_ksweep.py --networks polblogs --k-values 5 10 15
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Betweenness approximation (avoids O(n^3) on large graphs)
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.evaluation.ie_budget import ie_strategy_budget, IE_K_SEEDS

# ── Protocol (mirrors eval_all_methods_ksweep.py) ────────────────────────────
K_VALUES = [5, 10, 15, 20, 30, 40]
C        = 0.3
W_HIGH   = 2.0
N_TRIALS = 3   # match existing sweep

METHOD   = "ie_budget"   # key written into each result cell

LOG_DIR  = "results/logs"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", nargs="+", default=None, metavar="NET")
    p.add_argument("--k-values", nargs="+", type=int, default=None, metavar="K")
    return p.parse_args()


def main():
    args  = _parse_args()
    k_filter  = args.k_values if args.k_values else K_VALUES
    net_filter = args.networks if args.networks else None

    # Derive shard filename
    net_tag = "_".join(net_filter) if net_filter else "all"
    k_tag   = "k" + "-".join(str(k) for k in k_filter)
    log_out = os.path.join(LOG_DIR, f"budget_sweep_IE_{net_tag}_{k_tag}.json")

    print(f"[IE-sweep] k={k_filter}  nets={net_filter or 'all'}  "
          f"k_seeds={IE_K_SEEDS}  C={C}  N_TRIALS={N_TRIALS}", flush=True)
    print(f"[IE-sweep] output → {log_out}", flush=True)

    t_start = time.time()

    # Load networks
    print("\n[IE-sweep] Loading networks...", flush=True)
    all_nets = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250, 250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }
    if net_filter:
        missing = [n for n in net_filter if n not in all_nets]
        if missing:
            print(f"ABORT: unknown network(s): {missing}", flush=True)
            sys.exit(1)
        all_nets = {n: all_nets[n] for n in net_filter}

    all_results: dict = {}
    for net_name, G in all_nets.items():
        all_results[net_name] = {}
        n = G.number_of_nodes()
        print(f"\n[IE-sweep] {net_name}  n={n}", flush=True)
        for k in k_filter:
            B   = k * C
            t_k = time.time()
            print(f"  k={k}  B={B:.1f} ...", end=" ", flush=True)

            res = ie_strategy_budget(G, B=B, c=C, k_seeds=IE_K_SEEDS,
                                     n_trials=N_TRIALS, weight_high=W_HIGH)
            # res['revenue'] is {"mean": float, "std": float, "all": list}
            rev_mean = float(res["revenue"]["mean"])

            all_results[net_name][k] = {METHOD: rev_mean}
            print(f"{METHOD}={rev_mean:.2f}  ({time.time()-t_k:.0f}s)", flush=True)

    wall = time.time() - t_start
    print(f"\n[IE-sweep] wall={wall:.0f}s  ({wall/60:.1f} min)", flush=True)

    # Save shard
    os.makedirs(LOG_DIR, exist_ok=True)
    out = {
        "protocol": {
            "method": METHOD, "k_seeds": IE_K_SEEDS,
            "k_values": k_filter, "C": C, "n_trials": N_TRIALS,
            "weight_high": W_HIGH, "networks": list(all_results.keys()),
        },
        "results": all_results,
        "wall_s":  wall,
    }
    with open(log_out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {log_out}", flush=True)


if __name__ == "__main__":
    main()
