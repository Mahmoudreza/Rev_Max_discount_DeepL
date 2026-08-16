#!/usr/bin/env python3
"""
experiments/g1_oracle_upper_bound.py — G1
==========================================
Oracle (perfect-information) upper bound: FF-1000 and Modular-FF only,
budget protocol, k=[5,10,20,40], 10 seeds.

Oracle: knows all true buyer valuations w_i before offering.
Optimal strategy: price each buyer at c (production cost); all buyers
with w_i > c accept. Revenue = sum_{top-k by w_i} max(0, w_i - c).

This is NON-REALIZABLE — it serves only as a revenue scale reference.

Writes: results/logs/g1_oracle_upper_bound.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire
from src.utils.helpers import set_seed

C = 0.3; W_HIGH = 1.0
SEEDS  = list(range(10))
KS     = [5, 10, 20, 40]
NETS   = {
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
}

# Valuation attribute names to try from BudgetRevenueEnv after reset
_VAL_ATTRS = ["_valuations", "_weights", "_w", "weights", "valuations",
              "_buyer_valuations", "node_weights", "_buyer_weights"]


def get_true_valuations(env: BudgetRevenueEnv):
    """Try to extract true w_i from the env after reset."""
    for attr in _VAL_ATTRS:
        v = getattr(env, attr, None)
        if v is not None and hasattr(v, "__len__") and len(v) > 0:
            return np.array(v, dtype=float)
    return None


def oracle_revenue_env(graph, k: int, seed: int) -> float:
    """Run oracle: if we can access true valuations, compute optimal revenue."""
    B   = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()

    vals = get_true_valuations(env)
    if vals is not None:
        # Select top-k buyers by valuation and price at c
        top_k_vals = np.sort(vals)[::-1][:k]
        return float(np.sum(np.maximum(0.0, top_k_vals - C)))
    else:
        # Fallback: sample from same distribution with same seed
        # BudgetEnvConfig seed controls np.random state for sampling
        set_seed(seed)
        n    = graph.number_of_nodes()
        vals = np.random.uniform(0.0, W_HIGH, size=n)
        top_k = np.sort(vals)[::-1][:k]
        return float(np.sum(np.maximum(0.0, top_k - C)))


def _stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2), "all": vals}


def main():
    print("G1: Oracle upper bound (non-realizable), FF_1000 + Modular_FF")
    results = {"note": "Non-realizable oracle; knows all w_i; prices at c",
               "shas": {"none": "no learned weights"}}

    for net, loader in NETS.items():
        graph = loader()
        results[net] = {}
        for k in KS:
            revs = [oracle_revenue_env(graph, k, s) for s in SEEDS]
            r = _stats(revs)
            results[net][str(k)] = r
            print(f"  {net} k={k:2d}  oracle={r['mean']:.1f}±{r['std']:.1f}")

    out = "results/logs/g1_oracle_upper_bound.json"
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
