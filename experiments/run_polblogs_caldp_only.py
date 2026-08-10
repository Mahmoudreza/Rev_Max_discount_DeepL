#!/usr/bin/env python3
"""run_polblogs_caldp_only.py — Re-run Cal-DP v2/v3/composite for polblogs only.
Merges caldp_v2, caldp_v3, caldp (composite) back into polblogs_budget_sweep.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
_orig_bc = nx.betweenness_centrality
def _approx_bc(G, normalized=True, **kwargs):
    k_pivots = min(200, G.number_of_nodes())
    return _orig_bc(G, k=k_pivots, normalized=normalized, **kwargs)
nx.betweenness_centrality = _approx_bc

from src.env.polblogs_loader import load_polblogs
from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.revenue_env import RevenueEnv
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget

def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.0
    tw = sum(self._link_weights.get((node,n),0.0) for n in nb)
    if tw==0: return 0.0
    return sum(self._link_weights.get((node,n),0.0) for n in nb if n in self.S)/tw
RevenueEnv.get_current_influence = _fast_gci

C = 0.3
N_TRIALS = 3
WEIGHT_HIGH = 2.0
K_LIST = [1, 3, 5, 10, 15, 20, 30, 40]
OUT_JSON = "results/logs/polblogs_budget_sweep.json"

def eval_caldp_both(graph, k):
    B = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0, weight_high=WEIGHT_HIGH)
    try:
        r2 = dp_calibrated_v2_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        m2 = float(r2.get("revenue", {}).get("mean", r2.get("mean", 0.0)))
    except Exception as e:
        print(f"  v2 k={k} error: {e}"); m2 = 0.0
    try:
        r3 = dp_calibrated_v3_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        m3 = float(r3.get("revenue", {}).get("mean", r3.get("mean", 0.0)))
    except Exception as e:
        print(f"  v3 k={k} error: {e}"); m3 = 0.0
    comp = max(m2, m3)
    ver = "v3" if m3 >= m2 else "v2"
    return m2, m3, comp, ver

def main():
    t0 = time.time()
    graph = load_polblogs()
    print(f"Polblogs n={graph.number_of_nodes()} m={graph.number_of_edges()}")

    # Load existing JSON
    existing = {}
    if os.path.exists(OUT_JSON):
        with open(OUT_JSON) as f:
            existing = json.load(f)
    results = existing.get("results", {})

    print(f"{'k':>3} | {'v2':>8} | {'v3':>8} | {'comp':>8} | {'ver':>4}")
    print("-" * 44)
    for k in K_LIST:
        v2, v3, comp, ver = eval_caldp_both(graph, k)
        sk = str(k)
        if sk not in results:
            results[sk] = {}
        results[sk]["caldp_v2"] = v2
        results[sk]["caldp_v3"] = v3
        results[sk]["caldp"]    = comp
        results[sk]["caldp_ver"]= ver
        print(f"{k:>3} | {v2:>8.1f} | {v3:>8.1f} | {comp:>8.1f} | {ver:>4}")

    existing["results"] = results
    with open(OUT_JSON, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\nMerged into {OUT_JSON}. Wall: {(time.time()-t0)/60:.1f} min")

if __name__ == "__main__":
    main()
