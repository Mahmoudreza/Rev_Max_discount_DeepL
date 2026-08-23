#!/usr/bin/env python3
"""eval_ie_variants.py — Compare IE+Budget (fixed k=30) vs IE+Budget-Aware (k=min(30,kappa)).

All five networks, kappa in {5,10,20,40}, 10 seeds.
Reports profit Pi=R-c|S_T|, |S_T|, revenue side by side.
Writes results/logs/ie_variants_comparison.json
Conclusion: if budget-aware >= fixed at all kappas, adopt it as the IE baseline.

Usage:
  venv/bin/python3 -u experiments/eval_ie_variants.py > /tmp/ie_variants.log 2>&1
"""
from __future__ import annotations
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import networkx as nx

from src.env.graph_generators import (generate_forest_fire,
                                       generate_modular_forest_fire,
                                       load_rice_facebook)
from src.evaluation.ie_budget import ie_strategy_budget, ie_strategy_budget_aware

_ROOT  = str(Path(__file__).parent.parent)
C      = 0.3
KAPPAS = [5, 10, 20, 40]
N_TRIALS = 10
W_HIGH   = 2.0

NETWORKS: dict = {
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
    "Modular_FF": lambda: generate_modular_forest_fire([200,300,500], 0.37, 0.32, 0.01, seed=0),
    "Rice_FB":    load_rice_facebook,
}
try:
    from src.env.graph_generators import load_polblogs
    NETWORKS["polblogs"] = load_polblogs
except ImportError:
    pass


def _get(d, key):
    v = d.get(key, {})
    if isinstance(v, dict):
        return float(v.get("mean", 0.0)), float(v.get("std", 0.0))
    return float(v), 0.0


def _profit_stats(agg):
    rev_m, rev_s = _get(agg, "revenue")
    n_m,   _     = _get(agg, "n_in_S")
    # profit per trial = revenue - c * n_in_S
    revs = agg.get("revenue", {}).get("all", [rev_m]*N_TRIALS)
    sts  = agg.get("n_in_S",  {}).get("all", [n_m]*N_TRIALS)
    profs = [r - C*s for r,s in zip(revs, sts)]
    return float(np.mean(profs)), float(np.std(profs))


def main():
    print(f"=== IE Variants Comparison ===")
    print(f"C={C}  kappas={KAPPAS}  n_trials={N_TRIALS}  W_HIGH={W_HIGH}")
    print(f"IE-fixed   : k_seeds=30 always")
    print(f"IE-aware   : k_seeds=min(30, floor(B0/c)) = min(30, kappa)")
    print(flush=True)

    hdr = (f"{'net':12s} {'k':3s}  "
           f"{'IE-fixed profit':>16s}±{'std':>5s}  {'|S_T|':>5s}  {'rev':>8s}  |  "
           f"{'IE-aware profit':>16s}±{'std':>5s}  {'|S_T|':>5s}  {'rev':>8s}  "
           f"{'k_used':>6s}  {'winner':>7s}")
    print(hdr)
    print("-"*len(hdr))

    results = {}
    aware_wins = 0; fixed_wins = 0; ties = 0
    all_cells = []

    for net_name, gfn in NETWORKS.items():
        try:
            G = gfn()
        except Exception as e:
            print(f"[skip] {net_name}: {e}"); continue
        print(f"\n{net_name} (n={G.number_of_nodes()})", flush=True)
        results[net_name] = {}

        for kappa in KAPPAS:
            B0     = kappa * C
            k_used = min(30, kappa)

            try:
                agg_f = ie_strategy_budget(G, B0, C,
                                            k_seeds=30, n_trials=N_TRIALS,
                                            weight_high=W_HIGH)
            except Exception as e:
                print(f"  ERR fixed k={kappa}: {e}"); continue

            try:
                agg_a = ie_strategy_budget_aware(G, B0, C,
                                                  n_trials=N_TRIALS,
                                                  weight_high=W_HIGH)
            except Exception as e:
                print(f"  ERR aware k={kappa}: {e}"); continue

            pf_m, pf_s = _profit_stats(agg_f)
            pa_m, pa_s = _profit_stats(agg_a)
            rev_f, _   = _get(agg_f, "revenue")
            rev_a, _   = _get(agg_a, "revenue")
            st_f,  _   = _get(agg_f, "n_in_S")
            st_a,  _   = _get(agg_a, "n_in_S")

            if pa_m > pf_m + 0.01:
                winner = "AWARE"; aware_wins += 1
            elif pf_m > pa_m + 0.01:
                winner = "FIXED"; fixed_wins += 1
            else:
                winner = "TIE"; ties += 1

            print(f"  k={kappa:2d}  "
                  f"fixed: {pf_m:7.2f}±{pf_s:5.2f}  {st_f:5.1f}  {rev_f:8.2f}  |  "
                  f"aware: {pa_m:7.2f}±{pa_s:5.2f}  {st_a:5.1f}  {rev_a:8.2f}  "
                  f"k_used={k_used:2d}  {winner}", flush=True)

            results[net_name][f"k{kappa}"] = {
                "fixed": {"profit":[pf_m,pf_s], "rev":rev_f, "st":st_f},
                "aware": {"profit":[pa_m,pa_s], "rev":rev_a, "st":st_a},
                "k_used": k_used, "winner": winner,
            }
            all_cells.append({"net": net_name, "k": kappa,
                               "pf": pf_m, "pa": pa_m, "winner": winner})

    print(f"\n{'='*60}")
    print(f"SUMMARY: aware wins={aware_wins}  fixed wins={fixed_wins}  ties={ties}")
    print(f"Total cells: {len(all_cells)}")
    aware_dominant = (fixed_wins == 0)
    print(f"Budget-aware dominant (fixed wins=0): {aware_dominant}")
    if aware_dominant:
        print("→ ADOPT ie_strategy_budget_aware as the IE baseline.")
    else:
        print("→ Use max(fixed, aware) per cell OR report both.")

    out = os.path.join(_ROOT, "results/logs/ie_variants_comparison.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({
        "summary": {"aware_wins": aware_wins, "fixed_wins": fixed_wins,
                    "ties": ties, "aware_dominant": aware_dominant},
        "cells": all_cells,
        "results": results,
    }, open(out, "w"), indent=2)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
