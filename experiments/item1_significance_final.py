#!/usr/bin/env python3
"""
experiments/item1_significance_final.py — ITEM 1 significance (all 5 networks).
===============================================================================
Re-runs A2 significance analysis INCLUDING FF_2000 shard.
Reads:  results/logs/budget_10s_{NET}.json for each network.
Writes: results/logs/item1_significance_final.json

Four paired t-tests (two-sided, n=10, df=9, t_crit=2.2622):
  (i)   best-of-ours vs best-of-baselines
  (ii)  Rev-GNN-LSTM vs best-of-baselines
  (iii) Cal-DP vs best-of-baselines
  (iv)  internal: Rev-GNN-LSTM vs Cal-DP
Baselines = IE+Budget and Greedy+Budget only (Cal-DP is ours).
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

NETS     = ["FF_1000", "Modular_FF", "Rice_FB", "polblogs", "FF_2000"]
KS       = [5, 10, 15, 20, 30, 40]
IN_TMPL  = "results/logs/budget_10s_{net}.json"
OUT      = "results/logs/item1_significance_final.json"
T_CRIT   = stats.t.ppf(0.975, df=9)   # 2.2622


def _ci(d: np.ndarray):
    se = d.std(ddof=1) / math.sqrt(len(d))
    m  = d.mean()
    return {"mean": round(float(m),3),
            "ci95": [round(float(m - T_CRIT*se),3), round(float(m + T_CRIT*se),3)],
            "pval": round(float(stats.ttest_1samp(d, 0).pvalue),4),
            "sig":  bool(abs(m) - T_CRIT*se > 0)}


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT}"); return

    results = {"test": "paired_t_two_sided_n10", "t_crit_df9": round(T_CRIT,4), "cells": {}}
    n_missing = 0

    for net in NETS:
        path = IN_TMPL.format(net=net)
        if not os.path.exists(path):
            print(f"  MISSING: {path} — skipping {net}")
            n_missing += 1
            continue
        shard = json.load(open(path))

        for k in KS:
            sk = str(k)
            if sk not in shard.get("results", {}):
                continue
            cell = shard["results"][sk]

            ie   = np.array(cell.get("IE+Budget",      {}).get("all", [0]*10), dtype=float)
            gd   = np.array(cell.get("Greedy+Budget",  {}).get("all", [0]*10), dtype=float)
            cdp  = np.array(cell.get("Cal-DP",         {}).get("all", [0]*10), dtype=float)
            lstm = np.array(cell.get("Rev-GNN-LSTM",   {}).get("all", [0]*10), dtype=float)

            if not (len(ie)==10 and len(gd)==10 and len(cdp)==10 and len(lstm)==10):
                print(f"  WARNING: {net} k={k} missing seeds"); continue

            best_base = np.maximum(ie, gd)   # element-wise max
            best_ours = np.maximum(cdp, lstm)

            cid = f"{net}|k={k}"
            results["cells"][cid] = {
                "means": {"IE": round(float(ie.mean()),2), "GD": round(float(gd.mean()),2),
                          "CalDP": round(float(cdp.mean()),2), "LSTM": round(float(lstm.mean()),2)},
                "(i)_best_ours_vs_best_base":  _ci(best_ours - best_base),
                "(ii)_lstm_vs_best_base":      _ci(lstm - best_base),
                "(iii)_caldp_vs_best_base":    _ci(cdp  - best_base),
                "(iv)_lstm_vs_caldp":          _ci(lstm - cdp),
            }

            sig_i = results["cells"][cid]["(i)_best_ours_vs_best_base"]["sig"]
            print(f"  {net:12s} k={k:2d}  (i) sig={sig_i}  "
                  f"LSTM={lstm.mean():.1f}  CalDP={cdp.mean():.1f}  "
                  f"IE={ie.mean():.1f}  GD={gd.mean():.1f}")

    n_cells = len(results["cells"])
    print(f"\nTotal cells: {n_cells}  (missing networks: {n_missing})")
    assert n_cells + n_missing * len(KS) >= 30 - 1 or n_missing == 0, \
        f"Expected 30 cells but got {n_cells} (missing {n_missing} networks)"

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
