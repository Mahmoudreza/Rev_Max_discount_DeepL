#!/usr/bin/env python3
"""
experiments/significance_tests_a2.py — A2
==========================================
Paired significance tests: for each (network, k), compare the better of
our two methods (arm_b) against the better of the two baselines
(Greedy+Budget vs Cal-DP) across 10 shared seeds.

Test used: paired two-sided t-test (scipy.stats.ttest_rel).
95% CI: mean_diff ± t_{0.025,9} * se.
Flags every cell where p >= 0.05 (NOT significant).

Reads: results/logs/budget_10s_{NET}.json  (produced by budget_sweep_10seed.py)
Writes: results/logs/significance_a2.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats

NETS = ["FF_1000", "Modular_FF", "Rice_FB", "polblogs", "FF_2000"]
KS   = [5, 10, 15, 20, 30, 40]
IN_TMPL = "results/logs/budget_10s_{net}.json"
OUT     = "results/logs/significance_a2.json"

T_CRIT = stats.t.ppf(0.975, df=9)   # 2.262 for n=10


def load_per_seed(net: str) -> dict:
    path = IN_TMPL.format(net=net)
    if not os.path.exists(path):
        return {}
    d = json.load(open(path))
    # Layout: d["results"][str(k)] = {"IE+Budget":{...}, "Greedy+Budget":{...},
    #                                  "Cal-DP":{...}, "Rev-GNN-LSTM":{...}}
    # Per-seed array key: "all"
    return d.get("results", d)   # fall back to d itself if no "results" wrapper


def test_one(ours_seeds, baseline_seeds):
    """Paired t-test. Returns dict with mean_diff, ci_lo, ci_hi, p, significant."""
    a = np.array(ours_seeds, dtype=float)
    b = np.array(baseline_seeds, dtype=float)
    diff = a - b
    mean_d = float(np.mean(diff))
    se = float(np.std(diff, ddof=1) / np.sqrt(len(diff)))
    t_stat, p = stats.ttest_rel(a, b)
    ci_lo = mean_d - T_CRIT * se
    ci_hi = mean_d + T_CRIT * se
    return {
        "mean_diff": round(mean_d, 3),
        "ci_lo": round(ci_lo, 3),
        "ci_hi": round(ci_hi, 3),
        "t_stat": round(float(t_stat), 3),
        "p_value": round(float(p), 4),
        "significant_005": bool(p < 0.05),
    }


def main():
    results = {}
    n_cells = n_sig = n_insig = 0
    print(f"{'net':12s} {'k':>4s}  {'ours':>7s}  {'best_base':>9s}  "
          f"{'diff':>7s}  {'95%CI':^17s}  {'p':>6s}  sig?")
    print("-" * 80)

    for net in NETS:
        d = load_per_seed(net)
        if not d:
            print(f"{net:12s}  *** JSON not found: {IN_TMPL.format(net=net)} ***")
            continue
        results[net] = {}
        for k in KS:
            sk = str(k)
            if sk not in d:
                continue
            cell = d[sk]
            # Exact keys written by budget_sweep_10seed.py; "all" holds per-seed list
            ours_key = next((x for x in ["Rev-GNN-LSTM","GNN_arm_b","GNN"] if x in cell), None)
            gd_key   = next((x for x in ["Greedy+Budget","GD","Greedy"] if x in cell), None)
            cdp_key  = next((x for x in ["Cal-DP","CDP","CalDP"] if x in cell), None)

            if ours_key is None or (gd_key is None and cdp_key is None):
                print(f"{net:12s}  k={k:2d}  missing keys: {list(cell.keys())}")
                continue

            ours_s = cell[ours_key].get("all")
            gd_s   = cell[gd_key].get("all")  if gd_key  else None
            cdp_s  = cell[cdp_key].get("all") if cdp_key else None

            if ours_s is None:
                print(f"{net:12s}  k={k:2d}  no 'all' per-seed data for {ours_key}")
                continue

            # Best baseline = higher mean
            gd_mean  = float(np.mean(gd_s))  if gd_s  else -1e9
            cdp_mean = float(np.mean(cdp_s)) if cdp_s else -1e9
            if cdp_mean >= gd_mean and cdp_s:
                best_s, best_name, best_mean = cdp_s, "Cal-DP", cdp_mean
            else:
                best_s, best_name, best_mean = gd_s, "GD", gd_mean

            ours_mean = float(np.mean(ours_s))
            r = test_one(ours_s, best_s)
            r["ours_mean"]  = round(ours_mean, 2)
            r["best_baseline"] = best_name
            r["best_mean"]  = round(best_mean, 2)
            results[net][sk] = r
            n_cells += 1
            if r["significant_005"]: n_sig += 1
            else: n_insig += 1

            flag = "" if r["significant_005"] else "  ← NOT SIG"
            print(f"{net:12s}  k={k:2d}  {ours_mean:7.1f}  {best_mean:9.1f}  "
                  f"{r['mean_diff']:+7.1f}  [{r['ci_lo']:+7.1f},{r['ci_hi']:+7.1f}]  "
                  f"{r['p_value']:6.4f}{flag}")

    print(f"\nTotal cells: {n_cells}  Significant: {n_sig}  NOT significant: {n_insig}")
    os.makedirs("results/logs", exist_ok=True)
    out = {"results": results, "test": "paired_t_two_sided_n10",
           "alpha": 0.05, "t_crit_df9": round(T_CRIT, 4),
           "n_cells": n_cells, "n_significant": n_sig, "n_not_significant": n_insig}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
