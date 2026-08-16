#!/usr/bin/env python3
"""
experiments/significance_tests_a2v2.py — A2 (corrected)
=========================================================
Paired significance tests, budget protocol, 10 seeds.

BASELINES  = IE+Budget, Greedy+Budget
OUR METHODS = Rev-GNN-LSTM (sha 0b549f93), Cal-DP

For each (network, k) reports four tables:

  (i)   best-of-ours  vs best-of-baselines
  (ii)  Rev-GNN-LSTM  vs best-of-baselines
  (iii) Cal-DP        vs best-of-baselines

  (iv)  INTERNAL COMPARISON: Rev-GNN-LSTM vs Cal-DP
        (deployment note — NOT a claim of superiority over baselines)

Test: paired two-sided t-test, n=10. 95% CI: mean ± t_{0.025,9} × se.
Cells flagged where p >= 0.05.

Reads:  results/logs/budget_10s_{NET}.json
Writes: results/logs/significance_a2v2.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
from scipy import stats

NETS    = ["FF_1000", "Modular_FF", "Rice_FB", "polblogs", "FF_2000"]
KS      = [5, 10, 15, 20, 30, 40]
IN_TMPL = "results/logs/budget_10s_{net}.json"
OUT     = "results/logs/significance_a2v2.json"
T_CRIT  = stats.t.ppf(0.975, df=9)   # 2.262


def load_net(net: str) -> dict:
    p = IN_TMPL.format(net=net)
    if not os.path.exists(p):
        return {}
    d = json.load(open(p))
    return d.get("results", d)


def _pair_test(a_seeds, b_seeds):
    a, b   = np.array(a_seeds, dtype=float), np.array(b_seeds, dtype=float)
    diff   = a - b
    mean_d = float(diff.mean())
    se     = float(diff.std(ddof=1) / np.sqrt(len(diff)))
    _, p   = stats.ttest_rel(a, b)
    return {
        "mean_diff": round(mean_d, 3),
        "ci_lo":     round(mean_d - T_CRIT * se, 3),
        "ci_hi":     round(mean_d + T_CRIT * se, 3),
        "p_value":   round(float(p), 4),
        "sig":       bool(p < 0.05),
        "a_mean":    round(float(a.mean()), 2),
        "b_mean":    round(float(b.mean()), 2),
    }


def _row(net, k, label, r):
    flag = "" if r["sig"] else "  ← NOT SIG"
    print(f"  {net:12s} k={k:2d}  {r['a_mean']:7.1f} vs {r['b_mean']:7.1f}"
          f"  diff={r['mean_diff']:+7.1f}  [{r['ci_lo']:+7.1f},{r['ci_hi']:+7.1f}]"
          f"  p={r['p_value']:.4f}{flag}")


def main():
    results   = {"test": "paired_t_two_sided_n10", "t_crit_df9": round(T_CRIT, 4)}
    sections  = {
        "i_best_ours_vs_best_base":  {},
        "ii_gnn_vs_best_base":       {},
        "iii_caldp_vs_best_base":    {},
        "iv_gnn_vs_caldp_internal":  {},
    }
    totals    = {k: [0, 0] for k in sections}   # [sig, not_sig]

    for sec_key in sections:
        label = {
            "i_best_ours_vs_best_base": "(i)  BEST-OF-OURS vs BEST-OF-BASELINES",
            "ii_gnn_vs_best_base":      "(ii) Rev-GNN-LSTM vs BEST-OF-BASELINES",
            "iii_caldp_vs_best_base":   "(iii)Cal-DP vs BEST-OF-BASELINES",
            "iv_gnn_vs_caldp_internal": "(iv) INTERNAL: Rev-GNN-LSTM vs Cal-DP [not a baseline claim]",
        }[sec_key]
        print(f"\n{'='*72}\n{label}\n{'='*72}")
        print(f"  {'net':12s} {'k':>4s}  {'A':>7s} vs {'B':>7s}"
              f"  {'diff':>8s}  {'95% CI':^17s}  p-val")
        print("  " + "-"*68)

        for net in NETS:
            d = load_net(net)
            if not d:
                print(f"  {net:12s}  *** budget_10s_{net}.json not found ***")
                continue
            sections[sec_key][net] = {}
            for k in KS:
                cell = d.get(str(k))
                if not cell: continue

                gnn_s  = cell.get("Rev-GNN-LSTM", {}).get("all")
                cdp_s  = cell.get("Cal-DP",        {}).get("all")
                ie_s   = cell.get("IE+Budget",     {}).get("all")
                gd_s   = cell.get("Greedy+Budget", {}).get("all")

                if not gnn_s or not cdp_s or not ie_s or not gd_s:
                    continue

                # best-of-ours = higher mean
                best_ours = gnn_s if np.mean(gnn_s) >= np.mean(cdp_s) else cdp_s
                # best-of-baselines = higher mean (IE vs GD)
                best_base = ie_s  if np.mean(ie_s)  >= np.mean(gd_s)  else gd_s

                if sec_key == "i_best_ours_vs_best_base":
                    r = _pair_test(best_ours, best_base)
                elif sec_key == "ii_gnn_vs_best_base":
                    r = _pair_test(gnn_s, best_base)
                elif sec_key == "iii_caldp_vs_best_base":
                    r = _pair_test(cdp_s, best_base)
                elif sec_key == "iv_gnn_vs_caldp_internal":
                    r = _pair_test(gnn_s, cdp_s)

                sections[sec_key][net][str(k)] = r
                _row(net, k, sec_key, r)
                if r["sig"]: totals[sec_key][0] += 1
                else:        totals[sec_key][1] += 1

        s, ns = totals[sec_key]
        print(f"\n  → significant: {s}   NOT significant: {ns}")

    results.update(sections)
    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
