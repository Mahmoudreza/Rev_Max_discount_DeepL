#!/usr/bin/env python3
"""
experiments/merge_budget_10seed.py  — Block A (A1 + A2)
========================================================
Merges budget_10s_{NET}.json shards → results/logs/budget_sweep_10seed.json
Then runs paired tests (A2): Rev-GNN-LSTM vs best baseline per (network, k).
A3 note: training seeds for FF+BA (sha 0b549f93) = 1 (single run, n=1).
"""
import json, os, sys, math
import numpy as np

NETS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
KVALS = [5, 10, 15, 20, 30, 40]
METHODS = ["IE+Budget", "Greedy+Budget", "Cal-DP", "Rev-GNN-LSTM"]
INDIR = "results/logs"
OUT   = "results/logs/budget_sweep_10seed.json"


def _t_test_paired(a, b):
    """Paired t-test; returns (t, p, mean_diff, ci95_low, ci95_high)."""
    diffs = np.array(a) - np.array(b)
    n = len(diffs)
    md = diffs.mean(); sd = diffs.std(ddof=1)
    t  = md / (sd / math.sqrt(n)) if sd > 0 else float("inf")
    # Normal approx for p (n=10 adequate)
    from math import erf, sqrt
    z = abs(t)
    p = 2 * (1 - 0.5 * (1 + erf(z / sqrt(2))))
    sem = sd / math.sqrt(n)
    ci_lo = md - 1.96 * sem; ci_hi = md + 1.96 * sem
    return float(t), float(p), float(md), float(ci_lo), float(ci_hi)


def main():
    merged = {}
    missing = []
    for net in NETS:
        path = os.path.join(INDIR, f"budget_10s_{net}.json")
        if not os.path.exists(path):
            missing.append(net); continue
        shard = json.load(open(path))
        merged[net] = shard

    if missing:
        print(f"WARNING: missing shards for: {missing}")

    # Save merged
    with open(OUT, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged → {OUT}")

    # ── A1: print table ───────────────────────────────────────────────────────
    print("\n── A1: Budget Sweep 10-seed means ± std ──")
    for net in NETS:
        if net not in merged: continue
        print(f"\n{net}")
        hdr = f"  {'k':>3}  " + "".join(f"  {m[:10]:>12}±std" for m in METHODS)
        print(hdr)
        r = merged[net]["results"]
        for k in KVALS:
            row = f"  {k:>3}  "
            for m in METHODS:
                mean = r[str(k)][m]["mean"]; std = r[str(k)][m]["std"]
                row += f"  {mean:>8.1f}±{std:<4.1f}"
            print(row)

    # ── A2: paired tests ──────────────────────────────────────────────────────
    print("\n── A2: Paired t-tests Rev-GNN-LSTM vs best baseline ──")
    print("Test: paired t-test (n=10 matched weight seeds; same stochastic env realization per seed).")
    print(f"{'net':12s}  {'k':>3}  {'vs':>14}  {'mean_diff':>9}  {'95%CI':>18}  {'p':>7}  sig")
    for net in NETS:
        if net not in merged: continue
        r = merged[net]["results"]
        for k in KVALS:
            gnn = r[str(k)]["Rev-GNN-LSTM"]["all"]
            best_m, best_vals = None, None
            best_mean = -1e9
            for m in ["Greedy+Budget", "Cal-DP"]:
                vals = r[str(k)][m]["all"]
                mn = float(np.mean(vals))
                if mn > best_mean:
                    best_mean = mn; best_m = m; best_vals = vals
            t, p, md, ci_lo, ci_hi = _t_test_paired(gnn, best_vals)
            sig = "**" if p < 0.01 else ("*" if p < 0.05 else "ns")
            print(f"  {net:12s}  {k:>3}  {best_m[:14]:>14}  {md:>+9.1f}  "
                  f"[{ci_lo:+.1f},{ci_hi:+.1f}]  {p:>7.4f}  {sig}")

    # ── A3: training seeds ────────────────────────────────────────────────────
    print("\n── A3: Training seeds for FF+BA (sha 0b549f93) ──")
    print("  n=1 independently trained policy for this configuration.")
    print("  Other checkpoints (c1_ffba_2to1, c1_ffba_50_50) differ in FF:BA data ratio, not training seed.")
    print("  Disclosure: all reported results use a single trained policy (n=1 training seed).")


if __name__ == "__main__":
    main()
