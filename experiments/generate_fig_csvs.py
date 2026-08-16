#!/usr/bin/env python3
"""
experiments/generate_fig_csvs.py — Step 3
==========================================
Emit two CSVs for plotting:
  results/logs/fig1_unconstrained.csv  — unconstrained protocol
  results/logs/fig2_budget.csv         — budget protocol (RAW values)

Reads from existing result JSONs. Does not run any experiments.
Run after all blocks complete.
"""
from __future__ import annotations
import csv, json, os, sys
import numpy as np

NETS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
KS   = [5, 10, 15, 20, 30, 40]

def safe(d, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict): return default
        d = d.get(k, None)
        if d is None: return default
    return d

# ── Fig 1: unconstrained ─────────────────────────────────────────────────────
def emit_fig1():
    out = "results/logs/fig1_unconstrained.csv"
    unc_path = "results/logs/ablation_unc_10seed.json"
    g4_path  = "results/logs/g4_caldp_unconstrained_10seed.json"

    unc = {}
    if os.path.exists(unc_path):
        raw = json.load(open(unc_path))
        unc = raw.get("results", raw)
    g4  = {}
    if os.path.exists(g4_path):
        raw = json.load(open(g4_path))
        g4  = {k: v for k, v in raw.items() if k not in ("shas",)}

    rows = []
    header = ["network", "method", "mean", "std"]
    for net in NETS:
        u = unc.get(net, {})
        g = g4.get(net, {})

        method_map = {
            "IE":              ("IE",            u.get("IE",{})),
            "mu-Discount":     ("mu-Discount",   u.get("mu_discount", u.get("mu-Discount",{}))),
            "Greedy-Discount": ("Greedy-Discount",u.get("GD",{})),
            "Rev-GNN-IM-RL":   ("Rev-GNN-IM-RL", u.get("arm_b_free", u.get("Rev-GNN-IM-RL",{}))),
            "Learned-Policy":  ("Learned-Policy",u.get("arm_b_free", {})),
            "Cal-DP":          ("Cal-DP",        g.get("Cal-DP_unc", {})),
        }
        for label, (_, d) in method_map.items():
            m = d.get("mean") if isinstance(d, dict) else None
            s = d.get("std")  if isinstance(d, dict) else None
            rows.append([net, label, m if m is not None else "", s if s is not None else ""])

    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"fig1: {out}  ({len(rows)} rows)")
    return len(rows)


# ── Fig 2: budget protocol ───────────────────────────────────────────────────
def emit_fig2():
    out    = "results/logs/fig2_budget.csv"
    header = ["network", "k", "method", "mean", "std"]
    rows   = []

    for net in NETS:
        path = f"results/logs/budget_10s_{net}.json"
        if not os.path.exists(path):
            continue
        raw  = json.load(open(path))
        data = raw.get("results", raw)
        for k in KS:
            cell = data.get(str(k), {})
            for meth in ["IE+Budget", "Greedy+Budget", "Cal-DP", "Rev-GNN-LSTM"]:
                d = cell.get(meth, {})
                rows.append([net, k, meth,
                              d.get("mean","") if isinstance(d,dict) else "",
                              d.get("std","")  if isinstance(d,dict) else ""])

    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"fig2: {out}  ({len(rows)} rows)")
    return len(rows)


if __name__ == "__main__":
    n1 = emit_fig1()
    n2 = emit_fig2()
    print(f"Total: fig1={n1} rows, fig2={n2} rows")
