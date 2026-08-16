#!/usr/bin/env python3
"""
c1_core/experiments/plot_paper_gen_comparison.py
=================================================
Bar-chart comparison of all methods across 5 networks.
Uses paper_gen_updated.json — no GPU / no re-runs needed.

Also draws a Babaei-style line figure: X = network size n (FF graphs),
Y = revenue, one line per method.

Outputs:
  results/figures/c1_bar_comparison.pdf / .png
  results/figures/c1_revenue_vs_n.pdf   / .png

Run locally:
  python c1_core/experiments/plot_paper_gen_comparison.py
"""
import json, os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA  = "results/logs/paper_gen_updated.json"
OUTD  = "results/figures"

# ── colour / style map (paper-consistent) ────────────────────────────────────
METHOD_STYLE = {
    "IE-Strategy":     dict(color="#d62728", ls=":", marker="s",  lw=1.6, label="IE-Strategy"),
    "Greedy-Discount": dict(color="#2ca02c", ls="-.", marker="^", lw=1.8, label="Greedy-Discount"),
    "Rev-GNN-IM-RL":   dict(color="#ff7f0e", ls="--", marker="o", lw=1.8, label="Rev-GNN-IM-RL"),
    "Rev-GNN-LSTM":    dict(color="#1f77b4", ls="-",  marker="D", lw=2.2, label="Rev-GNN-LSTM"),
}

os.makedirs(OUTD, exist_ok=True)
d = json.load(open(DATA))


# ── Figure 1: Grouped bar chart (all 5 networks) ──────────────────────────────
networks = [n for n in d.keys() if n != "FF n=500"]
methods  = list(METHOD_STYLE.keys())
x        = np.arange(len(networks))
w        = 0.18
offsets  = np.linspace(-1.5 * w, 1.5 * w, 4)

fig1, ax1 = plt.subplots(figsize=(10, 5))

for i, m in enumerate(methods):
    vals = [d[net][m]["mean"] for net in networks]
    st   = METHOD_STYLE[m]
    ax1.bar(x + offsets[i], vals, width=w, color=st["color"],
            label=st["label"], alpha=0.88, edgecolor="white", linewidth=0.5)

short_labels = ["FF-1000", "FF-2000", "Modular-FF", "Rice-FB"]
ax1.set_xticks(x)
ax1.set_xticklabels(short_labels, fontsize=10)
ax1.set_ylabel("Revenue", fontsize=12)
ax1.set_title("Revenue Comparison Across Networks", fontsize=12)
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(axis="y", alpha=0.3)
ax1.set_ylim(bottom=0)

for ext in ("pdf", "png"):
    p = f"{OUTD}/c1_bar_comparison.{ext}"
    fig1.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
plt.close(fig1)


# ── Figure 2: Babaei-style line — X = n, Y = revenue (FF only) ───────────────
# X axis: graph size n (Babaei uses "number of initial seed nodes k"; we use n
# because in our unconstrained setting the graph size determines the market)
ff_nets   = ["FF n=500", "FF n=1000", "FF n=2000"]
ff_n      = [500, 1000, 2000]
ff_labels = ["FF-500\n(n=500)", "FF-1000\n(n=1000)", "FF-2000\n(n=2000)"]

fig2, ax2 = plt.subplots(figsize=(6.5, 4.5))

for m in methods:
    vals = [d[net][m]["mean"] for net in ff_nets]
    st   = METHOD_STYLE[m]
    ax2.plot(ff_n, vals,
             color=st["color"], ls=st["ls"], marker=st["marker"],
             lw=st["lw"], ms=7, label=st["label"])

# Annotate percentage improvement of LSTM over GD at each n
for n, net in zip(ff_n, ff_nets):
    lstm = d[net]["Rev-GNN-LSTM"]["mean"]
    gd   = d[net]["Greedy-Discount"]["mean"]
    pct  = 100 * (lstm - gd) / gd
    ax2.annotate(f"+{pct:.0f}%",
                 xy=(n, lstm), xytext=(0, 6), textcoords="offset points",
                 ha="center", fontsize=7.5, color="#1f77b4", fontweight="bold")

ax2.set_xlabel("Market Size $n$  (Forest Fire graph)", fontsize=12)
ax2.set_ylabel("Revenue", fontsize=12)
ax2.set_title("Revenue vs. Market Size  (Babaei et al. style)", fontsize=12)
ax2.set_xticks(ff_n)
ax2.set_xticklabels(ff_labels, fontsize=9)
ax2.legend(fontsize=8.5, loc="upper left")
ax2.grid(alpha=0.3)
ax2.set_xlim(300, 2200)
ax2.set_ylim(bottom=0)

for ext in ("pdf", "png"):
    p = f"{OUTD}/c1_revenue_vs_n.{ext}"
    fig2.savefig(p, dpi=150, bbox_inches="tight")
    print(f"Saved → {p}")
plt.close(fig2)

# ── Print table ───────────────────────────────────────────────────────────────
print("\n── paper_gen_updated.json — Revenue Table ──")
hdr = f"{'Network':<18}" + "".join(f"{m[:12]:>13}" for m in methods)
print(hdr)
print("-" * len(hdr))
for net in networks:
    vals = [d[net][m]["mean"] for m in methods]
    row  = f"{net:<18}" + "".join(f"{v:>13.1f}" for v in vals)
    print(row)
