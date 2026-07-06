"""src/utils/visualization.py — Plotting utilities (CLAUDE.md: save to results/figures/).

All figures are generated here and called from thin experiment scripts.
Never use bare print() — pass a logger if status is needed.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import os
import numpy as np


METHOD_STYLES: Dict[str, dict] = {
    "Greedy+Budget":    {"color": "#e41a1c", "lw": 2.0, "ls": "-",  "marker": "o", "ms": 5},
    "Efficiency-Greedy":{"color": "#ff7f00", "lw": 1.5, "ls": "--", "marker": "s", "ms": 4},
    "Two-Phase-DP":     {"color": "#984ea3", "lw": 1.5, "ls": ":",  "marker": "^", "ms": 4},
    "LSTM-Idea1":       {"color": "#377eb8", "lw": 2.0, "ls": "-",  "marker": "D", "ms": 5},
    "LSTM-Idea3":       {"color": "#4daf4a", "lw": 2.5, "ls": "-",  "marker": "*", "ms": 7},
}

PANEL_LABELS = {
    "strong_U02": r"Link weights $w_{ij}\sim U(0,2)$ (standard)",
    "weak_U01":   r"Link weights $w_{ij}\sim U(0,1)$ (weak influence)",
}


def _get_rev(panel_res: dict, sweep_k: List[int], method: str) -> List[float]:
    vals = []
    for k in sweep_k:
        r = panel_res.get(f"k={k}", {}).get(method, {})
        v = r.get("revenue", {})
        vals.append(v.get("mean", 0.0) if isinstance(v, dict) else float(v or 0))
    return vals


def _get_std(panel_res: dict, sweep_k: List[int], method: str) -> List[float]:
    vals = []
    for k in sweep_k:
        r = panel_res.get(f"k={k}", {}).get(method, {})
        v = r.get("revenue", {})
        vals.append(v.get("std", 0.0) if isinstance(v, dict) else 0.0)
    return vals


def _annotate_knee_crossover(ax, sweep_k, greedy_revs, idea3_revs):
    """Add vertical dashed lines for Greedy's saturation knee and LSTM-Idea3 crossover."""
    # Knee: first k where incremental gain < 5% of peak incremental gain
    deltas = [greedy_revs[i+1] - greedy_revs[i] for i in range(len(greedy_revs)-1)]
    peak_d = max(deltas) if deltas else 1.0
    for i, d in enumerate(deltas):
        if d < 0.05 * peak_d:
            knee_k = sweep_k[i+1]
            ax.axvline(x=knee_k, color="gray", lw=1.0, ls="--", alpha=0.6)
            ax.text(knee_k * 1.05, ax.get_ylim()[0] + 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    f"Greedy\nknee\n$k={knee_k}$", fontsize=7, color="gray", va="bottom")
            break
    # Crossover: first k where Greedy > LSTM-Idea3
    for k, g, l3 in zip(sweep_k, greedy_revs, idea3_revs):
        if g > l3 and l3 > 0:
            ax.axvline(x=k, color="#4daf4a", lw=1.0, ls=":", alpha=0.8)
            ax.text(k * 1.05, ax.get_ylim()[1] * 0.85,
                    f"crossover\n$k={k}$", fontsize=7, color="#4daf4a", va="top")
            break


def plot_budget_revenue_panels(
    results_strong: dict,
    results_weak: dict,
    sweep_k: List[int],
    c: float,
    out_path: str,
    logger=None,
    methods: Optional[List[str]] = None,
) -> None:
    """Generate fig_b1_v2: two-panel Revenue vs k figure.

    Panel (a): U(0,2) standard weights.
    Panel (b): U(0,1) weak-influence robustness.

    X-axis: k = B/c (log-ish ticks at sweep values).
    Y-axis: Revenue (mean ± std shaded).
    Annotations: Greedy's saturation knee, LSTM-Idea3 vs Greedy crossover.

    Args:
        results_strong: run_budget_comparison results for U(0,2), keyed by "k=...".
        results_weak:   run_budget_comparison results for U(0,1), keyed by "k=...".
        sweep_k:        List of integer k values used in the sweep.
        c:              Production cost (used for x-axis label).
        out_path:       Save path (.pdf or .png).
        logger:         Optional ExperimentLogger for status messages.
        methods:        Methods to plot (default: METHOD_STYLES keys).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    if methods is None:
        methods = list(METHOD_STYLES.keys())

    panels = [results_strong, results_weak]
    panel_labels = [
        r"(a) $w_{ij}\sim U(0,2)$ — standard",
        r"(b) $w_{ij}\sim U(0,1)$ — weak influence",
    ]

    x = np.array(sweep_k, dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    fig.subplots_adjust(wspace=0.35)

    for ax, res, plabel in zip(axes, panels, panel_labels):
        for m in methods:
            if m not in METHOD_STYLES:
                continue
            revs = _get_rev(res, sweep_k, m)
            stds = _get_std(res, sweep_k, m)
            if not any(v > 0 for v in revs):
                continue
            sty = METHOD_STYLES[m]
            revs_arr = np.array(revs)
            stds_arr = np.array(stds)
            ax.plot(x, revs_arr, label=m, color=sty["color"],
                    lw=sty["lw"], ls=sty["ls"],
                    marker=sty["marker"], markersize=sty["ms"])
            ax.fill_between(x, revs_arr - stds_arr, revs_arr + stds_arr,
                            alpha=0.12, color=sty["color"])

        # Annotations (after plotting, so ylim is set)
        ax.set_xscale("log")
        ax.set_xticks(sweep_k)
        ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
        ax.set_xlabel(r"Budget in production-cost units,  $k = B/c$", fontsize=10)
        ax.set_ylabel("Total Revenue", fontsize=10)
        ax.set_title(plabel, fontsize=11)
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.8)

        greedy_revs = _get_rev(res, sweep_k, "Greedy+Budget")
        idea3_revs  = _get_rev(res, sweep_k, "LSTM-Idea3")
        if any(v > 0 for v in idea3_revs):
            _annotate_knee_crossover(ax, sweep_k, greedy_revs, idea3_revs)

    fig.suptitle(
        rf"Budget-Constrained Revenue vs k  ($c={c}$, Forest-Fire $n=1000$)",
        fontsize=12, y=1.01,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    if logger:
        logger.info(f"  fig_b1_v2 saved → {out_path}")
