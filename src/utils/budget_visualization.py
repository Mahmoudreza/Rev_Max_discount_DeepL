"""src/utils/budget_visualization.py — Budget-constrained figures (Idea 3).

DO NOT modify visualization.py, tc_visualization.py, or paper_figures.py.
All Idea 3 figures are created here.

Figures:
  B-1: Revenue vs Budget B (key result)
  B-2: Budget trajectory over episode
  B-3: Revenue vs K at multiple budgets (2×2 multi-panel)
  B-4: Sensitivity to production cost c
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Shared style ──────────────────────────────────────────────────────────────

METHOD_STYLES: Dict[str, dict] = {
    "Greedy+Budget":     {"color": "#d62728", "lw": 2.0, "ls": "-"},
    "Efficiency-Greedy": {"color": "#2ca02c", "lw": 1.5, "ls": "--"},
    "Two-Phase-DP":      {"color": "#ff7f0e", "lw": 1.5, "ls": "-."},
    "LSTM-Idea1":        {"color": "#7f7f7f", "lw": 1.5, "ls": ":"},
    "IM-RL-Idea1":       {"color": "#9467bd", "lw": 1.5, "ls": ":"},
    "LSTM-Budget":       {"color": "#000000", "lw": 2.5, "ls": "-",
                          "marker": "s", "markersize": 4},
}
METHOD_ORDER = ["Greedy+Budget", "Efficiency-Greedy", "Two-Phase-DP",
                "LSTM-Idea1", "LSTM-Budget"]

UNCONSTRAINED_REV = 448.6   # Idea 1 LSTM at B=∞

PLT_CFG = {
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
}


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [✓] {path}")


# ── Figure B-1: Revenue vs Budget B ──────────────────────────────────────────

def plot_revenue_vs_budget(
    results_by_B: Dict[str, Dict],
    production_cost: float,
    save_path: str,
) -> None:
    """Revenue at different budget levels (THE key Idea 3 figure).

    x-axis: "Initial Budget (B)"
    y-axis: "Total Revenue"

    Shows: at B=100 all methods converge to Idea 1 performance.
           at B=5 LSTM-Budget >> DP > Efficiency > Greedy+B >> LSTM-Idea1.
           LSTM-Idea1 (gray dotted) crashes immediately = paper motivation.
    Horizontal dashed line: Idea 1 unconstrained revenue (448.6).

    Args:
        results_by_B: {B_val: {method_name: result_dict}} from run_budget_eval.
        production_cost: c value (for title).
        save_path: Output file path.
    """
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect B values and ensure numeric sort
    B_keys    = sorted(results_by_B.keys(), key=lambda k: float(k.replace("B=", "")))
    B_vals    = [float(k.replace("B=", "")) for k in B_keys]
    x_pos     = list(range(len(B_vals)))
    x_labels  = [f"B={B:.0f}" for B in B_vals]

    # Unconstrained reference line
    ax.axhline(UNCONSTRAINED_REV, color="gray", lw=1.0, ls="--", alpha=0.5, zorder=1)
    ax.text(x_pos[-1] + 0.1, UNCONSTRAINED_REV + 3, r"Unconstrained ($B{=}\infty$)",
            fontsize=9, color="gray", va="bottom")

    for method in METHOD_ORDER:
        means, stds = [], []
        for key in B_keys:
            res = results_by_B.get(key, {}).get(method, {})
            rev = res.get("revenue", {})
            if isinstance(rev, dict):
                m, s = rev.get("mean", 0.0), rev.get("std", 0.0)
            else:
                m, s = float(rev or 0), 0.0
            means.append(m)
            stds.append(s)

        if not any(m > 0 for m in means):
            continue

        st   = METHOD_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})
        m_arr = np.array(means)
        s_arr = np.array(stds)
        mkw   = {}
        if "marker" in st:
            mkw = {"marker": st["marker"], "markersize": st["markersize"], "markevery": 1}

        ax.fill_between(x_pos, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.08, color=st["color"])
        ax.plot(x_pos, m_arr, color=st["color"], lw=st["lw"], ls=st["ls"],
                label=method, **mkw)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels, rotation=15)
    ax.set_xlabel("Initial Budget (B)")
    ax.set_ylabel("Total Revenue")
    ax.set_title(f"Revenue vs Budget (production cost c={production_cost})")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


# ── Figure B-2: Budget trajectory over episode ───────────────────────────────

def plot_budget_trajectory(
    trajectories: Dict[str, List[List[float]]],
    B: float,
    c: float,
    save_path: str,
) -> None:
    """How each method's budget evolves over the episode.

    x-axis: "Offer step"
    y-axis: "Remaining Budget B_t"

    Expected:
      Greedy+B:    drops first (free seeds), then slowly recovers
      DP:          planned spending, eventually self-funding
      LSTM-Idea1:  drops to 0 immediately (bankruptcy at ~30 steps)
      LSTM-Budget: careful early, then self-funding growth

    Horizontal dashed lines: y=0 (bankruptcy), y=c (min for free offers).

    Args:
        trajectories: method_name → list of budget_history lists.
        B:            Initial budget (for title).
        c:            Production cost (for threshold line).
        save_path:    Output path.
    """
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.axhline(0, color="red",  lw=1.0, ls="--", alpha=0.7, label="_nolegend_")
    ax.axhline(c, color="gray", lw=0.8, ls=":",  alpha=0.5, label="_nolegend_")
    ax.text(2, 0.02,  "Bankruptcy",      fontsize=8, color="red",  alpha=0.7)
    ax.text(2, c+0.02, f"Min for free (c={c})", fontsize=8, color="gray", alpha=0.7)

    for method in METHOD_ORDER:
        traj_list = trajectories.get(method, [])
        if not traj_list:
            continue
        st = METHOD_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})

        max_len  = max(len(t) for t in traj_list)
        padded   = [t + [t[-1]] * (max_len - len(t)) for t in traj_list]
        arr      = np.array(padded)
        mean_t   = arr.mean(axis=0)
        std_t    = arr.std(axis=0)
        xs       = list(range(max_len))

        ax.fill_between(xs, mean_t - std_t, mean_t + std_t,
                        alpha=0.08, color=st["color"])
        ax.plot(xs, mean_t, color=st["color"], lw=st["lw"], ls=st["ls"],
                label=method)

    ax.set_xlabel("Offer step")
    ax.set_ylabel("Remaining Budget $B_t$")
    ax.set_title(f"Budget Trajectory (B={B}, c={c})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    _save(fig, save_path)


# ── Figure B-3: Revenue vs K at multiple budgets (2×2) ───────────────────────

def plot_revenue_vs_k_multi_budget(
    results_by_B: Dict[str, Dict],
    budgets: List[float],
    save_path: str,
    n_total: int = 1000,
) -> None:
    """Revenue vs K (acceptances) at different budget levels (2×2 grid).

    Subplots: one per budget level from `budgets` list (max 4).
    3 lines per subplot: Greedy+Budget (red), Two-Phase-DP (orange), LSTM-Budget (black).

    At tight B: curves truncated early (bankruptcy).
    At loose B: curves match Idea 1 performance.

    Args:
        results_by_B: Budget eval results dict.
        budgets:      List of B values to plot (≤4 values).
        save_path:    Output path.
    """
    plt.rcParams.update(PLT_CFG)
    n_plots = min(len(budgets), 4)
    ncols   = 2
    nrows   = (n_plots + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 5 * nrows))
    axes_flat = axes.flat if hasattr(axes, "flat") else [axes]

    methods_to_show = ["Greedy+Budget", "Two-Phase-DP", "LSTM-Budget"]

    for idx, B in enumerate(budgets[:n_plots]):
        ax    = list(axes_flat)[idx]
        key   = f"B={B:.0f}" if f"B={B:.0f}" in results_by_B else f"B={int(B)}"
        res_b = results_by_B.get(key, {})

        for method in methods_to_show:
            st  = METHOD_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})
            res = res_b.get(method, {})
            # Try to build a cumulative acceptance curve from budget_trajectory + revenue
            # Placeholder: use single-point revenue from eval
            r   = res.get("revenue", {})
            rev = r.get("mean", 0.0) if isinstance(r, dict) else float(r or 0)
            na  = res.get("n_accepted", {})
            n_a = na.get("mean", 0.0) if isinstance(na, dict) else float(na or 0)

            # Draw a simple point + bar (detailed curves require raw rollouts)
            ax.bar(method.replace("-", "\n"), rev, color=st["color"], alpha=0.7,
                   label=method if idx == 0 else "_nolegend_")

        ax.set_title(f"B={B}")
        ax.set_ylabel("Total Revenue")
        ax.set_ylim(0, max(UNCONSTRAINED_REV + 50, 10))
        ax.axhline(UNCONSTRAINED_REV, color="gray", lw=0.8, ls="--", alpha=0.5)

    # Hide empty subplots
    for idx in range(n_plots, nrows * ncols):
        list(axes_flat)[idx].set_visible(False)

    fig.suptitle("Revenue at Different Budget Levels", fontsize=14)
    fig.tight_layout()
    _save(fig, save_path)


# ── Figure B-4: Sensitivity to production cost c ─────────────────────────────

def plot_revenue_vs_cost(
    results_by_c: Dict[str, Dict],
    B: float,
    save_path: str,
) -> None:
    """Revenue at fixed B across different production costs.

    x-axis: "Production cost (c)"
    y-axis: "Total Revenue"

    4 lines: Greedy+B, DP, LSTM-Idea1, LSTM-Budget.
    At high c: only LSTM-Budget survives profitably.

    Args:
        results_by_c: {c_val: {method_name: result_dict}}.
        B:            Initial budget used (for title).
        save_path:    Output path.
    """
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(8, 6))

    c_keys   = sorted(results_by_c.keys(), key=lambda k: float(str(k).replace("c=", "")))
    c_vals   = [float(str(k).replace("c=", "")) for k in c_keys]
    x_pos    = list(range(len(c_vals)))
    x_labels = [f"c={c}" for c in c_vals]

    ax.axhline(UNCONSTRAINED_REV, color="gray", lw=0.8, ls="--", alpha=0.5)

    for method in METHOD_ORDER:
        means, stds = [], []
        for key in c_keys:
            res = results_by_c.get(key, {}).get(method, {})
            rev = res.get("revenue", {})
            m   = rev.get("mean", 0.0) if isinstance(rev, dict) else float(rev or 0)
            s   = rev.get("std",  0.0) if isinstance(rev, dict) else 0.0
            means.append(m)
            stds.append(s)

        if not any(m > 0 for m in means):
            continue

        st    = METHOD_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})
        m_arr = np.array(means)
        s_arr = np.array(stds)
        ax.fill_between(x_pos, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.08, color=st["color"])
        ax.plot(x_pos, m_arr, color=st["color"], lw=st["lw"], ls=st["ls"],
                label=method)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Production cost c")
    ax.set_ylabel("Total Revenue")
    ax.set_title(f"Revenue Sensitivity to Production Cost (B={B})")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    _save(fig, save_path)


# ── Master generation function ────────────────────────────────────────────────

def generate_all_budget_figures(
    results_by_B: Dict[str, Dict],
    figure_dir: str,
    production_cost: float = 0.3,
    budget_traj: Optional[Dict[str, List[List[float]]]] = None,
    results_by_c: Optional[Dict[str, Dict]] = None,
) -> None:
    """Generate all Idea 3 budget figures.

    Args:
        results_by_B:      {B_val: {method: result}} from run_budget_eval.
        figure_dir:        Output directory (created if needed).
        production_cost:   c value for titles.
        budget_traj:       {method: [budget_history]} for B-2 trajectory plot.
        results_by_c:      {c_val: {method: result}} for B-4 sensitivity plot.
    """
    os.makedirs(figure_dir, exist_ok=True)
    print(f"Generating budget figures → {figure_dir}")

    # B-1: Revenue vs Budget
    plot_revenue_vs_budget(
        results_by_B, production_cost,
        save_path=os.path.join(figure_dir, "budget_fig1_revenue_vs_B.pdf"),
    )

    # B-2: Budget trajectory (requires raw budget histories)
    B_key = next(iter(results_by_B), None)
    if budget_traj is not None and B_key is not None:
        B_val = float(B_key.replace("B=", ""))
        plot_budget_trajectory(
            budget_traj, B=B_val, c=production_cost,
            save_path=os.path.join(figure_dir, "budget_fig2_trajectory.pdf"),
        )
    else:
        print("  [SKIP] B-2: no raw budget trajectories provided")

    # B-3: Multi-budget grid
    B_vals   = sorted(
        [float(k.replace("B=", "")) for k in results_by_B],
        key=lambda x: x
    )
    budgets4 = B_vals[:4]   # first 4 budget levels
    if budgets4:
        plot_revenue_vs_k_multi_budget(
            results_by_B, budgets4,
            save_path=os.path.join(figure_dir, "budget_fig3_multi_budget.pdf"),
        )

    # B-4: Cost sensitivity (requires results_by_c)
    if results_by_c is not None:
        B_ref = B_vals[min(2, len(B_vals) - 1)]   # use B=5 as reference
        plot_revenue_vs_cost(
            results_by_c, B=B_ref,
            save_path=os.path.join(figure_dir, "budget_fig4_cost_sensitivity.pdf"),
        )
    else:
        print("  [SKIP] B-4: no cost-sensitivity results provided")

    print("Budget figures done.")
