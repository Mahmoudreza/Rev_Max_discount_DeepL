"""src/utils/dp_upgrade_visualization.py — Figures and LaTeX tables for DP-Upgrade eval.

Generates:
  fig_dp1_revenue_vs_k.{pdf,png}   — Revenue vs k (budget in units of c)
  fig_dp2_bmin_feasibility.{pdf,png} — 3-panel B_min feasibility analysis
  paper_table_dp_upgrade.tex        — Methods × k revenue table
  paper_table_bmin.tex              — Methods × k feasibility table
"""

from __future__ import annotations

import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Style constants ────────────────────────────────────────────────────────────
# Reuse exact rcParams from earlier figure work in the project.
_STYLE = "seaborn-v0_8-paper"
_DPI   = 300

# Line style spec: (label, color, linestyle, linewidth, marker, zorder)
_LINE_STYLES = {
    "Greedy+Budget":  ("Greedy",          "#e41a1c", "-",    2.0, None, 3),
    "Two-Phase-DP":   ("DP-naive",         "#ff7f00", "-.",   1.2, None, 2),
    "DP-Calibrated":  ("DP-Calibrated",    "#ff7f00", "-",    2.0, None, 3),
    "DP-Receding":    ("DP-Receding",      "#8b4513", "-",    1.5, None, 3),
    "LSTM-Idea3":     ("LSTM-Idea3",       "#000000", "-",    2.5, "s",  4),
    "LSTM-Idea1":     ("LSTM-Idea1",       "#984ea3", "--",   1.5, None, 2),
    "DP-Oracle":      ("DP-Oracle (UB)",   "#888888", ":",    1.5, None, 1),
}

_ORACLE_KEY = "DP-Oracle"


def _gv(r: dict, key: str, sub: str = "mean") -> float:
    v = r.get(key, {})
    return v.get(sub, 0.0) if isinstance(v, dict) else float(v or 0)


def _rev_array(results: dict, sweep_k: List[int], method: str) -> np.ndarray:
    """Extract revenue mean array over sweep_k for a given method."""
    vals = []
    for k in sweep_k:
        k_key = f"k={k}"
        r = results.get(k_key, {}).get(method, None)
        vals.append(_gv(r, "revenue") if r is not None else float("nan"))
    return np.array(vals)


# ── Figure 1: Revenue vs k ─────────────────────────────────────────────────────

def generate_dp_figures(
    results: dict,
    sweep_k: List[int],
    fig_dir: str,
) -> None:
    """Generate fig_dp1 (revenue vs k) and fig_dp2 (bmin feasibility).

    Args:
        results:  dict from run_dp_upgrade_eval (k_label → method → agg).
        sweep_k:  List of k values (budget in units of c).
        fig_dir:  Directory to save figures.
    """
    os.makedirs(fig_dir, exist_ok=True)
    _plot_revenue_vs_k(results, sweep_k, fig_dir)
    _plot_bmin_feasibility(results, sweep_k, fig_dir)


def _plot_revenue_vs_k(results: dict, sweep_k: List[int], fig_dir: str) -> None:
    """fig_dp1_revenue_vs_k: revenue curves for all methods."""
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.array(sweep_k, dtype=float)

    methods_in_results = set()
    for k_key in results:
        methods_in_results.update(results[k_key].keys())

    # Draw oracle last so other lines are visible
    draw_order = [m for m in _LINE_STYLES if m != _ORACLE_KEY and m in methods_in_results]
    draw_order.append(_ORACLE_KEY)

    for method in draw_order:
        if method not in methods_in_results:
            continue
        label, color, ls, lw, marker, zo = _LINE_STYLES[method]
        y = _rev_array(results, sweep_k, method)
        valid = ~np.isnan(y)
        if not valid.any():
            continue

        kw = dict(color=color, linestyle=ls, linewidth=lw, zorder=zo)
        if marker:
            kw["marker"] = marker
            kw["markersize"] = 6

        if method == _ORACLE_KEY:
            ax.plot(x[valid], y[valid], label=label, **kw)
            # Annotate oracle line
            last_x, last_y = x[valid][-1], y[valid][-1]
            ax.annotate("oracle upper bound", xy=(last_x, last_y),
                        xytext=(-60, 8), textcoords="offset points",
                        fontsize=7, color=color, arrowprops=dict(arrowstyle="-", color=color))
        else:
            ax.plot(x[valid], y[valid], label=label, **kw)

    ax.set_xlabel("k  (budget = k × c)", fontsize=11)
    ax.set_ylabel("Revenue (mean over trials)", fontsize=11)
    ax.set_title("DP-Upgrade: Revenue vs Budget (Forest Fire, n=1000)", fontsize=11)
    ax.legend(fontsize=8, ncol=2, loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"fig_dp1_revenue_vs_k.{ext}")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ── Figure 2: Bmin feasibility (3-panel) ──────────────────────────────────────

def _plot_bmin_feasibility(results: dict, sweep_k: List[int], fig_dir: str) -> None:
    """fig_dp2_bmin_feasibility: 3-panel feasibility + revenue-among-feasible."""
    bmin_fracs = [0.0, 0.25, 0.5]
    panel_titles = [r"$B_{\min} = 0$", r"$B_{\min} = 0.25 B_0$", r"$B_{\min} = 0.5 B_0$"]

    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=False)
    x = np.array(sweep_k, dtype=float)

    methods_in_results = set()
    for k_key in results:
        methods_in_results.update(results[k_key].keys())

    for panel_idx, (bfrac, title) in enumerate(zip(bmin_fracs, panel_titles)):
        ax = axes[panel_idx]

        draw_order = [m for m in _LINE_STYLES if m != _ORACLE_KEY and m in methods_in_results]
        draw_order.append(_ORACLE_KEY)

        for method in draw_order:
            if method not in methods_in_results:
                continue
            label, color, ls, lw, marker, zo = _LINE_STYLES[method]

            # Extract revenue per k, but grey out infeasible points
            y_rev   = []
            feasible_mask = []
            for k in sweep_k:
                k_key = f"k={k}"
                r = results.get(k_key, {}).get(method, None)
                if r is None:
                    y_rev.append(float("nan"))
                    feasible_mask.append(False)
                    continue

                rev_mean = _gv(r, "revenue")
                y_rev.append(rev_mean)

                # Feasibility: check if ALL trials kept budget >= bfrac * B0
                B0   = k * 0.3
                traj = r.get("budget_trajectory", [])
                if isinstance(traj, list) and len(traj) > 0:
                    if isinstance(traj[0], list):
                        frac_ok = all(
                            min(t) >= bfrac * B0 - 1e-9 if t else (bfrac <= 0)
                            for t in traj
                        )
                    else:
                        frac_ok = min(traj) >= bfrac * B0 - 1e-9
                else:
                    frac_ok = True
                feasible_mask.append(frac_ok)

            y_rev = np.array(y_rev)
            feasible_mask = np.array(feasible_mask)

            # Feasible points: solid, normal
            kw = dict(color=color, linestyle=ls, linewidth=lw, zorder=zo)
            if marker:
                kw["marker"] = marker; kw["markersize"] = 6
            ax.plot(x[feasible_mask], y_rev[feasible_mask], label=label, **kw)

            # Infeasible points: hollow grey markers
            infeas = ~feasible_mask & ~np.isnan(y_rev)
            if infeas.any():
                ax.scatter(x[infeas], y_rev[infeas],
                           s=30, facecolors="none", edgecolors="grey",
                           linewidths=0.8, zorder=1)

        # Annotate which methods are feasible at k<=5
        k_small = [k for k in sweep_k if k <= 5]
        feasible_at_5 = []
        for method in draw_order:
            if method not in methods_in_results or method == _ORACLE_KEY:
                continue
            all_feas = True
            for k in k_small:
                k_key = f"k={k}"
                r = results.get(k_key, {}).get(method, {})
                B0 = k * 0.3
                traj = r.get("budget_trajectory", [])
                if isinstance(traj, list) and traj:
                    if isinstance(traj[0], list):
                        feas = all(min(t) >= bfrac * B0 - 1e-9 if t else (bfrac <= 0)
                                   for t in traj)
                    else:
                        feas = min(traj) >= bfrac * B0 - 1e-9
                else:
                    feas = True
                if not feas:
                    all_feas = False
                    break
            if all_feas:
                feasible_at_5.append(_LINE_STYLES[method][0])

        if feasible_at_5:
            ax.text(0.98, 0.02, "Feasible k≤5:\n" + ", ".join(feasible_at_5),
                    transform=ax.transAxes, fontsize=7, ha="right", va="bottom",
                    color="#333333", bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_title(title, fontsize=10)
        ax.set_xlabel("k  (B = k·c)", fontsize=9)
        if panel_idx == 0:
            ax.set_ylabel("Revenue (mean)", fontsize=9)
        ax.grid(True, alpha=0.3)
        if panel_idx == 1:
            ax.legend(fontsize=7, ncol=1, loc="upper left")

    fig.suptitle("Budget Feasibility Analysis (Forest Fire, n=1000)", fontsize=11)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        path = os.path.join(fig_dir, f"fig_dp2_bmin_feasibility.{ext}")
        fig.savefig(path, dpi=_DPI, bbox_inches="tight")
    plt.close(fig)


# ── LaTeX tables ───────────────────────────────────────────────────────────────

def generate_latex_tables(
    results: dict,
    bmin_results: dict,
    sweep_k: List[int],
    log_dir: str,
) -> None:
    """Generate paper_table_dp_upgrade.tex and paper_table_bmin.tex.

    Args:
        results:      dp_upgrade_eval.json dict.
        bmin_results: apply_bmin_analysis output dict.
        sweep_k:      List of k values.
        log_dir:      Directory to save .tex files.
    """
    os.makedirs(log_dir, exist_ok=True)
    _table_revenue(results, sweep_k, log_dir)
    _table_bmin(bmin_results, sweep_k, log_dir)


# Non-oracle methods in display order
# DP-Receding dropped from paper; Two-Phase-DP optional (shown if present)
_NON_ORACLE = ["Greedy+Budget", "Two-Phase-DP", "DP-Calibrated",
               "LSTM-Idea1", "LSTM-Idea3"]
_DISPLAY_NAMES = {
    "Greedy+Budget":  "Greedy+Budget",
    "Two-Phase-DP":   "DP-naive",
    "DP-Calibrated":  "DP-Calibrated",
    "LSTM-Idea1":     "LSTM-Idea1",
    "LSTM-Idea3":     "LSTM-Budget",
    "DP-Oracle":      r"\textit{DP-Oracle (UB)}",
}


def _table_revenue(results: dict, sweep_k: List[int], log_dir: str) -> None:
    """Revenue table: methods × k, bold best non-oracle, oracle in italics on top row."""
    k_subset = [k for k in sweep_k]  # all 10 k values
    methods_present = set()
    for k_key in results:
        methods_present.update(results[k_key].keys())

    non_oracle = [m for m in _NON_ORACLE if m in methods_present]
    has_oracle = _ORACLE_KEY in methods_present

    col_spec = "l" + "r" * len(k_subset)
    header   = " & ".join([f"k={k}" for k in k_subset])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Revenue vs. budget (k = B/c). \textbf{Bold}: best non-oracle per column. "
        r"\textit{Italic}: oracle upper bound (not achievable). n=1000 Forest-Fire.}",
        r"\label{tab:dp_upgrade}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Method & {header} \\",
        r"\midrule",
    ]

    # Oracle row first (italics)
    if has_oracle:
        oracle_vals = []
        for k in k_subset:
            r = results.get(f"k={k}", {}).get(_ORACLE_KEY, None)
            v = _gv(r, "revenue") if r else float("nan")
            oracle_vals.append(f"\\textit{{{v:.1f}}}" if not math.isnan(v) else "--")
        disp = _DISPLAY_NAMES[_ORACLE_KEY]
        lines.append(rf"{disp} & {' & '.join(oracle_vals)} \\")
        lines.append(r"\midrule")

    # Non-oracle rows
    for method in non_oracle:
        disp = _DISPLAY_NAMES.get(method, method)
        vals = []
        revs_per_k = []
        for k in k_subset:
            r = results.get(f"k={k}", {}).get(method, None)
            v = _gv(r, "revenue") if r else float("nan")
            revs_per_k.append(v)

        # Find best non-oracle per column
        best_per_col = {}
        for ki, k in enumerate(k_subset):
            col_vals = [
                _gv(results.get(f"k={k}", {}).get(m, None) or {}, "revenue")
                for m in non_oracle
            ]
            col_vals = [v for v in col_vals if not math.isnan(v)]
            best_per_col[ki] = max(col_vals) if col_vals else float("nan")

        for ki, v in enumerate(revs_per_k):
            if math.isnan(v):
                vals.append("--")
            elif abs(v - best_per_col.get(ki, float("nan"))) < 0.05:
                vals.append(rf"\textbf{{{v:.1f}}}")
            else:
                vals.append(f"{v:.1f}")

        lines.append(rf"{disp} & {' & '.join(vals)} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(log_dir, "paper_table_dp_upgrade.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def _table_bmin(bmin_results: dict, sweep_k: List[int], log_dir: str) -> None:
    """Feasibility table at bmin=50% B0: feasible_rate per (method, k)."""
    frac_key  = "bmin_50"
    bmin_data = bmin_results.get(frac_key, {})

    methods_present: set = set()
    for k_key in bmin_data:
        methods_present.update(bmin_data[k_key].keys())

    non_oracle = [m for m in _NON_ORACLE if m in methods_present]
    k_subset   = [k for k in sweep_k]
    col_spec   = "l" + "r" * len(k_subset)
    header     = " & ".join([f"k={k}" for k in k_subset])

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Feasibility rate at $B_{\min} = 0.5 B_0$ solvency floor. "
        r"Entry = fraction of trials satisfying the floor. \textbf{Bold}: methods with 100\% feasibility.}",
        r"\label{tab:bmin}",
        rf"\begin{{tabular}}{{{col_spec}}}",
        r"\toprule",
        rf"Method & {header} \\",
        r"\midrule",
    ]

    for method in non_oracle:
        disp = _DISPLAY_NAMES.get(method, method)
        vals = []
        for k in k_subset:
            k_key = f"k={k}"
            entry = bmin_data.get(k_key, {}).get(method, {})
            fr    = entry.get("feasible_rate", None)
            if fr is None:
                vals.append("--")
            elif fr >= 1.0 - 1e-6:
                vals.append(r"\textbf{1.00}")
            else:
                vals.append(f"{fr:.2f}")
        lines.append(rf"{disp} & {' & '.join(vals)} \\")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(log_dir, "paper_table_bmin.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
