"""src/utils/idea3_figures.py — Publication-quality figures for Idea 3 (Budget-constrained).

Generates:
  fig_idea3_main.{pdf,png}        — 2-panel Revenue vs k (FF + Rice-FB)
  fig_idea3_oracle_pct.{pdf,png}  — 2-panel % of Oracle achieved
  paper_table_idea3_final.tex     — Combined LaTeX table (both networks)

Usage (called from experiments/make_idea3_figures.py):
  from src.utils.idea3_figures import make_idea3_figures
  make_idea3_figures(ff_json, rice_json, fig_dir, log_dir)
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Style ──────────────────────────────────────────────────────────────────────
_STYLE = "seaborn-v0_8-paper"
_DPI   = 300

# (label, color, ls, lw, marker, zorder)
_SPEC: Dict[str, tuple] = {
    "DP-Oracle":      ("Oracle (upper bound)", "#aaaaaa", ":",  1.5, None,  1),
    "Greedy+Budget":  ("Greedy+Budget",         "#d62728", "-",  2.0, None,  3),
    "DP-Calibrated":  ("DP-Calibrated",         "#ff7f0e", "-",  1.5, None,  3),
    "LSTM-Idea1":     ("LSTM-Idea1",            "#9467bd", "--", 1.5, None,  2),
    "LSTM-Idea3":     ("LSTM-Budget (ours)",    "#000000", "-",  2.5, "s",   4),
}

# Methods in drawing order
_DRAW_ORDER = ["DP-Oracle", "Greedy+Budget", "DP-Calibrated", "LSTM-Idea1", "LSTM-Idea3"]


def _gv(r: Optional[dict], key: str = "revenue", sub: str = "mean") -> float:
    if r is None:
        return float("nan")
    v = r.get(key, {})
    return v.get(sub, 0.0) if isinstance(v, dict) else float(v or 0)


def _rev_array(data: dict, sweep_k: List[int], method: str) -> np.ndarray:
    return np.array([
        _gv(data.get(f"k={k}", {}).get(method), "revenue")
        for k in sweep_k
    ])


def _load(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


# ── Figure 1: Revenue vs k (2 panels side by side) ────────────────────────────

def _plot_panel(ax: plt.Axes, data: dict, sweep_k: List[int],
                title: str, annotate: str = "ff") -> None:
    """Draw one panel of the revenue-vs-k figure.

    Args:
        ax:       Matplotlib axes object.
        data:     Results dict (k-label → method → stats).
        sweep_k:  List of k values.
        title:    Panel title.
        annotate: 'ff' or 'rice' — determines annotation style.
    """
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    x = np.array(sweep_k, dtype=float)
    methods_in_data = set()
    for k_key in data:
        methods_in_data.update(data[k_key].keys())

    for method in _DRAW_ORDER:
        if method not in methods_in_data:
            continue
        label, color, ls, lw, marker, zo = _SPEC[method]
        y = _rev_array(data, sweep_k, method)
        valid = ~np.isnan(y)
        if not valid.any():
            continue

        kw: dict = dict(color=color, linestyle=ls, linewidth=lw, zorder=zo, label=label)
        if marker:
            kw["marker"] = marker
            kw["markersize"] = 5
            kw["markevery"] = max(1, len(sweep_k) // 6)

        ax.plot(x[valid], y[valid], **kw)

    ax.set_xlabel("Budget  $k$  (units of production cost $c$)", fontsize=9)
    ax.set_ylabel("Revenue (mean over trials)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    if annotate == "ff":
        _annotate_ff(ax, data, sweep_k, x)
    elif annotate == "rice":
        _annotate_rice(ax, data, sweep_k, x)


def _annotate_ff(ax: plt.Axes, data: dict, sweep_k: List[int],
                 x: np.ndarray) -> None:
    """Annotate the FF panel: DP-Cal vs Greedy crossover + 'DP-Cal beats Greedy' region."""
    y_greedy  = _rev_array(data, sweep_k, "Greedy+Budget")
    y_dpcal   = _rev_array(data, sweep_k, "DP-Calibrated")
    y_lstm3   = _rev_array(data, sweep_k, "LSTM-Idea3")

    # Find crossover k where Greedy overtakes DP-Cal (if any)
    crossover_k = None
    for i in range(len(sweep_k) - 1):
        if (not np.isnan(y_greedy[i]) and not np.isnan(y_dpcal[i]) and
                y_dpcal[i] > y_greedy[i] and
                not np.isnan(y_dpcal[i+1]) and not np.isnan(y_greedy[i+1]) and
                y_greedy[i+1] >= y_dpcal[i+1] - 1.0):
            crossover_k = sweep_k[i+1]
            crossover_y = y_greedy[i+1]
            break

    if crossover_k is not None:
        ax.axvline(crossover_k, color="#aaaaaa", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.annotate(f"Greedy–DP-Cal\ncrossover k≈{crossover_k}",
                    xy=(crossover_k, crossover_y),
                    xytext=(crossover_k + 2, crossover_y * 0.85),
                    fontsize=7, color="#555555",
                    arrowprops=dict(arrowstyle="->", color="#888888", lw=0.8))

    # Shade region where DP-Cal > Greedy
    shade_ks = [k for i, k in enumerate(sweep_k)
                if not np.isnan(y_dpcal[i]) and not np.isnan(y_greedy[i])
                and y_dpcal[i] > y_greedy[i]]
    if shade_ks:
        ax.axvspan(min(shade_ks) - 0.5, max(shade_ks) + 0.5,
                   alpha=0.05, color="#ff7f0e", label="_no_legend_")
        ax.text(max(shade_ks) / 2 + 1, ax.get_ylim()[1] * 0.92,
                "DP-Cal\n> Greedy", fontsize=7, color="#ff7f0e",
                ha="center", alpha=0.8)


def _annotate_rice(ax: plt.Axes, data: dict, sweep_k: List[int],
                   x: np.ndarray) -> None:
    """Annotate the Rice-FB panel: LSTM3 phase transition at k=10."""
    y_lstm3 = _rev_array(data, sweep_k, "LSTM-Idea3")
    y_dpcal = _rev_array(data, sweep_k, "DP-Calibrated")

    # Find k=10 index
    if 10 in sweep_k:
        idx10 = sweep_k.index(10)
        lstm3_10  = y_lstm3[idx10]
        dpcal_10  = y_dpcal[idx10]

        if not np.isnan(lstm3_10) and lstm3_10 > 1:
            # Phase transition annotation
            ax.annotate("Phase transition\nat k=10",
                        xy=(10, lstm3_10),
                        xytext=(14, lstm3_10 * 0.75),
                        fontsize=7, color="#000000",
                        arrowprops=dict(arrowstyle="->", color="#333333", lw=0.8))

            if not np.isnan(dpcal_10) and dpcal_10 > 0:
                ratio = lstm3_10 / dpcal_10
                ax.text(10, lstm3_10 * 1.04,
                        f"LSTM3: {ratio:.0f}× DP-Cal",
                        fontsize=7, color="#000000", ha="center",
                        fontweight="bold")


def make_revenue_figure(
    ff_data: dict,
    rice_data: dict,
    sweep_k_ff: List[int],
    sweep_k_rice: List[int],
    out_dir: str,
) -> None:
    """Generate fig_idea3_main: 2-panel revenue vs k.

    Args:
        ff_data:       FF results dict (k-label → method → stats).
        rice_data:     Rice-FB results dict.
        sweep_k_ff:    k values for FF panel.
        sweep_k_rice:  k values for Rice-FB panel.
        out_dir:       Directory to save figure.
    """
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    _plot_panel(axes[0], ff_data, sweep_k_ff,
                title="(a) Forest Fire  ($n=1000$, avg\_deg$\\approx$4.8, $w_{ij}\\sim U(0,2)$)",
                annotate="ff")

    _plot_panel(axes[1], rice_data, sweep_k_rice,
                title="(b) Rice–Facebook  ($n=443$, avg\_deg$\\approx$44, real network)",
                annotate="rice")

    # Shared legend below the panels
    handles, labels = [], []
    for method in _DRAW_ORDER:
        present = any(method in ff_data.get(k, {}) or method in rice_data.get(k, {})
                      for k in ff_data) or any(method in rice_data.get(k, {})
                                                for k in rice_data)
        if not present:
            continue
        label, color, ls, lw, marker, _ = _SPEC[method]
        line = plt.Line2D([], [], color=color, linestyle=ls, linewidth=lw,
                          label=label,
                          marker=marker if marker else "None",
                          markersize=5 if marker else 0)
        handles.append(line)
        labels.append(label)

    fig.legend(handles, labels, loc="lower center", ncol=len(handles),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.04))

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_idea3_main.{ext}"),
                    dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/fig_idea3_main.{{pdf,png}}")


# ── Figure 2: % of Oracle achieved ────────────────────────────────────────────

def make_oracle_pct_figure(
    ff_data: dict,
    rice_data: dict,
    sweep_k_ff: List[int],
    sweep_k_rice: List[int],
    out_dir: str,
) -> None:
    """Generate fig_idea3_oracle_pct: revenue as % of Oracle per network.

    Args:
        ff_data:   FF results dict.
        rice_data: Rice-FB results dict.
        sweep_k_ff, sweep_k_rice: k-value lists.
        out_dir:   Output directory.
    """
    try:
        plt.style.use(_STYLE)
    except Exception:
        pass

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.0))
    _pct_methods = ["Greedy+Budget", "DP-Calibrated", "LSTM-Idea3"]

    for ax, data, sweep_k, panel_title in [
        (axes[0], ff_data,   sweep_k_ff,   "(a) Forest Fire ($n=1000$)"),
        (axes[1], rice_data, sweep_k_rice, "(b) Rice–Facebook ($n=443$)"),
    ]:
        x = np.array(sweep_k, dtype=float)
        oracle = _rev_array(data, sweep_k, "DP-Oracle")
        oracle_safe = np.where(oracle > 0, oracle, np.nan)

        for method in _pct_methods:
            label, color, ls, lw, marker, zo = _SPEC[method]
            y = _rev_array(data, sweep_k, method)
            pct = y / oracle_safe * 100.0
            valid = ~np.isnan(pct)
            if not valid.any():
                continue

            kw: dict = dict(color=color, linestyle=ls, linewidth=lw, label=label, zorder=zo)
            if marker:
                kw["marker"] = marker
                kw["markersize"] = 5
            ax.plot(x[valid], pct[valid], **kw)

        ax.axhline(100, color="#aaaaaa", linestyle=":", linewidth=0.8, label="Oracle (100%)")
        ax.set_ylim(0, 115)
        ax.set_xlim(left=0)
        ax.set_xlabel("Budget  $k$  (units of $c$)", fontsize=9)
        ax.set_ylabel("Revenue as % of Oracle upper bound", fontsize=9)
        ax.set_title(panel_title, fontsize=10, fontweight="bold")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    os.makedirs(out_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(out_dir, f"fig_idea3_oracle_pct.{ext}"),
                    dpi=_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_dir}/fig_idea3_oracle_pct.{{pdf,png}}")


# ── LaTeX table: paper_table_idea3_final ──────────────────────────────────────

_TABLE_METHODS = [
    ("DP-Oracle",     r"\textit{Oracle (UB)}"),
    ("Greedy+Budget", "Greedy+Budget"),
    ("DP-Calibrated", "DP-Calibrated"),
    ("LSTM-Idea1",    "LSTM-Idea1"),
    ("LSTM-Idea3",    r"\textbf{LSTM-Budget (ours)}"),
]

_TABLE_K = [3, 10, 20, 40]


def make_final_table(
    ff_data: dict,
    rice_data: dict,
    log_dir: str,
) -> None:
    """Generate paper_table_idea3_final.tex.

    Args:
        ff_data:   FF results dict.
        rice_data: Rice-FB results dict.
        log_dir:   Directory for .tex output.
    """
    os.makedirs(log_dir, exist_ok=True)

    # Find which non-oracle methods are present in both
    non_oracle_keys = [k for k, _ in _TABLE_METHODS if k != "DP-Oracle"]

    def best_nonoracle(data: dict, k: int) -> float:
        vals = [_gv(data.get(f"k={k}", {}).get(m))
                for m in non_oracle_keys]
        vals = [v for v in vals if not math.isnan(v)]
        return max(vals) if vals else float("nan")

    lines = [
        r"\begin{table}[t]",
        r"\caption{Budget-constrained revenue ($c{=}0.3$, $B{=}k \cdot c$). "
        r"Oracle is a perfect-information upper bound (unachievable). "
        r"Best non-oracle in \textbf{bold}.}",
        r"\label{tab:budget_final}",
        r"\centering",
        r"\small",
        r"\begin{tabular}{lcccccccc}",
        r"\toprule",
        r"& \multicolumn{4}{c}{Forest Fire ($n{=}1000$)}"
        r"& \multicolumn{4}{c}{Rice--Facebook ($n{=}443$)} \\",
        r"\cmidrule(lr){2-5} \cmidrule(lr){6-9}",
        "Method & "
        + " & ".join([f"$k{{=}}{k}$" for k in _TABLE_K])
        + " & "
        + " & ".join([f"$k{{=}}{k}$" for k in _TABLE_K])
        + r" \\",
        r"\midrule",
    ]

    for method_key, display_name in _TABLE_METHODS:
        is_oracle = (method_key == "DP-Oracle")
        row_vals = []

        for data in (ff_data, rice_data):
            best_per_k = {k: best_nonoracle(data, k) for k in _TABLE_K}
            for k in _TABLE_K:
                v = _gv(data.get(f"k={k}", {}).get(method_key))
                if math.isnan(v):
                    row_vals.append("--")
                elif is_oracle:
                    row_vals.append(r"\textit{" + f"{v:.0f}" + r"}")
                else:
                    best = best_per_k.get(k, float("nan"))
                    if not math.isnan(best) and abs(v - best) < 0.5:
                        row_vals.append(r"\textbf{" + f"{v:.0f}" + r"}")
                    else:
                        row_vals.append(f"{v:.0f}")

        lines.append(f"{display_name} & " + " & ".join(row_vals) + r" \\")
        if is_oracle:
            lines.append(r"\midrule")

    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    path = os.path.join(log_dir, "paper_table_idea3_final.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


# ── Main entry point ──────────────────────────────────────────────────────────

def make_idea3_figures(
    ff_json: str,
    rice_json: str,
    fig_dir: str,
    log_dir: str,
) -> None:
    """Generate all Idea 3 publication figures and tables.

    Args:
        ff_json:   Path to dp_upgrade_eval_ff.json (FF n=1000 results).
        rice_json: Path to dp_upgrade_eval_rice_lstm.json (Rice-FB results).
        fig_dir:   Directory to save figures (will create budget/ subdirectory).
        log_dir:   Directory to save LaTeX tables.
    """
    ff_data   = _load(ff_json)
    rice_data = _load(rice_json)

    sweep_k_ff   = sorted([int(k.split("=")[1]) for k in ff_data])
    sweep_k_rice = sorted([int(k.split("=")[1]) for k in rice_data])

    out_dir = os.path.join(fig_dir, "budget")

    make_revenue_figure(ff_data, rice_data, sweep_k_ff, sweep_k_rice, out_dir)
    make_oracle_pct_figure(ff_data, rice_data, sweep_k_ff, sweep_k_rice, out_dir)
    make_final_table(ff_data, rice_data, log_dir)

    # Print summary stats for reporting
    _print_summary(ff_data, rice_data, sweep_k_ff, sweep_k_rice)


def _print_summary(
    ff_data: dict, rice_data: dict,
    sweep_k_ff: List[int], sweep_k_rice: List[int],
) -> None:
    """Print per-k winner table to stdout."""
    print("\n" + "=" * 70)
    print("IDEA 3 FINAL RESULTS SUMMARY")
    print("=" * 70)

    for label, data, sweep_k in [
        ("Forest Fire n=1000", ff_data, sweep_k_ff),
        ("Rice-Facebook n=443", rice_data, sweep_k_rice),
    ]:
        print(f"\n{'─'*50}")
        print(f"  {label}")
        print(f"{'─'*50}")
        print(f"  {'k':<6} {'Greedy':>8} {'DP-Cal':>8} {'LSTM1':>8} {'LSTM3':>10} "
              f"{'Oracle':>8} {'LSTM3/Oracle%':>14} {'Winner':>12}")
        print(f"  {'─'*80}")

        for k in sweep_k:
            k_key = f"k={k}"
            r = data.get(k_key, {})
            greedy  = _gv(r.get("Greedy+Budget"))
            dpcal   = _gv(r.get("DP-Calibrated"))
            lstm1   = _gv(r.get("LSTM-Idea1"))
            lstm3   = _gv(r.get("LSTM-Idea3"))
            oracle  = _gv(r.get("DP-Oracle"))
            oracle_pct = lstm3 / oracle * 100 if oracle > 0 and not math.isnan(lstm3) else float("nan")

            # Find winner among non-oracle methods
            candidates = {"Greedy": greedy, "DP-Cal": dpcal,
                          "LSTM1": lstm1, "LSTM3": lstm3}
            valid = {n: v for n, v in candidates.items() if not math.isnan(v)}
            winner = max(valid, key=valid.get) if valid else "?"

            pct_str = f"{oracle_pct:>7.1f}%" if not math.isnan(oracle_pct) else "     --"
            print(f"  k={k:<4} {greedy:>8.1f} {dpcal:>8.1f} {lstm1:>8.1f} "
                  f"{lstm3:>10.1f} {oracle:>8.1f} {pct_str:>14} {winner:>12}")

    print("\nKey comparisons at specific k values:")
    for data, label, k_highlight in [
        (ff_data,   "FF", 10),
        (rice_data, "Rice", 10),
    ]:
        r = data.get(f"k={k_highlight}", {})
        lstm3  = _gv(r.get("LSTM-Idea3"))
        dpcal  = _gv(r.get("DP-Calibrated"))
        oracle = _gv(r.get("DP-Oracle"))
        if not math.isnan(lstm3) and dpcal > 0:
            ratio = lstm3 / dpcal
            pct   = lstm3 / oracle * 100 if oracle > 0 else float("nan")
            print(f"  [{label}] k={k_highlight}: LSTM3={lstm3:.1f}  DP-Cal={dpcal:.1f}  "
                  f"ratio={ratio:.1f}×  oracle_pct={pct:.1f}%")

    for data, label, k_val in [(ff_data, "FF", 3), (rice_data, "Rice", 3)]:
        r = data.get(f"k={k_val}", {})
        lstm3  = _gv(r.get("LSTM-Idea3"))
        oracle = _gv(r.get("DP-Oracle"))
        if oracle > 0 and not math.isnan(lstm3):
            pct = lstm3 / oracle * 100
            print(f"  [{label}] k={k_val}: LSTM3/Oracle = {pct:.1f}%")

    print("=" * 70)
