"""
src/utils/tc_visualization.py — TC figure suite (Idea 2).

Sequential model only (Babaei et al.). No IC cascade. No "cascade days".
τ = number of ACCEPTANCES (|S| = τ). Same model as Idea 1.

Produces 4 paper-ready figures:
  TC-Fig 1: Revenue vs deadline τ (THE core Idea 2 figure)
  TC-Fig 2: Cumulative revenue trajectory (all 4 methods)
  TC-Fig 3: Revenue vs K at fixed τ checkpoints (mirrors Idea 1 Figs 1-2)
  TC-Fig 4: Standard LSTM vs TC-fine-tuned LSTM across τ

Usage:
    from src.utils.tc_visualization import generate_all_tc_figures
    generate_all_tc_figures(tc_results, figure_dir, tc_results_v2=None)

    tc_results: output of src.evaluation.tc_baselines.run_tc_comparison_multi_graph()
"""

import json
import os
from typing import Dict, List, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Shared style (mirrors paper_figures.py exactly) ──────────────────────────

METHOD_STYLES = {
    "IE-Strategy":      {"color": "#1f77b4", "ls": "-",  "lw": 1.5, "marker": "o", "ms": 4},
    "Greedy-Discount":  {"color": "#d62728", "ls": "-",  "lw": 2.0, "marker": None, "ms": 5},
    "Rev-GNN-IM-RL":    {"color": "#ff7f0e", "ls": "--", "lw": 2.0, "marker": "^", "ms": 4},
    "Rev-GNN-LSTM":     {"color": "#000000", "ls": "-",  "lw": 2.5, "marker": "s", "ms": 4},
    "Rev-GNN-LSTM-TC":  {"color": "#2ca02c", "ls": "-.", "lw": 2.5, "marker": "D", "ms": 4},
}
METHOD_4 = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]

PLT_CFG = {
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
}


def _save(fig, name: str, figure_dir: str) -> None:
    os.makedirs(figure_dir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{figure_dir}/{name}.{ext}")
    plt.close(fig)
    print(f"  [✓] {figure_dir}/{name}.{{pdf,png}}")


# ── TC-Figure 1: Revenue vs Deadline τ (THE main figure) ─────────────────────

def tc_fig1_revenue_vs_tau(
    tc_results: Dict[str, Dict],
    figure_dir: str,
    title: str = "FF n=1000",
    tc_results_v2: Optional[Dict[str, Dict]] = None,
) -> None:
    """Line plot: revenue at deadline τ (number of acceptances).

    THE core Idea 2 figure. Shows LSTM maintains high revenue at small τ
    while IE-Strategy / Greedy collapse because early steps are free seeds.

    x-axis: "Deadline τ (number of acceptances)"
    y-axis: "Revenue within deadline τ"

    Args:
        tc_results:    Output of evaluate_tc_comparison() for base methods.
        figure_dir:    Output directory.
        title:         Dataset label for plot title (e.g., "FF n=1000").
        tc_results_v2: Optional dict with "Rev-GNN-LSTM-TC" key (TC fine-tuned).
    """
    # Collect τ values from results (sorted)
    sample_method = next(iter(tc_results))
    taus = sorted(tc_results[sample_method]["checkpoints"].keys())
    if not taus:
        print("  [SKIP] TC-Fig 1: no checkpoint data")
        return

    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = list(range(len(taus)))
    x_labels = [str(t) for t in taus]

    methods_to_plot = list(tc_results.keys())
    if tc_results_v2 and "Rev-GNN-LSTM-TC" in tc_results_v2:
        extra = {"Rev-GNN-LSTM-TC": tc_results_v2["Rev-GNN-LSTM-TC"]}
    else:
        extra = {}

    all_results = {**tc_results, **extra}
    for m in methods_to_plot + list(extra.keys()):
        res = all_results.get(m)
        if res is None:
            continue
        st = METHOD_STYLES.get(m, {"color": "gray", "ls": "-", "lw": 1.5,
                                    "marker": None, "ms": 4})
        means = [res["checkpoints"].get(t, 0.0) for t in taus]
        stds  = [res["checkpoints_std"].get(t, 0.0) for t in taus]
        m_arr = np.array(means, dtype=float)
        s_arr = np.array(stds,  dtype=float)
        mkw = ({"marker": st["marker"], "markersize": st["ms"],
                "markevery": max(1, len(taus) // 5)}
               if st["marker"] else {})
        ax.fill_between(x, m_arr - s_arr, m_arr + s_arr,
                        alpha=0.10, color=st["color"])
        ax.plot(x, m_arr, color=st["color"], ls=st["ls"], lw=st["lw"],
                label=m.replace("Rev-GNN-", ""), **mkw)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=10)
    ax.set_xlabel("Deadline τ (number of acceptances)")
    ax.set_ylabel("Revenue within deadline τ")
    ax.set_title(f"Time-Critical Revenue ({title})")
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    _save(fig, "tc_fig1_revenue_vs_tau", figure_dir)


# ── TC-Figure 2: Cumulative revenue trajectories ─────────────────────────────

def tc_fig2_cumulative_trajectories(
    curves_by_method: Dict[str, List[List[float]]],
    figure_dir: str,
    title: str = "FF n=1000",
    max_k: int = 500,
) -> None:
    """Cumulative revenue curve over acceptance count.

    Shows HOW revenue accumulates for each method.
    Expected: Greedy flat at 0 for first ~15 steps (free seeds),
              LSTM rises from step 1.

    x-axis: "Acceptance count (|S|)"
    y-axis: "Cumulative revenue"

    Args:
        curves_by_method: Dict: method → list of cum_rev_by_S curves (per trial).
        figure_dir:       Output directory.
        title:            Dataset label.
        max_k:            Maximum x-axis limit.
    """
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    for m, curves in curves_by_method.items():
        if not curves:
            continue
        st = METHOD_STYLES.get(m, {"color": "gray", "ls": "-", "lw": 1.5,
                                    "marker": None, "ms": 4})
        # Trim / pad all curves to max_k length, then average
        padded = []
        for c in curves:
            trimmed = c[:max_k]
            if trimmed:
                last = trimmed[-1]
                padded.append(trimmed + [last] * (max_k - len(trimmed)))
            else:
                padded.append([0.0] * max_k)
        arr = np.array(padded)
        mean_c = arr.mean(axis=0)
        std_c  = arr.std(axis=0)
        xs = list(range(1, max_k + 1))
        ax.fill_between(xs, mean_c - std_c, mean_c + std_c,
                        alpha=0.10, color=st["color"])
        ax.plot(xs, mean_c, color=st["color"], ls=st["ls"], lw=st["lw"],
                label=m.replace("Rev-GNN-", ""))

    ax.set_xlabel("Acceptance count (|S|)")
    ax.set_ylabel("Cumulative revenue")
    ax.set_title(f"Revenue Accumulation ({title})")
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    _save(fig, "tc_fig2_cumulative_trajectories", figure_dir)


# ── TC-Figure 3: Revenue vs τ, two datasets ──────────────────────────────────

def tc_fig3_revenue_vs_tau_two_datasets(
    tc_results_ff: Dict[str, Dict],
    tc_results_rice: Dict[str, Dict],
    figure_dir: str,
) -> None:
    """Revenue vs τ for FF n=1000 and Rice-Facebook (2-panel mirror of Idea 1 Figs 1-2).

    Args:
        tc_results_ff:   TC results for FF n=1000.
        tc_results_rice: TC results for Rice-Facebook.
        figure_dir:      Output directory.
    """
    plt.rcParams.update(PLT_CFG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _fill_ax(ax, results, subtitle):
        if not results:
            ax.text(0.5, 0.5, "No data", ha="center", transform=ax.transAxes)
            return
        sample = next(iter(results))
        taus = sorted(results[sample]["checkpoints"].keys())
        x = list(range(len(taus)))
        for m in METHOD_4:
            res = results.get(m)
            if res is None:
                continue
            st = METHOD_STYLES[m]
            means = np.array([res["checkpoints"].get(t, 0.0) for t in taus])
            stds  = np.array([res["checkpoints_std"].get(t, 0.0) for t in taus])
            mkw = ({"marker": st["marker"], "markersize": st["ms"],
                    "markevery": max(1, len(taus) // 5)}
                   if st["marker"] else {})
            ax.fill_between(x, means - stds, means + stds,
                            alpha=0.10, color=st["color"])
            ax.plot(x, means, color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=m.replace("Rev-GNN-", ""), **mkw)
        ax.set_xticks(x)
        ax.set_xticklabels([str(t) for t in taus], fontsize=9)
        ax.set_xlabel("Deadline τ (acceptances)")
        ax.set_ylabel("Revenue within τ")
        ax.set_title(subtitle)
        ax.legend(loc="lower right", fontsize=9)

    _fill_ax(ax1, tc_results_ff,   "(a) Forest Fire n=1000")
    _fill_ax(ax2, tc_results_rice, "(b) Rice-Facebook n=443 (zero-shot)")
    fig.suptitle("Time-Critical Revenue vs Deadline τ", fontsize=14)
    fig.tight_layout()
    _save(fig, "tc_fig3_two_datasets", figure_dir)


# ── TC-Figure 4: Standard LSTM vs TC-trained LSTM ────────────────────────────

def tc_fig4_standard_vs_tc_lstm(
    tc_results_base: Dict[str, Dict],
    tc_results_tc: Optional[Dict[str, Dict]],
    figure_dir: str,
) -> None:
    """Revenue vs τ: Idea 1 LSTM vs TC-fine-tuned LSTM.

    Shows TC fine-tuning benefit at small τ.
    Skipped if tc_results_tc not provided (run run_tc_training.py first).

    Args:
        tc_results_base: TC evaluation with standard Idea 1 LSTM.
        tc_results_tc:   TC evaluation with TC-fine-tuned LSTM. None → skip.
        figure_dir:      Output directory.
    """
    if tc_results_tc is None:
        print("  [SKIP] TC-Fig 4: no TC-trained results (run experiments/run_tc_training.py first)")
        return

    sample = next(iter(tc_results_base))
    taus = sorted(tc_results_base[sample]["checkpoints"].keys())
    x = list(range(len(taus)))

    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Reference line: Greedy-Discount
    if "Greedy-Discount" in tc_results_base:
        grd = tc_results_base["Greedy-Discount"]
        means = np.array([grd["checkpoints"].get(t, 0.0) for t in taus])
        ax.plot(x, means, color="#d62728", ls="--", lw=1.5, alpha=0.7,
                label="Greedy-Discount (ref.)")

    # Standard LSTM (Idea 1)
    if "Rev-GNN-LSTM" in tc_results_base:
        res = tc_results_base["Rev-GNN-LSTM"]
        means = np.array([res["checkpoints"].get(t, 0.0) for t in taus])
        stds  = np.array([res["checkpoints_std"].get(t, 0.0) for t in taus])
        ax.fill_between(x, means - stds, means + stds, alpha=0.10, color="black")
        ax.plot(x, means, color="black", ls="-", lw=2.5, marker="s",
                markersize=4, markevery=2, label="LSTM (Idea 1 training)")

    # TC-fine-tuned LSTM (Idea 2)
    tc_key = "Rev-GNN-LSTM-TC" if "Rev-GNN-LSTM-TC" in tc_results_tc else "Rev-GNN-LSTM"
    if tc_key in tc_results_tc:
        res = tc_results_tc[tc_key]
        means = np.array([res["checkpoints"].get(t, 0.0) for t in taus])
        stds  = np.array([res["checkpoints_std"].get(t, 0.0) for t in taus])
        ax.fill_between(x, means - stds, means + stds, alpha=0.10, color="#2ca02c")
        ax.plot(x, means, color="#2ca02c", ls="-.", lw=2.5, marker="D",
                markersize=4, markevery=2, label="LSTM-TC (Idea 2 training)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(t) for t in taus], fontsize=10)
    ax.set_xlabel("Deadline τ (number of acceptances)")
    ax.set_ylabel("Revenue within deadline τ")
    ax.set_title("TC Fine-Tuning Benefit (FF n=1000)")
    ax.legend(fontsize=10)
    fig.tight_layout()
    _save(fig, "tc_fig4_standard_vs_tc", figure_dir)


# ── Profit figures ────────────────────────────────────────────────────────────

# Method styles consistent across all TC figures
_PROFIT_STYLES: Dict[str, dict] = {
    "IE-Strategy":     {"color": "#1f77b4", "lw": 1.5, "ls": "-"},
    "Greedy-Discount": {"color": "#d62728", "lw": 2.0, "ls": "-"},
    "Rev-GNN-IM-RL":   {"color": "#9467bd", "lw": 1.5, "ls": "-."},
    "Rev-GNN-LSTM":    {"color": "#000000", "lw": 2.5, "ls": "-",
                        "marker": "s", "markersize": 3, "markevery": 40},
}
_METHOD_ORDER_PROFIT = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]


def plot_profit_vs_tau(
    method_trajectories: Dict[str, List[List[dict]]],
    production_cost: float,
    save_path: str,
) -> None:
    """Plot cumulative profit curves for all 4 Idea 1 methods.

    x-axis: "Number of acceptances (τ)"
    y-axis: "Cumulative profit (revenue − cost)"

    Horizontal dashed line at y=0 (breakeven). Vertical arrows at breakeven.

    Args:
        method_trajectories: method_name → list of trajectory lists (one per trial).
                             Each trajectory is a list of step dicts with
                             "accepted" bool and "price" float.
        production_cost:     Cost c per item delivered.
        save_path:           Full path for output PNG/PDF.
    """
    from src.evaluation.tc_evaluation import profit_curve, breakeven_point

    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Horizontal breakeven line
    ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6, zorder=1)

    for method in _METHOD_ORDER_PROFIT:
        trials = method_trajectories.get(method, [])
        if not trials:
            continue
        style = _PROFIT_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})

        all_curves = [profit_curve(t, production_cost) for t in trials]
        all_be = [breakeven_point(t, production_cost) for t in trials]

        # Average curve: interpolate onto common τ grid
        max_tau = max((c[-1][0] for c in all_curves if c), default=0)
        if max_tau == 0:
            continue
        tau_grid = np.arange(1, max_tau + 1)
        interp_curves = []
        for c in all_curves:
            if not c:
                interp_curves.append(np.zeros(len(tau_grid)))
                continue
            xs = np.array([p[0] for p in c])
            ys = np.array([p[1] for p in c])
            interp_curves.append(np.interp(tau_grid, xs, ys))

        mean_curve = np.mean(interp_curves, axis=0)
        std_curve  = np.std(interp_curves,  axis=0)

        plot_kw = {k: v for k, v in style.items() if k not in ("lw",)}
        ax.fill_between(tau_grid, mean_curve - std_curve, mean_curve + std_curve,
                        alpha=0.10, color=style["color"])
        ax.plot(tau_grid, mean_curve, lw=style["lw"], label=method, **plot_kw)

        # Annotate breakeven point
        valid_be = [b for b in all_be if b is not None]
        if valid_be:
            be_mean = float(np.mean(valid_be))
            be_idx = min(int(be_mean) - 1, len(mean_curve) - 1)
            ax.annotate(
                f"BE≈{be_mean:.0f}",
                xy=(be_mean, 0), xytext=(be_mean, -max(abs(mean_curve)) * 0.15),
                fontsize=7, color=style["color"], ha="center",
                arrowprops=dict(arrowstyle="->", color=style["color"], lw=0.8),
            )

    ax.set_xlabel("Number of acceptances (τ)")
    ax.set_ylabel("Cumulative profit (revenue − cost)")
    ax.set_title(f"Cumulative Profit (production cost c={production_cost})")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_breakeven_vs_cost(
    breakeven_data: Dict[str, Dict[float, Optional[float]]],
    save_path: str,
) -> None:
    """Bar chart of breakeven points at different production costs.

    x-axis: production cost c (grouped by c value)
    y-axis: "Breakeven point (τ)"

    Grouped bars: one group per c value, one bar per method.

    Args:
        breakeven_data: method_name → {c: mean_breakeven_tau or None}.
        save_path:      Full path for output PNG/PDF.
    """
    plt.rcParams.update(PLT_CFG)
    costs = sorted({c for vals in breakeven_data.values() for c in vals})
    methods = [m for m in _METHOD_ORDER_PROFIT if m in breakeven_data]

    n_methods = len(methods)
    n_costs   = len(costs)
    width = 0.8 / max(n_methods, 1)
    x = np.arange(n_costs)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, method in enumerate(methods):
        style = _PROFIT_STYLES.get(method, {"color": "gray"})
        vals = []
        for c in costs:
            v = breakeven_data[method].get(c)
            vals.append(v if v is not None else float("nan"))
        offset = (i - n_methods / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=style["color"], alpha=0.8, label=method)
        # Label "∞" for never-breakeven
        for bar, v in zip(bars, vals):
            if np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, 10, "∞",
                        ha="center", va="bottom", fontsize=9,
                        color=style["color"])

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in costs])
    ax.set_xlabel("Production cost c")
    ax.set_ylabel("Breakeven point (τ)")
    ax.set_title("Breakeven Point vs Production Cost")
    ax.legend(fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


def plot_profit_multi_cost(
    method_trajectories: Dict[str, List[List[dict]]],
    costs: List[float],
    save_path: str,
) -> None:
    """Three subplots: one per production cost c.

    Shows how production cost affects each method's profitability differently.
    Figure size: (16, 5).

    Args:
        method_trajectories: method_name → list of trajectory lists.
        costs:               List of c values, e.g. [0.1, 0.2, 0.3].
        save_path:           Full path for output PNG/PDF.
    """
    from src.evaluation.tc_evaluation import profit_curve

    plt.rcParams.update(PLT_CFG)
    fig, axes = plt.subplots(1, len(costs), figsize=(16, 5), sharey=False)
    if len(costs) == 1:
        axes = [axes]

    for ax, c in zip(axes, costs):
        ax.axhline(0, color="gray", lw=1.0, ls="--", alpha=0.6, zorder=1)
        for method in _METHOD_ORDER_PROFIT:
            trials = method_trajectories.get(method, [])
            if not trials:
                continue
            style = _PROFIT_STYLES.get(method, {"color": "gray", "lw": 1.5, "ls": "-"})

            all_curves = [profit_curve(t, c) for t in trials]
            max_tau = max((cr[-1][0] for cr in all_curves if cr), default=0)
            if max_tau == 0:
                continue
            tau_grid = np.arange(1, max_tau + 1)
            interp_curves = []
            for cr in all_curves:
                if not cr:
                    interp_curves.append(np.zeros(len(tau_grid)))
                    continue
                xs_m = np.array([p[0] for p in cr])
                ys_m = np.array([p[1] for p in cr])
                interp_curves.append(np.interp(tau_grid, xs_m, ys_m))

            mean_c = np.mean(interp_curves, axis=0)
            std_c  = np.std(interp_curves,  axis=0)
            plot_kw = {k: v for k, v in style.items() if k not in ("lw",)}
            ax.fill_between(tau_grid, mean_c - std_c, mean_c + std_c,
                            alpha=0.10, color=style["color"])
            ax.plot(tau_grid, mean_c, lw=style["lw"], label=method, **plot_kw)

        ax.set_title(f"c = {c}")
        ax.set_xlabel("Number of acceptances (τ)")
        ax.set_ylabel("Cumulative profit")

    # Shared legend on first axis
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(_METHOD_ORDER_PROFIT),
               fontsize=9, bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Cumulative Profit at Multiple Production Costs", fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── Master generation function ────────────────────────────────────────────────

def generate_all_tc_figures(
    tc_results_ff: Dict[str, Dict],
    figure_dir: str,
    curves_ff: Optional[Dict[str, List[List[float]]]] = None,
    tc_results_rice: Optional[Dict[str, Dict]] = None,
    tc_results_tc: Optional[Dict[str, Dict]] = None,
    profit_trajectories: Optional[Dict[str, List[List[dict]]]] = None,
    profit_costs: Optional[List[float]] = None,
) -> None:
    """Generate all TC figures from pre-computed result dicts.

    Args:
        tc_results_ff:        TC evaluation results for FF n=1000.
        figure_dir:           Output directory (created if needed).
        curves_ff:            Raw cum_rev curves (optional, for Fig 2).
        tc_results_rice:      TC results for Rice-Facebook (optional, for Fig 3).
        tc_results_tc:        TC results with TC-trained LSTM (optional, Fig 4).
        profit_trajectories:  method → list of step-dict trajectories (for profit figs).
        profit_costs:         Production cost values for profit analysis.
    """
    os.makedirs(figure_dir, exist_ok=True)
    print(f"Generating TC figures → {figure_dir}")

    # Fig 1: Revenue vs τ (main figure)
    tc_fig1_revenue_vs_tau(
        tc_results_ff, figure_dir, title="FF n=1000",
        tc_results_v2=tc_results_tc,
    )

    # Fig 2: Cumulative trajectories (requires raw curves)
    if curves_ff is not None:
        tc_fig2_cumulative_trajectories(curves_ff, figure_dir, title="FF n=1000")
    else:
        print("  [SKIP] TC-Fig 2: no raw curves provided")

    # Fig 3: Two-dataset comparison
    if tc_results_rice is not None:
        tc_fig3_revenue_vs_tau_two_datasets(tc_results_ff, tc_results_rice, figure_dir)
    else:
        print("  [SKIP] TC-Fig 3: no Rice-Facebook results")

    # Fig 4: Standard vs TC-trained LSTM (skipped — no TC-LSTM in Idea 2)
    tc_fig4_standard_vs_tc_lstm(tc_results_ff, tc_results_tc, figure_dir)

    # Profit figures (TC-PROFIT-1, TC-PROFIT-2, TC-PROFIT-3)
    if profit_trajectories is not None:
        costs = profit_costs or [0.1, 0.2, 0.3]

        # TC-PROFIT-1: single cost figure (c=0.2)
        plot_profit_vs_tau(
            profit_trajectories, production_cost=0.2,
            save_path=os.path.join(figure_dir, "tc_profit_c02.pdf"),
        )
        # TC-PROFIT-2: breakeven bar chart
        from src.evaluation.tc_evaluation import breakeven_point as _be
        be_data: Dict[str, Dict[float, Optional[float]]] = {}
        for method, trials in profit_trajectories.items():
            be_data[method] = {}
            for c in costs:
                bes = [_be(t, c) for t in trials]
                valid = [b for b in bes if b is not None]
                be_data[method][c] = float(np.mean(valid)) if valid else None
        plot_breakeven_vs_cost(
            be_data,
            save_path=os.path.join(figure_dir, "tc_profit_breakeven.pdf"),
        )
        # TC-PROFIT-3: 3-panel multi-cost
        plot_profit_multi_cost(
            profit_trajectories, costs=costs,
            save_path=os.path.join(figure_dir, "tc_profit_multi_cost.pdf"),
        )
    else:
        print("  [SKIP] Profit figures: no trajectories provided")

    print("TC figures done.")
