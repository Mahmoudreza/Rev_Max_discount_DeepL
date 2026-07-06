#!/usr/bin/env python
"""
experiments/paper_figures.py — Generate ALL paper figures and LaTeX tables.

ALL FIGURES use 4 methods:  IE-Strategy, Greedy-Discount, Rev-GNN-IM-RL, Rev-GNN-LSTM
TABLE 1: same 4  |  TABLE 2: same 4 across 5 networks

Datasets:
  • Forest Fire n=1000 (monotone + non-monotone) — Figure 1 (2 subplots)
  • Rice-Facebook n=443 (real network, zero-shot) — Figure 2

Data strategy:
  • Budget sweep Greedy + LSTM  → loaded from existing 4-method JSON caches
  • Budget sweep IE-Strategy    → always computed fresh (fast, ~3 min)
  • Budget sweep IM-RL          → computed with NEW checkpoint (force if --force-rerun-budget)
  • 20-seed comparison          → IM-RL re-evaluated with new checkpoint, rest from cache
  • Generalization              → IM-RL re-evaluated with new checkpoint, rest from cache

Usage:
    python experiments/paper_figures.py --config configs/experiments/rev_gnn_lstm.yaml
    python experiments/paper_figures.py --config configs/experiments/rev_gnn_lstm.yaml \\
        --force-rerun-budget   # recompute IE + IM-RL budget curves
"""

import argparse, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.paper_eval import (
    load_lstm, load_im,
    ac_greedy, ac_lstm, ac_im_rl,
    revenue_at_k,
    run_discount_trajectory,
)
from src.evaluation.baselines import ie_strategy_trajectory

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Paths ─────────────────────────────────────────────────────────────────────
LSTM_CKPT  = "results/checkpoints/rev_gnn_lstm.pt"
IMRL_CKPT  = "results/checkpoints/rev_gnn_im_rl.pt"
FIGURE_DIR = "results/figures"
LOG_DIR    = "results/logs"

# Existing 2-method caches (Greedy + LSTM only — fast to reuse)
CACHE_OLD = {
    "ff_mono":    f"{LOG_DIR}/paper_ff_monotone.json",
    "ff_nonmono": f"{LOG_DIR}/paper_ff_nonmonotone.json",
    "rice":       f"{LOG_DIR}/paper_rice_facebook.json",
    "comp20":     f"{LOG_DIR}/paper_20seed_comparison.json",
    "gen":        f"{LOG_DIR}/paper_generalization.json",
    "disc_traj":  f"{LOG_DIR}/paper_discount_traj.json",
}

# New 4-method budget sweep caches (IE + Greedy + IM-RL + LSTM)
CACHE4 = {
    "ff_mono4":    f"{LOG_DIR}/paper_ff_mono4.json",
    "ff_nonmono4": f"{LOG_DIR}/paper_ff_nonmono4.json",
    "rice4":       f"{LOG_DIR}/paper_rice4.json",
}

# ── Method order + styles ──────────────────────────────────────────────────────
METHOD_4 = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]

METHOD_STYLES = {
    "IE-Strategy":     {"color": "#1f77b4", "ls": "-",  "lw": 1.5, "marker": "o",  "ms": 4},
    "Greedy-Discount": {"color": "#d62728", "ls": "-",  "lw": 2.0, "marker": None, "ms": 5},
    "Rev-GNN-IM-RL":   {"color": "#ff7f0e", "ls": "--", "lw": 2.0, "marker": "^",  "ms": 4},
    "Rev-GNN-LSTM":    {"color": "#000000", "ls": "-",  "lw": 2.5, "marker": "s",  "ms": 4},
}

PLT_CFG = {
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 14,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 100, "savefig.dpi": 300, "savefig.bbox": "tight",
    "axes.spines.top": False, "axes.spines.right": False,
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _load(path):
    with open(path) as f:
        return json.load(f)

def _save(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f)

def _save_fig(fig, name):
    ensure_dir(FIGURE_DIR)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGURE_DIR}/{name}.{ext}")
    plt.close(fig)
    print(f"  [✓] {FIGURE_DIR}/{name}.{{pdf,png}}")


# ── IE-Strategy acceptance curve ─────────────────────────────────────────────
def ac_ie_strategy(graph, cfg):
    """IE-Strategy acceptance curve.

    Seeds (discount=1.0) contribute 0 revenue but still count toward |S|.
    Non-seeds contribute their estimated valuation as revenue.
    Every buyer appears in |S|, so cum_rev has length n.

    Returns:
        cum_rev   (List[float]): cumulative revenue indexed by |S|.
        discounts (List[float]): discount at each offer.
        accepted  (List[bool]):  True for everyone (seeds=free, rest=full val).
    """
    traj = ie_strategy_trajectory(graph, cfg)
    cum_rev, discounts, accepted = [], [], []
    total = 0.0
    for (_, disc, rev) in traj:
        total += rev
        cum_rev.append(total)
        discounts.append(disc)
        accepted.append(True)
    return cum_rev, discounts, accepted


# ── Build 4-method budget-sweep dict ─────────────────────────────────────────
def build_budget_sweep_4methods(
    lstm_pol, im_pol, graph_fn, cfg, device, k_values, n_seeds, out_path,
    cfg_override=None,
    existing_cache_path=None,
    force_imrl=False,
):
    """Build 4-method sweep (IE, Greedy, IM-RL, LSTM).

    Greedy + LSTM → loaded from existing cache if available.
    IE-Strategy  → always computed fresh (fast).
    IM-RL        → computed fresh (uses NEW checkpoint). Always recomputed when
                   force_imrl=True or no old cache contains valid IM-RL curves.
    """
    from omegaconf import OmegaConf
    if cfg_override:
        cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        for k_o, v_o in cfg_override.items():
            OmegaConf.update(cfg, k_o, v_o)

    curves_by_method = {m: [] for m in METHOD_4}

    greedy_curves_loaded, lstm_curves_loaded = [], []
    if existing_cache_path and os.path.exists(existing_cache_path):
        old = _load(existing_cache_path)
        greedy_curves_loaded = old.get("Greedy-Discount", {}).get("curves", [])
        lstm_curves_loaded   = old.get("Rev-GNN-LSTM",    {}).get("curves", [])
        print(f"    Loaded {len(greedy_curves_loaded)} Greedy + "
              f"{len(lstm_curves_loaded)} LSTM curves from cache")

    for seed in range(n_seeds):
        graph = graph_fn(seed)
        print(f"    seed={seed+1}/{n_seeds} (n={graph.number_of_nodes()})... ", end="", flush=True)
        t_s = time.time()

        # IE-Strategy (fast, ~1-2 sec)
        ie_c, _, _ = ac_ie_strategy(graph, cfg)
        curves_by_method["IE-Strategy"].append(ie_c)

        # Greedy from cache or fresh
        if seed < len(greedy_curves_loaded):
            curves_by_method["Greedy-Discount"].append(greedy_curves_loaded[seed])
        else:
            g_c, _, _ = ac_greedy(graph, cfg)
            curves_by_method["Greedy-Discount"].append(g_c)

        # IM-RL fresh (new checkpoint)
        im_c, _, _ = ac_im_rl(im_pol, graph, cfg, device)
        curves_by_method["Rev-GNN-IM-RL"].append(im_c)

        # LSTM from cache or fresh
        if seed < len(lstm_curves_loaded):
            curves_by_method["Rev-GNN-LSTM"].append(lstm_curves_loaded[seed])
        else:
            l_c, _, _ = ac_lstm(lstm_pol, graph, cfg, device)
            curves_by_method["Rev-GNN-LSTM"].append(l_c)

        print(f"{time.time()-t_s:.0f}s", flush=True)

    result = {"k_values": k_values, "n_seeds": n_seeds}
    for m in METHOD_4:
        revs = np.array([[revenue_at_k(c, k) for k in k_values]
                         for c in curves_by_method[m]])
        result[m] = {
            "mean":   revs.mean(axis=0).tolist(),
            "std":    revs.std(axis=0).tolist(),
            "curves": [c for c in curves_by_method[m]],
        }
    _save(result, out_path)
    return result


# ── Figure 1: Revenue vs K, Forest Fire ───────────────────────────────────────
def fig1_revenue_vs_K_ff(mono_data, nonmono_data):
    plt.rcParams.update(PLT_CFG)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _plot(ax, data, title, show_legend):
        kv = np.array(data["k_values"])
        n_pts = len(kv)
        for m in METHOD_4:
            if m not in data:
                continue
            st   = METHOD_STYLES[m]
            mean = np.array(data[m]["mean"])
            std  = np.array(data[m]["std"])
            mkw  = ({"marker": st["marker"], "markersize": st["ms"],
                     "markevery": max(1, n_pts // 6)}
                    if st["marker"] else {})
            ax.fill_between(kv, mean - std, mean + std,
                            alpha=0.10, color=st["color"])
            ax.plot(kv, mean, color=st["color"], ls=st["ls"], lw=st["lw"],
                    label=m.replace("Rev-GNN-", ""), **mkw)
        ax.set_title(title)
        ax.set_xlabel("Number of buyers in set S")
        ax.set_ylabel("Revenue")
        if show_legend:
            ax.legend(loc="upper left", framealpha=0.9)

    _plot(ax1, mono_data,    "(a) Monotone Concave",     show_legend=True)
    _plot(ax2, nonmono_data, "(b) Non-Monotone Concave", show_legend=False)
    fig.suptitle("Forest Fire Network (n=1000, p=0.37, pb=0.32)", fontsize=14, y=1.01)
    fig.tight_layout()
    _save_fig(fig, "fig1_revenue_vs_K_forest_fire")


# ── Figure 2: Revenue vs K, Rice Facebook ─────────────────────────────────────
def fig2_revenue_vs_K_rice(rice_data):
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    kv = np.array(rice_data["k_values"])
    n_pts = len(kv)
    for m in METHOD_4:
        if m not in rice_data:
            continue
        st   = METHOD_STYLES[m]
        mean = np.array(rice_data[m]["mean"])
        std  = np.array(rice_data[m]["std"])
        mkw  = ({"marker": st["marker"], "markersize": st["ms"],
                 "markevery": max(1, n_pts // 6)}
                if st["marker"] else {})
        ax.fill_between(kv, mean - std, mean + std, alpha=0.10, color=st["color"])
        ax.plot(kv, mean, color=st["color"], ls=st["ls"], lw=st["lw"],
                label=m.replace("Rev-GNN-", ""), **mkw)
    ax.set_xlabel("Number of buyers in set S")
    ax.set_ylabel("Revenue")
    ax.set_title("Facebook Network (n=443) — Zero-Shot Transfer")
    ax.legend(loc="upper left", framealpha=0.9)
    fig.tight_layout()
    _save_fig(fig, "fig2_revenue_vs_K_rice_facebook")


# ── Figure 3: Box Plot (4 methods, 20 seeds) ──────────────────────────────────
def fig3_boxplot(comp_data):
    plt.rcParams.update(PLT_CFG)
    fig, ax = plt.subplots(figsize=(9, 6))

    vals   = [comp_data[m]["all"]   for m in METHOD_4 if m in comp_data]
    colors = [METHOD_STYLES[m]["color"] for m in METHOD_4 if m in comp_data]
    labels = [m.replace("Rev-GNN-", "") for m in METHOD_4 if m in comp_data]
    positions = range(1, len(labels) + 1)

    bp = ax.boxplot(vals, positions=list(positions), widths=0.45,
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2.0))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color); patch.set_alpha(0.75)

    rng = np.random.default_rng(42)
    for i, (m_vals, color) in enumerate(zip(vals, colors)):
        jitter = rng.uniform(-0.15, 0.15, len(m_vals))
        ax.scatter([i + 1 + j for j in jitter], m_vals,
                   s=18, color=color, alpha=0.4, zorder=5)

    # Greedy median reference line
    if "Greedy-Discount" in comp_data:
        ax.axhline(np.median(comp_data["Greedy-Discount"]["all"]),
                   color="#d62728", ls="--", lw=1.2, alpha=0.6,
                   label="Greedy median")
        ax.legend(loc="upper left", fontsize=10)

    # Annotation on best box
    best_m = max([m for m in METHOD_4 if m in comp_data],
                 key=lambda m: comp_data[m]["mean"])
    best_pos = [m for m in METHOD_4 if m in comp_data].index(best_m) + 1
    best_val = max(comp_data[best_m]["all"])
    ax.annotate(
        f"{comp_data[best_m]['mean']:.1f}\n±{comp_data[best_m]['std']:.1f}",
        xy=(best_pos, best_val),
        xytext=(best_pos + 0.3, best_val * 0.99),
        fontsize=9, ha="left",
        bbox=dict(boxstyle="round,pad=0.25", fc="lightyellow", ec="gray"),
    )

    ax.set_xticks(list(positions))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Revenue")
    ax.set_title("Revenue Comparison (20 seeds, FF n=1000)")
    fig.tight_layout()
    _save_fig(fig, "fig3_boxplot")


# ── Figure 4: Generalization (4 methods, 5 networks) ─────────────────────────
def fig4_generalization(gen_data):
    plt.rcParams.update(PLT_CFG)
    NETS  = list(gen_data.keys())
    x     = np.arange(len(NETS))
    xlabels = [n.replace("Modular FF", "Modular\nFF")
                .replace(" n=", "\nn=") for n in NETS]

    fig, (ax_main, ax_delta) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Bar plot — absolute revenue for all 4 methods
    n_methods = sum(1 for m in METHOD_4 if
                    any(m in gen_data[n] for n in NETS))
    bar_w = 0.18
    offsets = np.linspace(-(n_methods - 1) / 2 * bar_w,
                          (n_methods - 1) / 2 * bar_w, n_methods)
    method_list = [m for m in METHOD_4 if any(m in gen_data[n] for n in NETS)]

    for i, m in enumerate(method_list):
        means = np.array([gen_data[n].get(m, {}).get("mean", 0) for n in NETS])
        stds  = np.array([gen_data[n].get(m, {}).get("std",  0) for n in NETS])
        st    = METHOD_STYLES[m]
        ax_main.bar(x + offsets[i], means, width=bar_w,
                    color=st["color"], alpha=0.8,
                    yerr=stds, capsize=3,
                    label=m.replace("Rev-GNN-", ""),
                    hatch="//" if st["ls"] == "--" else None)

    ax_main.set_xticks(x); ax_main.set_xticklabels(xlabels, fontsize=9)
    ax_main.set_ylabel("Revenue"); ax_main.set_title("Absolute Revenue (5 seeds)")
    ax_main.legend(fontsize=9)

    # Delta plot — LSTM vs Greedy
    g_means = np.array([gen_data[n].get("Greedy-Discount", {}).get("mean", 0)
                        for n in NETS])
    l_means = np.array([gen_data[n].get("Rev-GNN-LSTM",    {}).get("mean", 0)
                        for n in NETS])
    deltas  = (l_means - g_means) / (g_means + 1e-9) * 100
    colors_d = ["#000000" if d > 0 else "#d62728" for d in deltas]
    bars = ax_delta.bar(x, deltas, color=colors_d, alpha=0.8, edgecolor="white")
    for bar, d in zip(bars, deltas):
        ypos = bar.get_height() + 0.5 if d >= 0 else bar.get_height() - 1.8
        ax_delta.text(bar.get_x() + bar.get_width() / 2, ypos,
                      f"{d:+.1f}%", ha="center", fontsize=9, fontweight="bold")
    ax_delta.axhline(0, color="gray", lw=0.8)
    ax_delta.set_xticks(x); ax_delta.set_xticklabels(xlabels, fontsize=9)
    ax_delta.set_ylabel("Δ% vs Greedy-Discount")
    ax_delta.set_title("LSTM Improvement over Greedy")

    fig.suptitle("Zero-Shot Generalization (trained on FF n≤440)", fontsize=14)
    fig.tight_layout()
    _save_fig(fig, "fig4_generalization")


# ── Figure 5: Discount Trajectory (Greedy vs IM-RL vs LSTM) ─────────────────
def fig5_discount_trajectory(traj_data):
    plt.rcParams.update(PLT_CFG)
    METHODS_TRAJ = ["Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    fig, ax = plt.subplots(figsize=(12, 5))

    for m in METHODS_TRAJ:
        if m not in traj_data:
            continue
        st    = METHOD_STYLES[m]
        discs = np.array(traj_data[m]["discounts"])
        steps = np.arange(1, len(discs) + 1)
        ax.scatter(steps, discs, color=st["color"], alpha=0.08, s=5)
        win = 50
        if len(discs) >= win:
            rolled = np.convolve(discs, np.ones(win) / win, mode="valid")
            ax.plot(np.arange(win, len(discs) + 1), rolled,
                    color=st["color"], lw=2.5, ls=st["ls"],
                    label=m.replace("Rev-GNN-", ""))

    ax.set_xlabel("Offer step")
    ax.set_ylabel("Discount (0=full price, 1=free)")
    ax.set_title("Discount Strategy Comparison (FF n=1000, seed=42)")
    ax.legend()
    fig.tight_layout()
    _save_fig(fig, "fig5_discount_trajectory")


# ── LaTeX Table 1: 4 methods ──────────────────────────────────────────────────
def table1_latex(comp_data):
    TEX = {
        "IE-Strategy":    "IE-Strategy",
        "Greedy-Discount":"Greedy-Discount",
        "Rev-GNN-IM-RL":  "Rev-GNN-IM-RL (ours)",
        "Rev-GNN-LSTM":   "Rev-GNN-LSTM (ours)",
    }
    TYPE = {
        "IE-Strategy":    "IM + myopic pricing",
        "Greedy-Discount":"Influence-tier pricing",
        "Rev-GNN-IM-RL":  "Joint GNN+RL",
        "Rev-GNN-LSTM":   "Joint + LSTM memory",
    }

    avail = [m for m in METHOD_4 if m in comp_data]
    best_mean = max(comp_data[m]["mean"] for m in avail)

    def fmt_row(m):
        if m not in comp_data:
            return f"  {TEX[m]} & {TYPE[m]} & --- & --- \\\\"
        d = comp_data[m]
        mv, sv = d["mean"], d["std"]
        dp = d.get("delta_pct", 0.0)
        rev_s   = f"${mv:.1f} \\pm {sv:.1f}$"
        delt_s  = "---" if m == "Greedy-Discount" else f"${dp:+.1f}\\%$"
        if abs(mv - best_mean) < 0.1:
            rev_s  = f"\\textbf{{{mv:.1f}}} $\\pm$ {sv:.1f}"
            delt_s = f"\\textbf{{{dp:+.1f}\\%}}" if m != "Greedy-Discount" else "---"
        return f"  {TEX[m]} & {TYPE[m]} & {rev_s} & {delt_s} \\\\"

    rows = [fmt_row(m) for m in METHOD_4]
    tex = (
        "\\begin{table}[t]\n"
        "\\caption{Revenue comparison on Forest Fire ($n=1000$, monotone concave). "
        "20 seeds. Best \\textbf{bold}.}\n"
        "\\label{tab:main_results}\n"
        "\\centering\\small\n"
        "\\begin{tabular}{llcc}\n\\toprule\n"
        "Method & Type & Revenue (mean $\\pm$ std) & $\\Delta$ vs Greedy \\\\\n"
        "\\midrule\n"
        + rows[0] + "\n" + rows[1] + "\n\\midrule\n"
        + "\n".join(rows[2:]) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    path = f"{LOG_DIR}/paper_table1.tex"
    ensure_dir(LOG_DIR)
    with open(path, "w") as f:
        f.write(tex)
    print(f"  [✓] {path}")
    return tex


# ── LaTeX Table 2: Generalization 4×5 ────────────────────────────────────────
def table2_latex(gen_data):
    NETS = list(gen_data.keys())
    TEX  = {"IE-Strategy":"IE-Strategy","Greedy-Discount":"Greedy-Discount",
            "Rev-GNN-IM-RL":"Rev-GNN-IM-RL","Rev-GNN-LSTM":"Rev-GNN-LSTM"}
    net_headers = " & ".join(
        n.replace("Rice-FB n=443","Rice-FB").replace(" n=","\\\\n=")
        for n in NETS
    )

    def fmt(n, method):
        if n not in gen_data or method not in gen_data[n]:
            return "---"
        v  = gen_data[n][method]
        mv = v.get("mean", float("nan"))
        if mv != mv:
            return "---"
        col_best = max(
            (gen_data[n].get(mm, {}).get("mean", 0) for mm in METHOD_4),
            default=0
        )
        s = f"{mv:.1f}"
        return f"\\textbf{{{s}}}" if abs(mv - col_best) < 0.05 else s

    rows = []
    for method in METHOD_4:
        row = (TEX.get(method, method) + " & " +
               " & ".join(fmt(n, method) for n in NETS) + " \\\\")
        rows.append(row)

    g_strs = []
    for n in NETS:
        gm = gen_data[n].get("Greedy-Discount", {}).get("mean", None)
        lm = gen_data[n].get("Rev-GNN-LSTM",    {}).get("mean", None)
        if gm and lm and gm > 0:
            g_strs.append(f"${(lm - gm) / gm * 100:+.1f}\\%$")
        else:
            g_strs.append("---")

    col_spec = "l" + "c" * len(NETS)
    tex = (
        "\\begin{table}[t]\n"
        "\\caption{Zero-shot generalization. Trained on FF $n\\le440$. Best \\textbf{bold}.}\n"
        "\\label{tab:generalization}\n"
        "\\centering\\small\n"
        "\\begin{tabular}{" + col_spec + "}\n\\toprule\n"
        " & " + net_headers + " \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n\\midrule\n"
        "$\\Delta\\%$ LSTM vs Greedy & " + " & ".join(g_strs) + " \\\\\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    path = f"{LOG_DIR}/paper_table2.tex"
    ensure_dir(LOG_DIR)
    with open(path, "w") as f:
        f.write(tex)
    print(f"  [✓] {path}")
    return tex


# ── Evaluate new IM-RL on N seeds ─────────────────────────────────────────────
def eval_imrl_n_seeds(im_pol, graph_fn, cfg, device, n_seeds, greedy_mean):
    """Run IM-RL on n_seeds graphs. Returns {mean, std, all, delta_pct}."""
    revs = []
    for seed in range(n_seeds):
        graph = graph_fn(seed)
        c, _, _ = ac_im_rl(im_pol, graph, cfg, device)
        revs.append(float(revenue_at_k(c, len(c)) if c else 0.0))
    arr = np.array(revs)
    return {
        "mean":      float(arr.mean()),
        "std":       float(arr.std()),
        "all":       revs,
        "delta_pct": float((arr.mean() - greedy_mean) / greedy_mean * 100),
    }


def eval_imrl_generalization(im_pol, cfg, device, n_seeds=1):
    """Eval IM-RL on 5 generalization networks."""
    from src.env.graph_generators import generate_modular_forest_fire
    p, pb = cfg.graph.p, cfg.graph.pb
    NETS = {
        "FF n=500":      lambda s: generate_forest_fire(500,  p, pb, seed=s),
        "FF n=1000":     lambda s: generate_forest_fire(1000, p, pb, seed=s),
        "FF n=2000":     lambda s: generate_forest_fire(2000, p, pb, seed=s),
        "Modular FF":    lambda s: generate_modular_forest_fire([200, 300, 500], p, pb, 0.01, seed=s),
        "Rice-FB n=443": lambda _: load_rice_facebook(data_dir="data/raw"),
    }
    result = {}
    for net_name, graph_fn in NETS.items():
        revs = []
        for seed in range(n_seeds):
            try:
                g   = graph_fn(seed)
                c, _, _ = ac_im_rl(im_pol, g, cfg, device)
                revs.append(float(revenue_at_k(c, len(c)) if c else 0.0))
            except Exception as e:
                print(f"    [SKIP] {net_name} seed={seed}: {e}")
                revs.append(0.0)
        arr = np.array(revs)
        result[net_name] = {"Rev-GNN-IM-RL": {
            "mean": float(arr.mean()), "std": float(arr.std()), "all": revs,
        }}
    return result


# ── Fresh-compute helpers (used when old caches are absent) ──────────────────
def eval_all_methods_comp20(lstm_pol, im_pol, graph_fn, cfg, device, n_seeds):
    """Compute 20-seed comparison for ALL 4 methods from scratch (no cache needed)."""
    all_revs = {m: [] for m in METHOD_4}
    for seed in range(n_seeds):
        g = graph_fn(seed)
        print(f"    seed {seed+1}/{n_seeds}...", flush=True)
        ie_c, _, _ = ac_ie_strategy(g, cfg)
        gr_c, _, _ = ac_greedy(g, cfg)          # ac_greedy: (graph, cfg) — no device
        im_c, _, _ = ac_im_rl(im_pol, g, cfg, device)
        ls_c, _, _ = ac_lstm(lstm_pol, g, cfg, device)
        for method, c in zip(METHOD_4, [ie_c, gr_c, im_c, ls_c]):
            all_revs[method].append(float(revenue_at_k(c, len(c)) if c else 0.0))
    greedy_m = float(np.mean(all_revs["Greedy-Discount"]))
    result = {}
    for method in METHOD_4:
        arr = np.array(all_revs[method])
        result[method] = {
            "mean":      float(arr.mean()),
            "std":       float(arr.std()),
            "all":       all_revs[method],
            "delta_pct": float((arr.mean() - greedy_m) / greedy_m * 100) if greedy_m else 0.0,
        }
    return result


def eval_all_methods_generalization(lstm_pol, im_pol, cfg, device, n_seeds):
    """Compute generalization table for ALL 4 methods from scratch (no cache needed)."""
    from src.env.graph_generators import generate_modular_forest_fire
    p, pb = cfg.graph.p, cfg.graph.pb
    NETS = {
        "FF n=500":      lambda s: generate_forest_fire(500,  p, pb, seed=s),
        "FF n=1000":     lambda s: generate_forest_fire(1000, p, pb, seed=s),
        "FF n=2000":     lambda s: generate_forest_fire(2000, p, pb, seed=s),
        "Modular FF":    lambda s: generate_modular_forest_fire([200, 300, 500], p, pb, 0.01, seed=s),
        "Rice-FB n=443": lambda _: load_rice_facebook(data_dir="data/raw"),
    }
    result = {}
    for net_name, gfn in NETS.items():
        print(f"    {net_name} ({n_seeds} seed(s))...", flush=True)
        all_revs = {m: [] for m in METHOD_4}
        for seed in range(n_seeds):
            try:
                g = gfn(seed)
                ie_c, _, _ = ac_ie_strategy(g, cfg)
                gr_c, _, _ = ac_greedy(g, cfg)              # ac_greedy: (graph, cfg)
                im_c, _, _ = ac_im_rl(im_pol, g, cfg, device)
                ls_c, _, _ = ac_lstm(lstm_pol, g, cfg, device)
                for method, c in zip(METHOD_4, [ie_c, gr_c, im_c, ls_c]):
                    all_revs[method].append(float(revenue_at_k(c, len(c)) if c else 0.0))
            except Exception as e:
                print(f"      [SKIP] {net_name} seed={seed}: {e}")
                for method in METHOD_4:
                    all_revs[method].append(0.0)
        net_entry = {}
        for method in METHOD_4:
            arr = np.array(all_revs[method])
            net_entry[method] = {
                "mean": float(arr.mean()), "std": float(arr.std()), "all": all_revs[method],
            }
        result[net_name] = net_entry
    return result


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate paper figures (4 methods)")
    parser.add_argument("--config",            default="configs/experiments/rev_gnn_lstm.yaml")
    parser.add_argument("--n-seeds-budget",    type=int, default=5)
    parser.add_argument("--n-seeds-comp",      type=int, default=20)
    parser.add_argument("--n-seeds-gen",       type=int, default=1)
    parser.add_argument("--force-rerun-budget", action="store_true",
                        help="Recompute IE + IM-RL budget sweeps even if cached")
    args = parser.parse_args()

    t0 = time.time()
    ensure_dir(FIGURE_DIR); ensure_dir(LOG_DIR)
    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = torch.device("cpu")

    print("=" * 60)
    print("paper_figures.py — 4 methods: IE / Greedy / IM-RL / LSTM")
    print("=" * 60)
    print("Loading policy checkpoints...")
    lstm_pol = load_lstm(LSTM_CKPT, cfg, device)
    im_pol   = load_im(IMRL_CKPT,   cfg, device)
    print(f"  LSTM:  {sum(p.numel() for p in lstm_pol.parameters()):,} params")
    print(f"  IM-RL: {sum(p.numel() for p in im_pol.parameters()):,} params")

    p, pb  = cfg.graph.p, cfg.graph.pb
    FF1000 = lambda s: generate_forest_fire(1000, p, pb, seed=s)
    K_FF   = [50, 100, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000]
    K_RF   = [20, 50, 80, 100, 130, 160, 200, 250, 300, 350, 400]

    # ── Budget sweeps ─────────────────────────────────────────────────────────
    def _run_or_load(cache_key, label, graph_fn, k_vals, old_key, cfg_override=None):
        path = CACHE4[cache_key]
        if not args.force_rerun_budget and os.path.exists(path):
            print(f"  [cache] {label} — {path}")
            return _load(path)
        print(f"  [run]   {label}  (n_seeds={args.n_seeds_budget}) ...")
        t1 = time.time()
        res = build_budget_sweep_4methods(
            lstm_pol, im_pol, graph_fn, cfg, device, k_vals,
            n_seeds=args.n_seeds_budget, out_path=path,
            cfg_override=cfg_override,
            existing_cache_path=CACHE_OLD.get(old_key),
        )
        print(f"    → done in {(time.time()-t1)/60:.1f} min  saved: {path}")
        return res

    print(f"\n[1/5] FF n=1000 monotone budget sweep...")
    ff_mono = _run_or_load("ff_mono4",    "FF mono",    FF1000, K_FF, "ff_mono")

    print(f"[2/5] FF n=1000 non-monotone budget sweep...")
    ff_nm   = _run_or_load("ff_nonmono4", "FF nonmono", FF1000, K_FF, "ff_nonmono",
                            cfg_override={"influence_model": "non_monotone"})

    print(f"[3/5] Rice-Facebook budget sweep...")
    rice_data = None
    try:
        rf = load_rice_facebook(data_dir="data/raw")
        n_rf = rf.number_of_nodes()
        k_rf = [k for k in K_RF if k <= n_rf]
        rice_data = _run_or_load("rice4", "Rice-FB", lambda s: rf, k_rf, "rice")
    except FileNotFoundError as e:
        print(f"  [SKIP] Rice-Facebook: {e}")

    # ── 20-seed comparison ────────────────────────────────────────────────────
    print(f"[4/5] 20-seed comparison (all 4 methods, FF n=1000)...")
    comp20_path = CACHE_OLD["comp20"]
    if os.path.exists(comp20_path):
        print(f"  Loading existing cache + re-evaluating IM-RL...")
        comp20 = _load(comp20_path)
        greedy_mean = comp20["Greedy-Discount"]["mean"]
        imrl_new = eval_imrl_n_seeds(im_pol, FF1000, cfg, device,
                                     args.n_seeds_comp, greedy_mean)
        comp20["Rev-GNN-IM-RL"] = imrl_new
    else:
        print(f"  No cache — computing ALL 4 methods fresh ({args.n_seeds_comp} seeds)...")
        comp20 = eval_all_methods_comp20(lstm_pol, im_pol, FF1000, cfg, device, args.n_seeds_comp)
    greedy_mean = comp20["Greedy-Discount"]["mean"]
    for m in comp20:
        if m != "Greedy-Discount" and isinstance(comp20[m], dict) and "mean" in comp20[m]:
            comp20[m]["delta_pct"] = (comp20[m]["mean"] - greedy_mean) / greedy_mean * 100
    _save(comp20, f"{LOG_DIR}/paper_comp20_updated.json")
    imrl_d = comp20.get("Rev-GNN-IM-RL", {})
    lstm_d = comp20.get("Rev-GNN-LSTM", {})
    print(f"  IM-RL: {imrl_d.get('mean',0):.2f} ± {imrl_d.get('std',0):.2f}  "
          f"(Δ{imrl_d.get('delta_pct',0):+.1f}% vs Greedy {greedy_mean:.2f})")
    print(f"  LSTM:  {lstm_d.get('mean',0):.2f} ± {lstm_d.get('std',0):.2f}  "
          f"(Δ{lstm_d.get('delta_pct',0):+.1f}% vs Greedy {greedy_mean:.2f})")

    # ── Generalization ────────────────────────────────────────────────────────
    print(f"[5/5] Generalization across 5 networks...")
    gen_path = CACHE_OLD["gen"]
    if os.path.exists(gen_path):
        print(f"  Loading existing cache + re-evaluating IM-RL...")
        gen = _load(gen_path)
        imrl_gen_new = eval_imrl_generalization(im_pol, cfg, device,
                                                n_seeds=args.n_seeds_gen)
        for net_name in gen:
            if net_name in imrl_gen_new:
                gen[net_name]["Rev-GNN-IM-RL"] = imrl_gen_new[net_name]["Rev-GNN-IM-RL"]
    else:
        print(f"  No cache — computing ALL 4 methods fresh ({args.n_seeds_gen} seeds)...")
        gen = eval_all_methods_generalization(lstm_pol, im_pol, cfg, device, args.n_seeds_gen)
    _save(gen, f"{LOG_DIR}/paper_gen_updated.json")
    print(f"  Generalization updated across {len(gen)} networks.")

    # ── Discount trajectory ───────────────────────────────────────────────────
    disc_traj = None
    if os.path.exists(CACHE_OLD["disc_traj"]):
        disc_traj = _load(CACHE_OLD["disc_traj"])
    else:
        print("Computing discount trajectory (seed=42)...")
        g42 = generate_forest_fire(1000, p, pb, seed=42)
        disc_traj = run_discount_trajectory(lstm_pol, im_pol, g42, cfg, device)
        _save(disc_traj, CACHE_OLD["disc_traj"])

    # ── Generate figures ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Generating figures...")
    fig1_revenue_vs_K_ff(ff_mono, ff_nm)
    if rice_data:
        fig2_revenue_vs_K_rice(rice_data)
    else:
        print("  [SKIP] Figure 2 — Rice-Facebook not available")
    fig3_boxplot(comp20)
    fig4_generalization(gen)
    if disc_traj:
        fig5_discount_trajectory(disc_traj)

    # ── Generate tables ───────────────────────────────────────────────────────
    print("Generating LaTeX tables...")
    table1_latex(comp20)
    table2_latex(gen)

    # ── Summary ───────────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"All done in {elapsed/60:.1f} min")
    print(f"\nKey results (20 seeds, FF n=1000, monotone):")
    for m in METHOD_4:
        if m in comp20:
            d  = comp20[m]
            dp = d.get("delta_pct", 0.0)
            mark = " ★" if abs(d["mean"] - max(comp20[mm]["mean"] for mm in METHOD_4
                                               if mm in comp20)) < 0.1 else ""
            print(f"  {m:<22}: {d['mean']:>7.2f} ± {d['std']:>5.2f}  "
                  f"({dp:+.1f}% vs Greedy){mark}")
    print(f"\nOutputs:")
    print(f"  Figures → {FIGURE_DIR}/fig{{1-5}}.{{pdf,png}}")
    print(f"  Tables  → {LOG_DIR}/paper_table{{1,2}}.tex")


if __name__ == "__main__":
    main()
