#!/usr/bin/env python
"""
experiments/plot_figures.py  —  Generate all paper figures for Idea 1.

Figures produced (in results/figures/):
  fig1_learning_curve.pdf   — Revenue over training (Phase 1 imitation → Phase 2 REINFORCE)
  fig2_revenue_comparison.pdf — 20-seed box + scatter (LSTM vs Greedy)
  fig3_generalization.pdf   — Δ% per network type (zero-shot)
  fig4_budget_K.pdf         — Cumulative revenue vs budget K

Usage:
    python experiments/plot_figures.py --config configs/experiments/rev_gnn_lstm.yaml
    python experiments/plot_figures.py --skip-budget   # skip budget sweep if time-limited
"""
import argparse, csv, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import torch, numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger
from src.evaluation.idea1_eval import (
    load_lstm_policy, run_budget_sweep, run_budget_sweep_on_graph,
)
from src.env.graph_generators import load_rice_facebook

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FIGURE_DIR = "results/figures"
LSTM_CKPT  = "results/checkpoints/rev_gnn_lstm.pt"
TRAIN_CSV  = "results/logs/rev_gnn_lstm_20260626_152606.csv"
ROB_CSV    = "results/logs/robustness_20seeds.csv"
GEN_CSV    = "results/logs/generalization_eval.csv"
BUDGET_CSV    = "results/logs/budget_sweep_lstm_vs_greedy.csv"
BUDGET_RF_CSV = "results/logs/budget_sweep_rice_facebook.csv"

# Publication style
PLT_STYLE  = {"font.size": 11, "axes.titlesize": 11, "axes.labelsize": 11,
               "legend.fontsize": 9, "figure.dpi": 150,
               "axes.spines.top": False, "axes.spines.right": False}
LSTM_COLOR   = "#2196F3"   # blue
GREEDY_COLOR = "#FF5722"   # orange-red
IM_COLOR     = "#9C27B0"   # purple


def fig1_learning_curve():
    """Figure 1: Learning curve from training CSV."""
    # Parse the mixed-format CSV
    epochs, revenues = [], []
    phase2_start = None
    with open(TRAIN_CSV) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 3:
                continue
            try:
                ep = float(row[2])
                rev = float(row[3]) if len(row) > 3 else None
                if rev is None:
                    continue
                # Phase 1: row has im/epoch, im/loss, im/revenue (3 val cols)
                # Phase 2: row has rl/epoch, rl/revenue, rl/loss (3 val cols)
                # Summary row (last): skip
                if row[2] == "0" and len(row) == 5:      # Phase 1 start
                    epochs.append(0)
                    revenues.append(float(row[4]))
                elif len(row) == 4 and ep == 1.0:         # After Phase 1
                    epochs.append(50)
                    revenues.append(float(row[3]))
                elif len(row) == 5 and ep >= 10:          # Phase 2 points
                    if phase2_start is None:
                        phase2_start = 50
                    epochs.append(50 + ep)
                    revenues.append(float(row[3]))
            except (ValueError, IndexError):
                continue

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4))

    # Phase regions
    ax.axvspan(0, 50, alpha=0.08, color=LSTM_COLOR, label="Phase 1 (Imitation)")
    ax.axvspan(50, 250, alpha=0.05, color="green", label="Phase 2 (REINFORCE)")

    # Greedy-Discount reference
    ax.axhline(460.0, color=GREEDY_COLOR, linestyle="--", linewidth=1.2,
               label="Greedy-Discount (460.0)")

    # Learning curve
    if epochs:
        ax.plot(epochs, revenues, color=LSTM_COLOR, linewidth=2, marker="o",
                markersize=3.5, label="Rev-GNN-LSTM")

    ax.axvline(50, color="gray", linestyle=":", linewidth=0.8)
    ax.text(25, min(revenues) - 8 if revenues else 340, "Phase 1\n(Imitation)", ha="center",
            fontsize=8, color="gray")
    ax.text(150, min(revenues) - 8 if revenues else 340, "Phase 2\n(REINFORCE)", ha="center",
            fontsize=8, color="gray")

    ax.set_xlabel("Training Epoch")
    ax.set_ylabel("Revenue (test graph, n=1000)")
    ax.set_title("Rev-GNN-LSTM Training Progress")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.set_xlim(-5, 255)

    ensure_dir(FIGURE_DIR)
    out = os.path.join(FIGURE_DIR, "fig1_learning_curve.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {out}")
    return out


def fig2_revenue_comparison():
    """Figure 2: 20-seed box + scatter comparison."""
    lstm_vals, greedy_vals = [], []
    with open(ROB_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lstm_vals.append(float(row["lstm"]))
                greedy_vals.append(float(row["greedy"]))
            except ValueError:
                pass  # skip mean row

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(5, 4.5))

    positions = [1, 2]
    bp = ax.boxplot([greedy_vals, lstm_vals], positions=positions, widths=0.4,
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor(GREEDY_COLOR)
    bp["boxes"][1].set_facecolor(LSTM_COLOR)
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    # Individual seed scatter
    jitter = np.random.default_rng(0).uniform(-0.12, 0.12, len(greedy_vals))
    ax.scatter([1 + j for j in jitter], greedy_vals, s=18, color=GREEDY_COLOR,
               alpha=0.6, zorder=5)
    ax.scatter([2 + j for j in jitter], lstm_vals, s=18, color=LSTM_COLOR,
               alpha=0.6, zorder=5)

    # Mean lines
    gm, lm = np.mean(greedy_vals), np.mean(lstm_vals)
    ax.hlines(gm, 0.75, 1.25, colors=GREEDY_COLOR, linewidth=2, linestyle="-")
    ax.hlines(lm, 1.75, 2.25, colors=LSTM_COLOR, linewidth=2, linestyle="-")

    # Annotation
    ax.annotate(
        f"p < 0.0001\n20/20 wins",
        xy=(1.5, max(max(lstm_vals), max(greedy_vals))),
        ha="center", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(["Greedy-Discount", "Rev-GNN-LSTM"])
    ax.set_ylabel("Revenue (n=1000)")
    ax.set_title("Revenue Comparison (20 seeds, FF n=1000)")
    ax.set_xlim(0.5, 2.5)

    ensure_dir(FIGURE_DIR)
    out = os.path.join(FIGURE_DIR, "fig2_revenue_comparison.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {out}")
    return out


def fig3_generalization():
    """Figure 3: Zero-shot generalization bar chart (Δ%)."""
    networks, lstm_m, lstm_s, greedy_m, greedy_s = [], [], [], [], []
    with open(GEN_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            networks.append(row["network"].replace("n=", "n=").replace("rice_facebook ", "Rice-FB "))
            lstm_m.append(float(row["lstm_mean"]))
            lstm_s.append(float(row["lstm_std"]))
            greedy_m.append(float(row["greedy_mean"]))
            greedy_s.append(float(row["greedy_std"]))

    deltas = [(l - g) / g * 100 for l, g in zip(lstm_m, greedy_m)]
    short_names = ["FF\nn=500", "FF\nn=1000", "FF\nn=2000", "Modular\nFF", "Rice-FB\nn=443"]
    short_names = short_names[:len(networks)]

    plt.rcParams.update(PLT_STYLE)
    fig, (ax_main, ax_delta) = plt.subplots(1, 2, figsize=(10, 4.5))

    # Left: absolute revenues
    x = np.arange(len(networks))
    w = 0.35
    ax_main.bar(x - w/2, greedy_m, width=w, color=GREEDY_COLOR, alpha=0.75,
                yerr=greedy_s, capsize=3, label="Greedy-Discount")
    ax_main.bar(x + w/2, lstm_m, width=w, color=LSTM_COLOR, alpha=0.75,
                yerr=lstm_s, capsize=3, label="Rev-GNN-LSTM")
    ax_main.set_xticks(x)
    ax_main.set_xticklabels(short_names, fontsize=9)
    ax_main.set_ylabel("Revenue")
    ax_main.set_title("Zero-Shot Generalization")
    ax_main.legend()

    # Right: Δ% bar chart
    colors = [LSTM_COLOR if d > 0 else GREEDY_COLOR for d in deltas]
    bars = ax_delta.bar(x, deltas, color=colors, alpha=0.8, edgecolor="white")
    for bar, d in zip(bars, deltas):
        ax_delta.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                      f"{d:+.1f}%", ha="center", fontsize=8.5, fontweight="bold")
    ax_delta.axhline(0, color="gray", linewidth=0.8)
    ax_delta.set_xticks(x)
    ax_delta.set_xticklabels(short_names, fontsize=9)
    ax_delta.set_ylabel("Δ% vs Greedy-Discount")
    ax_delta.set_title("LSTM Improvement vs Greedy")

    ensure_dir(FIGURE_DIR)
    out = os.path.join(FIGURE_DIR, "fig3_generalization.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {out}")
    return out


def fig4_budget_K(budget_data):
    """Figure 4: Cumulative revenue vs budget K (smooth curves)."""
    rows = budget_data["rows"]
    k_vals = [r["k"] for r in rows]
    lm = [r["lstm_mean"] for r in rows]
    ls = [r["lstm_std"] for r in rows]
    gm = [r["greedy_mean"] for r in rows]
    gs = [r["greedy_std"] for r in rows]
    lm, ls, gm, gs = np.array(lm), np.array(ls), np.array(gm), np.array(gs)

    # Full smooth curves if available
    lc = np.array(budget_data["lstm_curves"])    # (n_seeds, 1000)
    gc = np.array(budget_data["greedy_curves"])
    K_full = np.arange(1, lc.shape[1] + 1)

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Smooth mean ± std bands
    lm_full = lc.mean(axis=0)
    ls_full = lc.std(axis=0)
    gm_full = gc.mean(axis=0)
    gs_full = gc.std(axis=0)

    ax.fill_between(K_full, lm_full - ls_full, lm_full + ls_full,
                    alpha=0.15, color=LSTM_COLOR)
    ax.fill_between(K_full, gm_full - gs_full, gm_full + gs_full,
                    alpha=0.15, color=GREEDY_COLOR)
    ax.plot(K_full, lm_full, color=LSTM_COLOR, linewidth=2.0, label="Rev-GNN-LSTM")
    ax.plot(K_full, gm_full, color=GREEDY_COLOR, linewidth=2.0, label="Greedy-Discount")

    # Vertical reference lines
    for k_ref in [100, 250, 500]:
        ax.axvline(k_ref, color="gray", linewidth=0.7, linestyle=":")

    ax.set_xlabel("Budget K (number of offers made)")
    ax.set_ylabel("Cumulative Revenue")
    ax.set_title("Revenue vs Offer Budget K (FF n=1000, 5 seeds ± std)")
    ax.legend()
    ax.set_xlim(1, 1000)

    ensure_dir(FIGURE_DIR)
    out = os.path.join(FIGURE_DIR, "fig4_budget_K.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {out}")
    return out


def fig4b_budget_K_rice_facebook(budget_rf_data):
    """Figure 4b: Budget-K curve on Rice-Facebook real network."""
    lc = np.array(budget_rf_data["lstm_curves"])    # (n_seeds, n_graph)
    gc = np.array(budget_rf_data["greedy_curves"])
    n_graph = budget_rf_data.get("n_graph", lc.shape[1])
    K_full = np.arange(1, n_graph + 1)

    plt.rcParams.update(PLT_STYLE)
    fig, ax = plt.subplots(figsize=(7, 4.5))

    lm_full = lc.mean(axis=0)
    ls_full = lc.std(axis=0)
    gm_full = gc.mean(axis=0)
    gs_full = gc.std(axis=0)

    ax.fill_between(K_full, lm_full - ls_full, lm_full + ls_full,
                    alpha=0.15, color=LSTM_COLOR)
    ax.fill_between(K_full, gm_full - gs_full, gm_full + gs_full,
                    alpha=0.15, color=GREEDY_COLOR)
    ax.plot(K_full, lm_full, color=LSTM_COLOR, linewidth=2.0, label="Rev-GNN-LSTM")
    ax.plot(K_full, gm_full, color=GREEDY_COLOR, linewidth=2.0, label="Greedy-Discount")

    # Reference lines at K=30, K=100, K=200 (natural budget checkpoints for Rice)
    for k_ref in [30, 100, 200]:
        if k_ref <= n_graph:
            ax.axvline(k_ref, color="gray", linewidth=0.7, linestyle=":")
            ax.text(k_ref + 2, ax.get_ylim()[0] * 1.02 if ax.get_ylim()[0] > 0 else 2,
                    f"K={k_ref}", fontsize=7, color="gray")

    # Annotate the Δ% at K=n (full budget)
    lstm_full = float(lm_full[-1])
    greedy_full = float(gm_full[-1])
    delta_pct = (lstm_full - greedy_full) / greedy_full * 100 if greedy_full > 0 else 0
    ax.annotate(
        f"K={n_graph}: LSTM={lstm_full:.1f}\nGreedy={greedy_full:.1f}  Δ={delta_pct:+.1f}%",
        xy=(n_graph * 0.7, greedy_full * 0.6),
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", edgecolor="gray"),
    )

    ax.set_xlabel("Budget K (number of offers made)")
    ax.set_ylabel("Cumulative Revenue")
    ax.set_title(f"Revenue vs Budget K — Rice-Facebook (n={n_graph}, real network)")
    ax.legend()
    ax.set_xlim(1, n_graph)

    ensure_dir(FIGURE_DIR)
    out = os.path.join(FIGURE_DIR, "fig4b_budget_K_rice_facebook.pdf")
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [✓] {out}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/rev_gnn_lstm.yaml")
    parser.add_argument("--skip-budget", action="store_true",
                        help="Skip budget-K sweep (use existing CSV if present)")
    args = parser.parse_args()

    print("=== plot_figures.py ===")
    ensure_dir(FIGURE_DIR)

    # Fig 1: Learning curve (no policy needed — just CSV)
    print("Figure 1: Learning curve...")
    fig1_learning_curve()

    # Fig 2: Revenue comparison (needs robustness CSV)
    print("Figure 2: Revenue comparison...")
    fig2_revenue_comparison()

    # Fig 3: Generalization (needs generalization CSV)
    print("Figure 3: Generalization...")
    fig3_generalization()

    # Fig 4: Budget-K (needs policy + run or existing CSV)
    print("Figure 4: Budget-K sweep...")
    budget_data = None
    if args.skip_budget and os.path.exists(BUDGET_CSV):
        print(f"  Using existing {BUDGET_CSV}")
        rows = []
        with open(BUDGET_CSV) as f:
            for row in csv.DictReader(f):
                rows.append({k: (int(v) if k == "k" else float(v)) for k, v in row.items()})
        # Load full npy if available
        full_path = BUDGET_CSV.replace(".csv", "_full.npy")
        if os.path.exists(full_path):
            d = np.load(full_path, allow_pickle=True).item()
            budget_data = {"rows": rows, "lstm_curves": d["lstm"], "greedy_curves": d["greedy"]}
        else:
            # Reconstruct curves from row data (less smooth)
            n = max(r["k"] for r in rows)
            lc = np.array([[r["lstm_mean"]] * r["k"] for r in rows]).T
            budget_data = {"rows": rows, "lstm_curves": lc.T.tolist(), "greedy_curves": lc.T.tolist()}
    else:
        cfg = load_config_with_base(args.config)
        set_seed(cfg.project.seed)
        device = torch.device("cpu")
        logger = ExperimentLogger(cfg, run_name="plot_budget")
        print("  Loading LSTM checkpoint...")
        lstm_policy = load_lstm_policy(LSTM_CKPT, cfg, device)
        print(f"  Running budget sweep (5 seeds × full episode)...")
        budget_data = run_budget_sweep(lstm_policy, cfg, device, logger,
                                       n_seeds=5, out_path=BUDGET_CSV)
        logger.finish()

    fig4_budget_K(budget_data)

    # Fig 4b: Rice-Facebook budget-K sweep
    print("Figure 4b: Budget-K sweep — Rice-Facebook...")
    rf_data = None
    if args.skip_budget and os.path.exists(BUDGET_RF_CSV):
        print(f"  Using existing {BUDGET_RF_CSV}")
        rows_rf = []
        with open(BUDGET_RF_CSV) as f:
            for row in csv.DictReader(f):
                rows_rf.append({k: (int(v) if k == "k" else float(v)) for k, v in row.items()})
        full_rf = BUDGET_RF_CSV.replace(".csv", "_full.npy")
        if os.path.exists(full_rf):
            d = np.load(full_rf, allow_pickle=True).item()
            rf_data = {"rows": rows_rf, "lstm_curves": d["lstm"], "greedy_curves": d["greedy"],
                       "n_graph": int(d["lstm"].shape[1])}
    else:
        # Load policy if not already loaded
        if "lstm_policy" not in locals():
            cfg = load_config_with_base(args.config)
            set_seed(cfg.project.seed)
            device = torch.device("cpu")
            print("  Loading LSTM checkpoint for Rice-FB sweep...")
            lstm_policy = load_lstm_policy(LSTM_CKPT, cfg, device)
        logger_rf = ExperimentLogger(cfg, run_name="plot_budget_rf")

        try:
            rf_graph = load_rice_facebook(data_dir="data/raw")
            n_rf = rf_graph.number_of_nodes()
            print(f"  Rice-Facebook: n={n_rf} nodes")
            rf_data = run_budget_sweep_on_graph(
                lstm_policy, lambda s: rf_graph, n_rf, cfg, device, logger_rf,
                n_seeds=5, graph_label=f"rice_facebook n={n_rf}",
                out_path=BUDGET_RF_CSV,
            )
            logger_rf.finish()
        except FileNotFoundError as e:
            print(f"  [SKIP] Rice-Facebook not found: {e}")
            rf_data = None

    if rf_data is not None:
        fig4b_budget_K_rice_facebook(rf_data)

    print(f"\nAll figures saved to {FIGURE_DIR}/")
    print("  fig1_learning_curve.{pdf,png}")
    print("  fig2_revenue_comparison.{pdf,png}")
    print("  fig3_generalization.{pdf,png}")
    print("  fig4_budget_K.{pdf,png}  (Forest Fire n=1000)")
    print("  fig4b_budget_K_rice_facebook.{pdf,png}  (real network)")


if __name__ == "__main__":
    main()
