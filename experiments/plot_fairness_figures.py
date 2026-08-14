"""experiments/plot_fairness_figures.py

Phase 4 — Fairness visualizations from results/logs/fairness_audit.json.
NEW FILE only — does NOT modify any existing file.

Produces (saved to paper/figures/):
  fig_fairness_adoption_gap.pdf   — adoption gap per method, Rice-FB + SBM h=0.9
  fig_fairness_rho_k_curve.pdf    — rho_A, rho_B vs K curve on Rice-FB
  fig_fairness_sub_share.pdf      — sub_share_B bar chart (all graphs)
"""
from __future__ import annotations
import sys, json, os
sys.path.insert(0, ".")

import numpy as np

OUTPUT_DIR = "paper/figures"
DATA_FILE  = "results/logs/fairness_audit.json"

# ── PLOTTING IMPORTS (deferred for headless) ─────────────────────────────────
def _get_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11,
        "axes.labelsize": 12, "axes.titlesize": 12,
        "legend.fontsize": 10, "figure.dpi": 150,
    })
    return plt

# ── Color / style map ─────────────────────────────────────────────────────────
METHOD_STYLE = {
    "Greedy-Discount": dict(color="#2196F3", marker="o", lw=2.0, ls="-"),
    "IE-Strategy":     dict(color="#FF9800", marker="s", lw=2.0, ls="--"),
    "Rev-GNN-IM-RL":   dict(color="#9C27B0", marker="^", lw=1.5, ls=":"),
    "Rev-GNN-LSTM":    dict(color="#4CAF50", marker="D", lw=1.5, ls="-."),
    "Fair-Greedy":     dict(color="#F44336", marker="*", lw=2.0, ls="-"),
}
DEFAULT_STYLE = dict(color="#888888", marker="x", lw=1.0, ls=":")


def _style(method: str) -> dict:
    return METHOD_STYLE.get(method, DEFAULT_STYLE)


def load_data(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def fig_adoption_gap(data: dict, out_dir: str) -> str:
    """Bar chart: final gap (rho_A - rho_B) per method, Rice-FB + SBM h=0.9."""
    plt = _get_mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    graph_keys = [("rice_fb", "Rice-FB (n=443)"), ("SBM_h0.9", "SBM h=0.9 (n=1000)")]
    for ax, (gkey, glabel) in zip(axes, graph_keys):
        if gkey not in data:
            ax.set_title(f"{glabel}\n(no data)")
            continue
        methods = data[gkey]["methods"]
        names, gaps, stds = [], [], []
        for m, agg in methods.items():
            fm = agg.get("final", {})
            gap_mean = fm.get("gap", {}).get("mean", float("nan"))
            gap_std  = fm.get("gap", {}).get("std",  0.0)
            names.append(m)
            gaps.append(gap_mean)
            stds.append(gap_std)

        colors = [_style(m)["color"] for m in names]
        x = np.arange(len(names))
        bars = ax.bar(x, gaps, yerr=stds, color=colors,
                      capsize=4, alpha=0.85, edgecolor="k", linewidth=0.6)
        ax.axhline(0.10, ls="--", color="red", lw=1.2, label="Gate F0 (gap≥0.10)")
        ax.axhline(0.0,  ls="-",  color="k",  lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_title(glabel)
        ax.set_ylabel("Adoption gap (rho_A − rho_B)")
        ax.legend(fontsize=9)

    fig.suptitle("Adoption Rate Gap by Method", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_fairness_adoption_gap.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def fig_rho_k_curve(data: dict, out_dir: str) -> str:
    """Line chart: rho_A(K), rho_B(K) on Rice-FB for each method."""
    plt = _get_mpl()
    if "rice_fb" not in data:
        print("  [skip] rice_fb not in data for rho-K curve")
        return ""

    methods = data["rice_fb"]["methods"]
    n_methods = len(methods)
    ncols = min(n_methods, 3)
    nrows = (n_methods + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)
    ax_flat = [ax for row in axes for ax in row]

    for i, (method, agg) in enumerate(methods.items()):
        ax = ax_flat[i]
        # Collect K checkpoints (integers) and final
        ks, rA_means, rA_stds, rB_means, rB_stds = [], [], [], [], []
        for K, mets in agg.items():
            if K == "final":
                continue
            try:
                k_int = int(K)
            except (ValueError, TypeError):
                continue
            rA = mets.get("rho_A", {})
            rB = mets.get("rho_B", {})
            ks.append(k_int)
            rA_means.append(rA.get("mean", float("nan")))
            rA_stds.append(rA.get("std",  0.0))
            rB_means.append(rB.get("mean", float("nan")))
            rB_stds.append(rB.get("std",  0.0))

        if ks:
            order = np.argsort(ks)
            ks_np  = np.array(ks)[order]
            rA_np  = np.array(rA_means)[order]
            rAs_np = np.array(rA_stds)[order]
            rB_np  = np.array(rB_means)[order]
            rBs_np = np.array(rB_stds)[order]

            ax.plot(ks_np, rA_np, label="rho_A (majority)", color="#2196F3", lw=2)
            ax.fill_between(ks_np, rA_np - rAs_np, rA_np + rAs_np, alpha=0.2, color="#2196F3")
            ax.plot(ks_np, rB_np, label="rho_B (minority)", color="#F44336", lw=2, ls="--")
            ax.fill_between(ks_np, rB_np - rBs_np, rB_np + rBs_np, alpha=0.2, color="#F44336")

        ax.set_title(method, fontsize=11)
        ax.set_xlabel("K (acceptances)")
        ax.set_ylabel("Adoption rate")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)

    # hide unused subplots
    for j in range(i + 1, len(ax_flat)):
        ax_flat[j].set_visible(False)

    fig.suptitle("Adoption Rate rho_A(K), rho_B(K) on Rice-FB", fontsize=13)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_fairness_rho_k_curve.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def fig_sub_share(data: dict, out_dir: str) -> str:
    """Grouped bar: sub_share_B vs node_share_B for all graph×method combos."""
    plt = _get_mpl()

    graph_labels = []
    all_methods_set = set()
    for gkey, gdata in data.items():
        graph_labels.append(gkey)
        all_methods_set.update(gdata["methods"].keys())

    all_methods = sorted(all_methods_set)
    n_graphs  = len(graph_labels)
    n_methods = len(all_methods)
    x = np.arange(n_graphs)
    width = 0.8 / max(n_methods, 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    for mi, method in enumerate(all_methods):
        offsets = (mi - n_methods / 2 + 0.5) * width
        bars_val, bars_err = [], []
        for gkey in graph_labels:
            if method in data[gkey]["methods"]:
                fm = data[gkey]["methods"][method].get("final", {})
                bars_val.append(fm.get("sub_share_B", {}).get("mean", float("nan")))
                bars_err.append(fm.get("sub_share_B", {}).get("std",  0.0))
            else:
                bars_val.append(float("nan"))
                bars_err.append(0.0)

        st = _style(method)
        ax.bar(x + offsets, bars_val, width, yerr=bars_err,
               color=st["color"], label=method,
               capsize=3, alpha=0.8, edgecolor="k", linewidth=0.5)

    # node_share_B reference line per graph
    for gi, gkey in enumerate(graph_labels):
        nsB = data[gkey].get("node_share_B", float("nan"))
        if not np.isnan(nsB):
            ax.hlines(nsB, gi - 0.4, gi + 0.4, colors="black", linestyles="--", lw=1.5)
            ax.hlines(0.67 * nsB, gi - 0.4, gi + 0.4, colors="red", linestyles=":", lw=1.2)

    ax.set_xticks(x)
    ax.set_xticklabels(graph_labels, rotation=20, ha="right")
    ax.set_ylabel("sub_share_B (fraction of subsidized items to group B)")
    ax.set_title("Subsidy Share of Minority Group B\n(dashed=node_share_B, dotted=0.67×node_share_B Gate F0)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "fig_fairness_sub_share.pdf")
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
    return out_path


def main():
    if not os.path.exists(DATA_FILE):
        print(f"ERROR: {DATA_FILE} not found. Run run_fairness_audit.py first.")
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data = load_data(DATA_FILE)
    print(f"Loaded {DATA_FILE}: {list(data.keys())}")

    p1 = fig_adoption_gap(data, OUTPUT_DIR)
    p2 = fig_rho_k_curve(data, OUTPUT_DIR)
    p3 = fig_sub_share(data, OUTPUT_DIR)

    print(f"\nFigures saved:")
    for p in [p1, p2, p3]:
        if p:
            print(f"  {p}")


if __name__ == "__main__":
    main()
