"""
experiments/figures/generate_figures.py

Generate all paper figures from logged results CSV files.

Figures:
  Fig 1: Revenue comparison bar chart (baselines vs GNN models)
  Fig 2: Learning curves (imitation loss, REINFORCE revenue)
  Fig 3: Baseline ordering: greedy > sigma > mu > IE (monotone model)
  Fig 4: Non-monotone model analysis (Rayleigh PDF plot)
  Fig 5: Ablation: influence model (monotone vs non-monotone)
  Fig 6: Ablation: graph type (BA vs WS vs ER)

All figures saved to results/figures/*.pdf
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import networkx as nx


def plot_rayleigh_models(save_dir: Path) -> None:
    """Fig 4: Plot monotone vs non-monotone Rayleigh valuation functions.

    Args:
        save_dir: Directory to save PDF.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping figure generation")
        return

    from src.env.influence_models import MonotoneInfluenceModel, NonMonotoneInfluenceModel

    xs = np.linspace(0, 1, 500)
    monotone = MonotoneInfluenceModel(b=1.0)
    non_monotone = NonMonotoneInfluenceModel(b=1.0)

    y_mono = monotone.batch(xs)
    y_non = non_monotone.batch(xs)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(xs, y_non, label="Non-monotone f(x)", color="#2196F3", linewidth=2)
    ax.plot(xs, y_mono, label="Monotone f(x) [clipped]", color="#FF5722",
            linewidth=2, linestyle="--")
    ax.axvline(x=0.5, color="gray", linestyle=":", alpha=0.7, label="Peak (x=0.5)")
    ax.set_xlabel("Normalized influence x")
    ax.set_ylabel("Valuation f(x)")
    ax.set_title("Rayleigh Valuation Functions (Babaei et al. 2013)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "fig4_rayleigh_models.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir / 'fig4_rayleigh_models.pdf'}")


def plot_baseline_ordering(save_dir: Path) -> None:
    """Fig 3: Baseline revenue ordering on BA graph.

    Demonstrates greedy > sigma > mu > ie on a small BA graph.

    Args:
        save_dir: Directory to save PDF.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    from src.utils.helpers import load_config_with_base
    from src.evaluation.baselines import run_all_baselines
    from src.env.graph_generators import generate_graph

    cfg = load_config_with_base("configs/experiments/rev_gnn_im_rl.yaml")
    graph = generate_graph(cfg, graph_type="ba", n=100)
    results = run_all_baselines(graph, cfg, n_trials=10)

    strategies = list(results.keys())
    revenues = [results[s] for s in strategies]
    colors = ["#FF7043", "#42A5F5", "#66BB6A", "#AB47BC"]

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(strategies, revenues, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Mean Revenue")
    ax.set_title("Baseline Strategy Comparison (BA n=100, 10 trials)")
    ax.set_xticklabels(["IE-Strategy", "µ-Discount", "Greedy-Disc.", "σ-Discount"],
                       rotation=20, ha="right")
    for bar, rev in zip(bars, revenues):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{rev:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()

    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "fig3_baseline_ordering.pdf", dpi=150)
    plt.close(fig)
    print(f"Saved: {save_dir / 'fig3_baseline_ordering.pdf'}")


def main() -> None:
    save_dir = Path("results/figures")
    save_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Fig 4: Rayleigh valuation models...")
    plot_rayleigh_models(save_dir)

    print("Generating Fig 3: Baseline ordering...")
    plot_baseline_ordering(save_dir)

    print("Done. Figures saved to results/figures/")


if __name__ == "__main__":
    main()
