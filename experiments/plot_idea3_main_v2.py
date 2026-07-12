"""experiments/plot_idea3_main_v2.py — Figure: DP family + composite (Idea 3 main).

Produces fig_idea3_main_v2.pdf / .png:
  • v1  (gray, dashed, thinner)   — DP-Calibrated original, appendix reference
  • v2  (blue, dashed)            — DP-Calibrated v2
  • v3  (green, dashed)           — DP-Calibrated v3
  • composite (orange, solid, thick) — max(v2, v3) per k; the paper line
  • shaded range band between v2 and v3 (light gray, alpha=0.2)

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    python experiments/plot_idea3_main_v2.py

Output:
    results/figures/fig_idea3_main_v2.pdf
    results/figures/fig_idea3_main_v2.png
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_data() -> dict:
    """Load merged DP curve data.

    Returns:
        Dict with 'ff_n1000' key containing per-k v1/v2/v3 stats.
    """
    merged = json.load(open(LOG_DIR / "dp_v3_full_curve_merged.json"))
    return merged["ff_n1000"]


def main() -> None:
    data = _load_data()

    K_LIST   = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
    k_vals   = np.array(K_LIST)

    # ── Extract means and stds ─────────────────────────────────────────────────
    v1_m = np.array([data[f"k={k}"]["v1"]["mean"] for k in K_LIST])
    v1_s = np.array([data[f"k={k}"]["v1"]["std"]  for k in K_LIST])
    v2_m = np.array([data[f"k={k}"]["v2"]["mean"] for k in K_LIST])
    v2_s = np.array([data[f"k={k}"]["v2"]["std"]  for k in K_LIST])
    v3_m = np.array([data[f"k={k}"]["v3"]["mean"] for k in K_LIST])
    v3_s = np.array([data[f"k={k}"]["v3"]["std"]  for k in K_LIST])

    # Composite = max(v2, v3) per k; std from the winning method
    comp_m = np.where(v3_m >= v2_m, v3_m, v2_m)
    comp_s = np.where(v3_m >= v2_m, v3_s, v2_s)

    # Transition annotation: k=5 → v3 wins; k=8 → v2 wins
    transition_k = 8   # first k where v2 beats v3

    # ── Plot ───────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    # v2/v3 range band (behind everything)
    band_lo = np.minimum(v2_m, v3_m)
    band_hi = np.maximum(v2_m, v3_m)
    ax.fill_between(k_vals, band_lo, band_hi,
                    color="gray", alpha=0.12, label="_nolegend_")

    # v1 — gray dashed thin (appendix reference)
    ax.plot(k_vals, v1_m, color="#888888", linestyle="--", linewidth=1.2,
            marker="s", markersize=4, label="v1 (original, appendix)")
    ax.fill_between(k_vals, v1_m - v1_s, v1_m + v1_s,
                    color="#888888", alpha=0.08)

    # v2 — blue dashed
    ax.plot(k_vals, v2_m, color="#1f77b4", linestyle="--", linewidth=1.6,
            marker="^", markersize=5, label="v2")
    ax.fill_between(k_vals, v2_m - v2_s, v2_m + v2_s,
                    color="#1f77b4", alpha=0.10)

    # v3 — green dashed
    ax.plot(k_vals, v3_m, color="#2ca02c", linestyle="--", linewidth=1.6,
            marker="D", markersize=5, label="v3")
    ax.fill_between(k_vals, v3_m - v3_s, v3_m + v3_s,
                    color="#2ca02c", alpha=0.10)

    # Composite — orange solid thick (main paper line)
    ax.plot(k_vals, comp_m, color="#ff7f0e", linestyle="-", linewidth=2.4,
            marker="o", markersize=6, zorder=5,
            label=r"Composite $\max(\mathrm{v2},\mathrm{v3})$")
    ax.fill_between(k_vals, comp_m - comp_s, comp_m + comp_s,
                    color="#ff7f0e", alpha=0.15)

    # Transition marker at k=8
    trans_y = comp_m[K_LIST.index(transition_k)]
    ax.axvline(x=transition_k, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)
    ax.annotate(
        "v3→v2\ntransition",
        xy=(transition_k, trans_y * 0.75),
        xytext=(transition_k + 1.5, trans_y * 0.60),
        fontsize=7, color="gray",
        arrowprops=dict(arrowstyle="->", color="gray", lw=0.8),
    )

    # Axes
    ax.set_xlabel("Number of offers $k$", fontsize=11)
    ax.set_ylabel("Revenue (mean ± std)", fontsize=11)
    ax.set_title(
        "DP-Calibrated Family — Forest Fire $n{=}1000$\n"
        r"Composite $= \max(\mathrm{v2}, \mathrm{v3})$ per $k$",
        fontsize=11,
    )
    ax.set_xticks(K_LIST)
    ax.set_xticklabels([str(k) for k in K_LIST], fontsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(0, 42)
    ax.set_ylim(bottom=-15)

    # Legend
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    # Annotation: v1 dominated everywhere
    ax.text(
        0.98, 0.08,
        "v1 dominated at every $k$\n(appendix only)",
        transform=ax.transAxes, fontsize=7, color="#555555",
        ha="right", va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cccccc", alpha=0.8),
    )

    fig.tight_layout()

    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_idea3_main_v2.{ext}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
