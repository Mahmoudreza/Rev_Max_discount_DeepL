"""experiments/plot_idea3_main_v3.py — Main budget figure with hybrid learned line.

Produces results/figures/fig_idea3_main_v2.pdf (overwrites, after backup):
  • orange (solid thick):  Composite Cal-DP = max(v2, v3) per k
  • blue   (solid):        Learned policy (unified for k<20; large-k specialist for k>=20)
  • gray   (dashed):       Greedy+Budget
  • vertical dotted line at k=20: deployment boundary (specialist regime starts)

Data sources:
  - DP composite: results/logs/dp_v3_full_curve_merged.json
  - Unified model (k<20): results/logs/unified_sweep.json
  - Specialist (k>=20):   results/logs/largek_specialist_eval.json
  - Greedy+Budget:        hardcoded from paper_table_idea3_final.tex (frozen)

DO NOT EDIT — regenerate only via this script.

Usage:
    cd revmax-aaai2027 && source venv/bin/activate
    python experiments/plot_idea3_main_v3.py
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT    = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "results" / "logs"
FIG_DIR = ROOT / "results" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEPLOYMENT_BOUNDARY = 20  # k < 20 → unified; k >= 20 → specialist (seed-matched measured boundary)


def _load_dp_composite() -> tuple[list[int], np.ndarray, np.ndarray]:
    """Load DP composite (max(v2,v3)) from dp_v3_full_curve_merged.json.

    Returns:
        k_list, mean_array, std_array
    """
    merged = json.load(open(LOG_DIR / "dp_v3_full_curve_merged.json"))
    data   = merged["ff_n1000"]
    K_LIST = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
    v2_m   = np.array([data[f"k={k}"]["v2"]["mean"] for k in K_LIST])
    v2_s   = np.array([data[f"k={k}"]["v2"]["std"]  for k in K_LIST])
    v3_m   = np.array([data[f"k={k}"]["v3"]["mean"] for k in K_LIST])
    v3_s   = np.array([data[f"k={k}"]["v3"]["std"]  for k in K_LIST])
    comp_m = np.where(v3_m >= v2_m, v3_m, v2_m)
    comp_s = np.where(v3_m >= v2_m, v3_s, v2_s)
    return K_LIST, comp_m, comp_s


def _load_learned_line() -> tuple[list[int], np.ndarray, np.ndarray]:
    """Build hybrid learned line: unified for k<BOUNDARY, specialist for k>=BOUNDARY.

    Returns:
        k_list, mean_array, std_array  (ordered ascending k)
    """
    # --- Unified model (k < BOUNDARY) ---
    unified_raw = json.load(open(LOG_DIR / "unified_sweep.json"))
    ff_u        = unified_raw["results"]["ff"]
    unified_k   = sorted([int(k) for k in ff_u.keys()])
    unified_pts = {k: (ff_u[str(k)]["rev_mean"], ff_u[str(k)]["rev_std"])
                   for k in unified_k}

    # --- Specialist (k >= BOUNDARY) ---
    spec_raw    = json.load(open(LOG_DIR / "largek_specialist_eval.json"))
    spec_r      = spec_raw["results"]
    specialist_k = sorted([int(k) for k in spec_r.keys()])
    specialist_pts = {int(k): (spec_r[k]["mean"], spec_r[k]["std"])
                      for k in spec_r.keys()}

    # Build combined k-list (union; no duplicates except at boundary)
    combined_k = sorted(set(
        [k for k in unified_k if k < DEPLOYMENT_BOUNDARY] +
        [k for k in specialist_k if k >= DEPLOYMENT_BOUNDARY]
    ))

    means = []
    stds  = []
    for k in combined_k:
        if k < DEPLOYMENT_BOUNDARY:
            m, s = unified_pts[k]
        else:
            m, s = specialist_pts.get(k, (float("nan"), float("nan")))
        means.append(m)
        stds.append(s)

    return combined_k, np.array(means), np.array(stds)


def _greedy_budget_data() -> tuple[list[int], np.ndarray]:
    """Greedy+Budget values (frozen from paper_table_idea3_final.tex).

    Returns:
        k_list, mean_array
    """
    # Frozen reference — do not recompute
    greedy_pts = {
        1: 7.4, 2: 18.9, 3: 23.6, 5: 48.9, 8: 77.9,
        10: 118.2, 15: 268.2, 20: 365.2, 30: 428.6, 40: 448.7,
    }
    k_list = sorted(greedy_pts.keys())
    return k_list, np.array([greedy_pts[k] for k in k_list])


def main() -> None:
    """Generate and save the budget figure."""
    # ── Load data ─────────────────────────────────────────────────────────────
    dp_k,   dp_m,   dp_s   = _load_dp_composite()
    lrn_k,  lrn_m,  lrn_s  = _load_learned_line()
    grdy_k, grdy_m          = _greedy_budget_data()

    # ── Back up old PDF ────────────────────────────────────────────────────────
    old_pdf = FIG_DIR / "fig_idea3_main_v2.pdf"
    bak_pdf = FIG_DIR / "fig_idea3_main_v2_prev.pdf"
    if old_pdf.exists():
        shutil.copy2(str(old_pdf), str(bak_pdf))
        print(f"Backed up → {bak_pdf}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.5))

    dp_k_arr = np.array(dp_k)

    # Gray: Greedy+Budget
    ax.plot(np.array(grdy_k), grdy_m,
            color="#888888", linestyle="--", linewidth=1.6,
            marker="s", markersize=4, label="Greedy+Budget", zorder=3)

    # Orange: DP composite with band
    ax.fill_between(dp_k_arr, dp_m - dp_s, dp_m + dp_s,
                    color="#ff7f0e", alpha=0.15)
    ax.plot(dp_k_arr, dp_m,
            color="#ff7f0e", linestyle="-", linewidth=2.4,
            marker="o", markersize=6, zorder=5,
            label=r"Cal-DP composite $\max(\mathrm{v2},\mathrm{v3})$")

    # Blue: Learned (hybrid unified + specialist)
    lrn_k_arr = np.array(lrn_k)
    ax.fill_between(lrn_k_arr, lrn_m - lrn_s, lrn_m + lrn_s,
                    color="#1f77b4", alpha=0.12)
    ax.plot(lrn_k_arr, lrn_m,
            color="#1f77b4", linestyle="-", linewidth=2.0,
            marker="D", markersize=5, zorder=4,
            label="Rev-GNN-LSTM (budget-aware)")

    # Vertical dotted line at deployment boundary
    ax.axvline(x=DEPLOYMENT_BOUNDARY, color="#1f77b4", linestyle=":",
               linewidth=1.0, alpha=0.7)
    ax.annotate(
        f"boundary $k={DEPLOYMENT_BOUNDARY}$\n(specialist regime)",
        xy=(DEPLOYMENT_BOUNDARY, 200),
        xytext=(DEPLOYMENT_BOUNDARY + 1.5, 170),
        fontsize=7, color="#1f77b4",
        arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=0.8),
    )

    # Axes
    ax.set_xlabel("Number of offers $k$", fontsize=11)
    ax.set_ylabel("Revenue (mean ± std)", fontsize=11)
    ax.set_title(
        "Budget-Constrained Revenue — Forest Fire n=1000, c=0.3\n"
        "Blue = hybrid learned (unified k<20, specialist k>=20)",
        fontsize=10,
    )

    # x-ticks: union of all k values shown
    all_k = sorted(set(dp_k + list(lrn_k) + grdy_k))
    ax.set_xticks(all_k)
    ax.set_xticklabels([str(k) for k in all_k], fontsize=7)
    ax.tick_params(axis="y", labelsize=8)
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.set_xlim(0, 42)
    ax.set_ylim(bottom=-15)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.85)

    fig.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    for ext in ("pdf", "png"):
        out = FIG_DIR / f"fig_idea3_main_v2.{ext}"
        fig.savefig(str(out), dpi=150, bbox_inches="tight")
        print(f"Saved → {out}")

    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
