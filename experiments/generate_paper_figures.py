#!/usr/bin/env python
"""
experiments/generate_paper_figures.py — Regenerate all Idea-1 paper figures from cache.

No re-evaluation. Loads all pre-computed JSON caches and calls figure/table
generation functions from paper_figures.py.

Uses paper_gen_patched.json (Modular FF IM-RL = 317.96, seed=0 confirmed fix).

Usage:
    python experiments/generate_paper_figures.py
    python experiments/generate_paper_figures.py --gen-cache results/logs/paper_gen_corrected.json
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from paper_figures import (
    fig1_revenue_vs_K_ff,
    fig2_revenue_vs_K_rice,
    fig3_boxplot,
    fig4_generalization,
    fig5_discount_trajectory,
    table1_latex,
    table2_latex,
    FIGURE_DIR,
    LOG_DIR,
)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load(path):
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Regenerate all paper figures from cached JSON data (no re-eval)"
    )
    parser.add_argument("--config",     default="configs/experiments/rev_gnn_lstm.yaml")
    parser.add_argument("--gen-cache",  default="results/logs/paper_gen_patched.json",
                        help="Generalization table JSON to use (default: patched = Modular FF fixed)")
    parser.add_argument("--comp-cache", default="results/logs/paper_comp20_updated.json",
                        help="20-seed comparison JSON")
    parser.add_argument("--mono-cache", default="results/logs/paper_ff_mono4.json",
                        help="FF monotone budget sweep JSON")
    parser.add_argument("--nonmono-cache", default="results/logs/paper_ff_nonmono4.json",
                        help="FF non-monotone budget sweep JSON")
    parser.add_argument("--rice-cache", default="results/logs/paper_rice4.json",
                        help="Rice-FB budget sweep JSON")
    parser.add_argument("--traj-cache", default="results/logs/paper_discount_traj.json",
                        help="Discount trajectory JSON")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    ensure_dir(FIGURE_DIR)
    ensure_dir(LOG_DIR)

    print("=" * 60)
    print("generate_paper_figures.py — figures from cached data")
    print("=" * 60)

    # ── Load all cached data ──────────────────────────────────────────────────
    print("\nLoading cached data...")

    gen_data = _load(args.gen_cache)
    print(f"  gen:     {args.gen_cache}")
    # Quick sanity check
    mff_imrl = gen_data.get("Modular FF", {}).get("Rev-GNN-IM-RL", {}).get("mean", -1)
    if mff_imrl < 1.0:
        print(f"  ⚠️  WARNING: Modular FF IM-RL = {mff_imrl:.2f} (expected ~317.96)")
    else:
        print(f"  ✓ Modular FF IM-RL = {mff_imrl:.2f} (fix confirmed)")

    comp20 = _load(args.comp_cache)
    print(f"  comp20:  {args.comp_cache}")
    imrl_mean = comp20.get("Rev-GNN-IM-RL", {}).get("mean", 0)
    lstm_mean = comp20.get("Rev-GNN-LSTM",  {}).get("mean", 0)
    print(f"  IM-RL={imrl_mean:.2f}  LSTM={lstm_mean:.2f}")

    ff_mono = _load(args.mono_cache)
    print(f"  ff_mono: {args.mono_cache}")

    ff_nm = _load(args.nonmono_cache)
    print(f"  ff_nm:   {args.nonmono_cache}")

    rice_data = None
    if os.path.exists(args.rice_cache):
        rice_data = _load(args.rice_cache)
        print(f"  rice:    {args.rice_cache}")
    else:
        print(f"  [SKIP]   rice cache not found: {args.rice_cache}")

    disc_traj = None
    if os.path.exists(args.traj_cache):
        disc_traj = _load(args.traj_cache)
        print(f"  traj:    {args.traj_cache}")
    else:
        print(f"  [SKIP]   disc traj not found: {args.traj_cache}")

    # ── Check comp20 has all 4 methods ────────────────────────────────────────
    # Ensure delta_pct is set for all methods
    greedy_mean = comp20.get("Greedy-Discount", {}).get("mean", 416.34)
    for m in comp20:
        if isinstance(comp20[m], dict) and "mean" in comp20[m] and m != "Greedy-Discount":
            if "delta_pct" not in comp20[m]:
                comp20[m]["delta_pct"] = (comp20[m]["mean"] - greedy_mean) / greedy_mean * 100

    print("\n" + "=" * 60)
    print("Generating figures...")

    # ── Figure 1: Revenue vs K — Forest Fire ─────────────────────────────────
    print("\n[1/7] Figure 1: FF budget sweep (mono + non-mono)...")
    fig1_revenue_vs_K_ff(ff_mono, ff_nm)

    # ── Figure 2: Revenue vs K — Rice-Facebook ────────────────────────────────
    if rice_data:
        print("[2/7] Figure 2: Rice-FB budget sweep...")
        fig2_revenue_vs_K_rice(rice_data)
    else:
        print("[2/7] [SKIP] Figure 2 — no Rice-FB cache")

    # ── Figure 3: Boxplot — 20 seeds ─────────────────────────────────────────
    print("[3/7] Figure 3: Boxplot (20 seeds)...")
    fig3_boxplot(comp20)

    # ── Figure 4: Generalization — 5 networks ────────────────────────────────
    print("[4/7] Figure 4: Generalization (5 networks)...")
    fig4_generalization(gen_data)

    # ── Figure 5: Discount trajectory ────────────────────────────────────────
    if disc_traj:
        print("[5/7] Figure 5: Discount trajectory...")
        fig5_discount_trajectory(disc_traj)
    else:
        print("[5/7] [SKIP] Figure 5 — no discount traj cache")

    # ── LaTeX Table 1 ────────────────────────────────────────────────────────
    print("[6/7] Table 1: Main results (LaTeX)...")
    table1_latex(comp20)

    # ── LaTeX Table 2 ────────────────────────────────────────────────────────
    print("[7/7] Table 2: Generalization (LaTeX)...")
    table2_latex(gen_data)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("All figures and tables generated!\n")

    # Print gen table for verification
    NETS = list(gen_data.keys())
    METHODS = ["IE-Strategy", "Greedy-Discount", "S2V-DQN (dec.)", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
    print("Generalization table (from gen cache):")
    print(f"  {'Method':<22}  " + "  ".join(f"{n[:10]:>10}" for n in NETS))
    print("  " + "-" * 78)
    for m in METHODS:
        row = []
        for n in NETS:
            v = gen_data.get(n, {}).get(m, {}).get("mean", float("nan"))
            row.append(f"{v:>10.2f}" if v == v else f"{'---':>10}")
        print(f"  {m:<22}  " + "  ".join(row))

    print(f"\n✓ All outputs in {FIGURE_DIR}/")

    print("\nFiles generated:")
    import glob
    for f in sorted(glob.glob(f"{FIGURE_DIR}/fig*.pdf")):
        print(f"  {f}")
    for f in sorted(glob.glob(f"{LOG_DIR}/paper_table*.tex")):
        print(f"  {f}")


if __name__ == "__main__":
    main()
