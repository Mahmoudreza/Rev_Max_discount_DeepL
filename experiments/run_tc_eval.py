"""experiments/run_tc_eval.py — Evaluate 4 Idea 1 methods under TC deadlines + profit.

No training.  Uses existing Idea 1 checkpoints only:
  - Rev-GNN-LSTM   (results/checkpoints/rev_gnn_lstm.pt)
  - Rev-GNN-IM-RL  (results/checkpoints/rev_gnn_im_rl.pt)
  - IE-Strategy    (analytic baseline, no checkpoint)
  - Greedy-Discount (analytic baseline, no checkpoint)

Outputs:
  1. Revenue table at τ checkpoints [50, 100, 200, 300, 500, 1000]
  2. Profit table at c ∈ {0.1, 0.2, 0.3} (profit = revenue - n_accepted × c)
  3. Figures: tc_fig1_revenue_vs_tau, tc_profit_c02, tc_profit_breakeven,
              tc_profit_multi_cost
"""

import argparse, os, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.env.graph_generators import generate_forest_fire
from src.evaluation.paper_eval import load_lstm, load_im
from src.evaluation.tc_baselines import run_tc_comparison_multi_graph, collect_tc_curves
from src.evaluation.tc_evaluation import (
    make_latex_table, cum_rev_to_trajectory,
    profit_at_checkpoints, breakeven_point,
)
from src.utils.tc_visualization import generate_all_tc_figures
from src.utils.logging import ExperimentLogger


METHOD_ORDER = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]
PRODUCTION_COSTS = [0.1, 0.2, 0.3]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/experiments/time_critical.yaml")
    parser.add_argument("--lstm_ckpt", default="results/checkpoints/rev_gnn_lstm.pt")
    parser.add_argument("--im_ckpt",   default="results/checkpoints/rev_gnn_im_rl.pt")
    parser.add_argument("--n_graphs",  type=int, default=10)
    parser.add_argument("--n_trials",  type=int, default=5)
    parser.add_argument("--out_dir",   default="results/logs")
    parser.add_argument("--fig_dir",   default="results/figures/tc")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name="tc_eval")
    logger.info(f"TC Evaluation | n_graphs={args.n_graphs} | n_trials={args.n_trials}")
    logger.info("Methods: IE-Strategy, Greedy-Discount, Rev-GNN-IM-RL, Rev-GNN-LSTM")

    checkpoints = list(cfg.time_critical.checkpoints)
    p, pb = float(cfg.graph.p), float(cfg.graph.pb)

    # ── Load Idea 1 policies (graceful skip if missing) ───────────────────────
    lstm_pol, im_pol = None, None
    if os.path.exists(args.lstm_ckpt):
        lstm_pol = load_lstm(args.lstm_ckpt, cfg, device)
        logger.info(f"  LSTM loaded: {args.lstm_ckpt}")
    else:
        logger.info(f"  [SKIP] LSTM ckpt not found: {args.lstm_ckpt}")
    if os.path.exists(args.im_ckpt):
        im_pol = load_im(args.im_ckpt, cfg, device)
        logger.info(f"  IM-RL loaded: {args.im_ckpt}")
    else:
        logger.info(f"  [SKIP] IM-RL ckpt not found: {args.im_ckpt}")

    graph_fn = lambda s: generate_forest_fire(1000, p, pb, seed=s)
    cache_path = os.path.join(args.out_dir, "tc_eval_ff_4method.json")

    # ── Run TC revenue comparison ─────────────────────────────────────────────
    logger.info(f"\n[FF n=1000] running {args.n_graphs} graphs × {args.n_trials} trials ...")
    tc_results = run_tc_comparison_multi_graph(
        graph_fn, cfg, checkpoints,
        n_graphs=args.n_graphs, lstm_pol=lstm_pol, im_pol=im_pol,
        tc_lstm_pol=None,               # no TC-fine-tuned model
        device=device, n_trials=args.n_trials, out_path=cache_path,
    )

    # ── Revenue table ─────────────────────────────────────────────────────────
    logger.info("=" * 78)
    logger.info("REVENUE AT DEADLINE τ  (FF n=1000, revenue within first τ acceptances)")
    logger.info("=" * 78)
    header = f"{'Method':<25}" + "".join(f"{'τ='+str(t):>10}" for t in checkpoints)
    logger.info(header)
    logger.info("-" * len(header))
    for m in METHOD_ORDER:
        if m not in tc_results:
            continue
        vals = tc_results[m]["checkpoints"]
        line = f"{m:<25}" + "".join(f"{vals.get(t, 0.0):>10.1f}" for t in checkpoints)
        logger.info(line)
    logger.info("")
    logger.info("Area under acceptance curve (higher = more front-loaded):")
    for m in METHOD_ORDER:
        if m in tc_results:
            logger.info(f"  {m:<25} area = {tc_results[m]['area']:.1f}")

    # ── Raw curves for profit analysis ────────────────────────────────────────
    logger.info("\nCollecting raw trajectories for profit analysis (1 graph × 3 trials)...")
    raw_graph = graph_fn(0)
    curves_ff = collect_tc_curves(
        raw_graph, cfg, lstm_pol=lstm_pol, im_pol=im_pol,
        device=device, n_trials=3,
    )

    # Convert cum_rev curves → step-dict trajectories
    profit_trajs = {
        method: [cum_rev_to_trajectory(c) for c in curve_list]
        for method, curve_list in curves_ff.items()
        if method in METHOD_ORDER
    }

    # ── Profit tables ─────────────────────────────────────────────────────────
    profit_checkpoints = [10, 25, 50, 100, 200, 300, 500, 1000]
    for c in PRODUCTION_COSTS:
        logger.info(f"\n{'='*78}")
        logger.info(f"PROFIT ANALYSIS  (production cost c={c}, profit = revenue − τ × c)")
        logger.info(f"{'='*78}")
        ph = f"{'Method':<25}" + "".join(f"{'τ='+str(t):>8}" for t in profit_checkpoints)
        ph += f"{'Breakeven':>12}"
        logger.info(ph)
        logger.info("-" * len(ph))
        for m in METHOD_ORDER:
            trials = profit_trajs.get(m, [])
            if not trials:
                continue
            profits_per_trial = [profit_at_checkpoints(t, profit_checkpoints, c)
                                  for t in trials]
            avg_profit = {tau: float(np.mean([p[tau] for p in profits_per_trial]))
                          for tau in profit_checkpoints}
            breakevens = [breakeven_point(t, c) for t in trials]
            valid_be = [b for b in breakevens if b is not None]
            be_str = f"{np.mean(valid_be):.0f}" if valid_be else "never"
            line = f"{m:<25}"
            for tau in profit_checkpoints:
                line += f"{avg_profit.get(tau, 0.0):>8.1f}"
            line += f"{be_str:>12}"
            logger.info(line)

    # ── Key findings ──────────────────────────────────────────────────────────
    logger.info("\nKEY FINDINGS:")
    for tau in [50, 100, 200, 1000]:
        row = {m: tc_results.get(m, {}).get("checkpoints", {}).get(tau, 0.0)
               for m in METHOD_ORDER}
        best_m = max(row, key=row.get)
        logger.info(
            f"  τ={tau:4d}  IE={row.get('IE-Strategy', 0):.1f}"
            f"  Greedy={row.get('Greedy-Discount', 0):.1f}"
            f"  IM-RL={row.get('Rev-GNN-IM-RL', 0):.1f}"
            f"  LSTM={row.get('Rev-GNN-LSTM', 0):.1f}"
            f"  → BEST: {best_m}"
        )
    # Breakeven comparison at c=0.2
    logger.info("\nBreakeven at c=0.2:")
    for m in METHOD_ORDER:
        trials = profit_trajs.get(m, [])
        bes = [breakeven_point(t, 0.2) for t in trials]
        valid = [b for b in bes if b is not None]
        be_str = f"τ≈{np.mean(valid):.0f}" if valid else "never"
        logger.info(f"  {m:<25} {be_str}")

    # ── Save LaTeX table ──────────────────────────────────────────────────────
    ensure_dir(args.out_dir)
    tex = make_latex_table(tc_results, checkpoints, method_order=METHOD_ORDER)
    tex_path = os.path.join(args.out_dir, "paper_table_tc_revenue.tex")
    with open(tex_path, "w") as f:
        f.write(tex)
    logger.info(f"\nLaTeX revenue table → {tex_path}")

    # ── Generate figures ──────────────────────────────────────────────────────
    ensure_dir(args.fig_dir)
    generate_all_tc_figures(
        tc_results_ff=tc_results,
        figure_dir=args.fig_dir,
        curves_ff=curves_ff,
        tc_results_rice=None,
        tc_results_tc=None,             # no TC-LSTM
        profit_trajectories=profit_trajs,
        profit_costs=PRODUCTION_COSTS,
    )
    logger.info(f"Figures → {args.fig_dir}")

    # ── List generated files ──────────────────────────────────────────────────
    logger.info("\nGenerated figures:")
    for f in sorted(os.listdir(args.fig_dir)):
        logger.info(f"  {args.fig_dir}/{f}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
