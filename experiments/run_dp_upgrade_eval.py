"""experiments/run_dp_upgrade_eval.py — DP-Upgrade Evaluation (Idea 3).

Evaluates calibrated DP, receding-horizon DP, and oracle upper bound
against all existing baselines + LSTM policies.

Usage:
  cd revmax-aaai2027 && source venv/bin/activate
  python -u experiments/run_dp_upgrade_eval.py \\
      --config configs/experiments/dp_upgrade.yaml \\
      [--n 200] [--n_sims 10] [--n_trials 2] [--no_lstm]

Output:
  results/logs/dp_upgrade_eval.json
  results/logs/paper_table_dp_upgrade.tex
  results/logs/paper_table_bmin.tex
  results/figures/fig_dp1_revenue_vs_k.{pdf,png}
  results/figures/fig_dp2_bmin_feasibility.{pdf,png}
"""

import argparse, json, os, sys
from pathlib import Path

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.budget_baselines import (
    greedy_discount_budget, two_phase_dp_budget,
    evaluate_policy_under_budget, evaluate_budget_aware_policy,
)
from src.evaluation.dp_calibrated import (
    dp_calibrated_budget, dp_upper_bound,
    # dp_receding_budget,  # dropped from paper (per Task 2)
)
from src.evaluation.bmin_feasibility import apply_bmin_analysis
from src.utils.logging import ExperimentLogger

LOG_DIR = "results/logs"
FIG_DIR = "results/figures"


def _gv(r, key, sub="mean"):
    v = r.get(key, {})
    return v.get(sub, 0.0) if isinstance(v, dict) else float(v or 0)


def _load_lstm(ckpt, cfg, device, input_dim=20):
    import torch
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy

    # Auto-detect LSTM hidden size from checkpoint to handle version mismatches
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    # weight_ih_l0 shape = (4*lstm_hidden, gnn_hidden)
    ih_shape = sd.get("sequence_model.lstm.weight_ih_l0",
                      sd.get("lstm.weight_ih_l0", None))
    gnn_hidden = int(cfg.encoder.hidden_dim)
    if ih_shape is not None:
        lstm_hidden_actual = ih_shape.shape[0] // 4
    else:
        lstm_hidden_actual = int(cfg.sequence_model.lstm_hidden)
    lstm_n_layers = int(cfg.sequence_model.lstm_n_layers)

    enc  = GraphSAGEEncoder(input_dim, gnn_hidden, int(cfg.encoder.n_layers), 0.0)
    lstm = EpisodeLSTM(gnn_hidden, lstm_hidden_actual, lstm_n_layers)
    pol  = SequentialJointPolicy(enc, lstm,
                                 gnn_dim=gnn_hidden,
                                 context_dim=lstm_hidden_actual)
    pol.load_state_dict(sd, strict=False)
    return pol.to(device).eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/dp_upgrade.yaml")
    parser.add_argument("--n",          type=int,   default=None,  help="Override graph n (forest-fire only)")
    parser.add_argument("--graph_pkl",  type=str,   default=None,  help="Path to pickled NetworkX graph (overrides forest-fire generation)")
    parser.add_argument("--n_sims",     type=int,   default=None,  help="Override calibration n_sims")
    parser.add_argument("--n_trials",   type=int,   default=None,  help="Override n_trials")
    parser.add_argument("--no_lstm",         action="store_true", help="Skip LSTM methods")
    parser.add_argument("--skip_two_phase",  action="store_true", help="Skip Two-Phase-DP (keep for paper final run)")
    parser.add_argument("--out_suffix", type=str,   default="",    help="Suffix for output JSON filename (e.g. '_rice')")
    parser.add_argument("--lstm_ckpt",   default="results/checkpoints/rev_gnn_lstm.pt")
    parser.add_argument("--budget_ckpt", default="results/checkpoints/rev_gnn_lstm_budget.pt")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(int(cfg.project.seed))

    c         = float(cfg.eval.c)
    weight_hi = float(cfg.eval.weight_high)
    n_trials  = int(args.n_trials or cfg.eval.n_trials)
    sweep_k   = list(cfg.budget_constrained.sweep_k)

    n_sims    = int(args.n_sims  or cfg.dp_upgrade.n_sims)
    n_classes = int(cfg.dp_upgrade.n_classes)
    delta     = float(cfg.dp_upgrade.delta)
    resolve_every = int(cfg.dp_upgrade.resolve_every)
    bmin_fracs    = list(cfg.bmin_fracs)

    device = get_device()
    ensure_dir(LOG_DIR); ensure_dir(FIG_DIR)
    log = ExperimentLogger(cfg, run_name="dp_upgrade_eval")

    # ── Build graph ───────────────────────────────────────────────────────────
    if args.graph_pkl:
        import pickle, networkx as nx
        with open(args.graph_pkl, "rb") as _f:
            graph = pickle.load(_f)
        if graph.is_directed():
            graph = graph.to_undirected()
        print(f"Graph (pkl): n={graph.number_of_nodes()}, m={graph.number_of_edges()} ({args.graph_pkl})")
    else:
        n = int(args.n or cfg.graph.n)
        graph = generate_forest_fire(n, float(cfg.graph.p), float(cfg.graph.pb),
                                      seed=int(cfg.project.seed))
        print(f"Graph (FF): n={graph.number_of_nodes()}, m={graph.number_of_edges()}")

    # Shared BudgetEnvConfig for calibration
    base_env_cfg = BudgetEnvConfig(
        budget_B=1.0, production_cost=c, weight_high=weight_hi,
        seed=int(cfg.project.seed),
    )

    # ── Optionally load LSTM policies ─────────────────────────────────────────
    lstm_idea1 = lstm_idea3 = None
    if not args.no_lstm:
        if os.path.exists(args.lstm_ckpt):
            lstm_idea1 = _load_lstm(args.lstm_ckpt, cfg, device, input_dim=20)
            print(f"Loaded LSTM-Idea1 from {args.lstm_ckpt}")
        if os.path.exists(args.budget_ckpt):
            lstm_idea3 = _load_lstm(args.budget_ckpt, cfg, device, input_dim=21)
            print(f"Loaded LSTM-Idea3 from {args.budget_ckpt}")

    # ── Run sweep ─────────────────────────────────────────────────────────────
    all_results: dict = {}
    METHOD_ORDER = [
        "Greedy+Budget", "Two-Phase-DP", "DP-Calibrated",
        "DP-Receding", "DP-Oracle",
        "LSTM-Idea1", "LSTM-Idea3",
    ]

    for k in sweep_k:
        B      = round(k * c, 6)
        k_key  = f"k={k}"
        all_results[k_key] = {}
        print(f"\n── k={k}  B={B:.3f}  c={c} ──")

        kw_dp = dict(n_classes=n_classes, n_sims=n_sims, delta=delta)

        # Greedy (existing)
        r = greedy_discount_budget(graph, B, c, n_trials=n_trials, weight_high=weight_hi)
        all_results[k_key]["Greedy+Budget"] = r
        print(f"  Greedy+Budget      rev={_gv(r,'revenue'):.2f}")

        # DP-naive (skip with --skip_two_phase for paper final run)
        if not args.skip_two_phase:
            r = two_phase_dp_budget(graph, B, c, n_trials=n_trials, delta=delta, weight_high=weight_hi)
            all_results[k_key]["Two-Phase-DP"] = r
            print(f"  Two-Phase-DP       rev={_gv(r,'revenue'):.2f}")

        # DP-Calibrated (new)
        r = dp_calibrated_budget(graph, base_env_cfg, B, c, n_trials=n_trials, **kw_dp)
        all_results[k_key]["DP-Calibrated"] = r
        print(f"  DP-Calibrated      rev={_gv(r,'revenue'):.2f}")

        # DP-Receding: DROPPED from paper (broken; use DP-Calibrated instead)
        # r = dp_receding_budget(graph, base_env_cfg, B, c, n_trials=n_trials,
        #                         resolve_every=resolve_every, **kw_dp)
        # all_results[k_key]["DP-Receding"] = r

        # DP-Oracle (new, upper bound)
        r = dp_upper_bound(graph, base_env_cfg, B, c, n_trials=n_trials, delta=delta)
        all_results[k_key]["DP-Oracle"] = r
        print(f"  DP-Oracle          rev={_gv(r,'revenue'):.2f}  [oracle UB]")

        # Log metrics dict for each method
        for mname, res in all_results[k_key].items():
            log.log({"k": k, "method": mname, "revenue": _gv(res, "revenue"),
                     "n_accepted": _gv(res, "n_accepted")})

        # Check oracle dominates all non-oracle methods
        oracle_rev = _gv(all_results[k_key]["DP-Oracle"], "revenue")
        for m in ["Greedy+Budget", "DP-Calibrated"]:
            if m not in all_results[k_key]:
                continue
            mv = _gv(all_results[k_key][m], "revenue")
            if mv > oracle_rev + 1e-3:
                print(f"  WARNING: {m} ({mv:.2f}) > oracle ({oracle_rev:.2f}) — check bound logic")

        # LSTM methods
        if lstm_idea1 is not None:
            r = evaluate_policy_under_budget(lstm_idea1, graph, B, c, device,
                                              n_trials=n_trials, has_lstm=True,
                                              weight_high=weight_hi)
            all_results[k_key]["LSTM-Idea1"] = r
            print(f"  LSTM-Idea1         rev={_gv(r,'revenue'):.2f}")
        if lstm_idea3 is not None:
            r = evaluate_budget_aware_policy(lstm_idea3, graph, B, c, device,
                                              n_trials=n_trials, weight_high=weight_hi)
            all_results[k_key]["LSTM-Idea3"] = r
            print(f"  LSTM-Idea3         rev={_gv(r,'revenue'):.2f}")

        # Accounting identity check for all new methods
        for mname in ["DP-Calibrated", "DP-Receding", "DP-Oracle"]:
            if mname not in all_results[k_key]:
                continue
            err = all_results[k_key][mname].get("accounting_err", {})
            max_err = max(err.get("all", [0.0])) if isinstance(err, dict) else 0.0
            if max_err > 1e-4:
                print(f"  ACCOUNTING VIOLATION {mname}: max_err={max_err:.6f} — ABORT")
                raise RuntimeError(f"Accounting identity violated for {mname} at k={k}")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    suffix = args.out_suffix or ""
    out_json = os.path.join(LOG_DIR, f"dp_upgrade_eval{suffix}.json")
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_json}")

    # ── Bmin feasibility analysis ─────────────────────────────────────────────
    bmin_out = os.path.join(LOG_DIR, f"bmin_analysis{suffix}.json")
    bmin_results = apply_bmin_analysis(out_json, bmin_fracs=tuple(bmin_fracs), c=c,
                                       out_path=bmin_out)
    print(f"Bmin analysis saved to {bmin_out}")

    # ── Figures + tables ──────────────────────────────────────────────────────
    from src.utils.dp_upgrade_visualization import generate_dp_figures, generate_latex_tables
    generate_dp_figures(all_results, sweep_k, FIG_DIR)
    generate_latex_tables(all_results, bmin_results, sweep_k, LOG_DIR)
    print("Figures and LaTeX tables generated.")


if __name__ == "__main__":
    main()
