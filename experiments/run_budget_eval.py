"""experiments/run_budget_eval.py — Budget-Constrained Evaluation (Idea 3).

Evaluates baselines + Idea 1/3 policies on:
  Panel A: standard weights U(0,2)
  Panel B: robustness, weak influence U(0,1)

Budget axis: k = B/c  (normalised units of production cost).
Run:
  cd revmax-aaai2027 && source venv/bin/activate
  python -u experiments/run_budget_eval.py \
    --config configs/experiments/budget_constrained.yaml \
    --budget_ckpt results/checkpoints/rev_gnn_lstm_budget.pt
"""

import argparse, os, sys, json
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.env.graph_generators import generate_forest_fire
from src.evaluation.budget_baselines import run_budget_comparison
from src.utils.logging import ExperimentLogger

METHOD_ORDER = ["Greedy+Budget", "Efficiency-Greedy", "Two-Phase-DP",
                "LSTM-Idea1", "LSTM-Idea3"]


def _get_val(r, key, sub="mean"):
    v = r.get(key, {})
    return v.get(sub, 0.0) if isinstance(v, dict) else float(v or 0)


def _load_lstm(ckpt, cfg, device):
    import torch
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy
    enc  = GraphSAGEEncoder(20, int(cfg.encoder.hidden_dim),
                            int(cfg.encoder.n_layers), float(getattr(cfg.encoder, "dropout", 0.0)))
    lstm = EpisodeLSTM(graph_dim=int(cfg.encoder.hidden_dim),
                       lstm_hidden=int(cfg.sequence_model.lstm_hidden),
                       n_layers=int(cfg.sequence_model.lstm_n_layers))
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=int(cfg.encoder.hidden_dim),
                                 context_dim=int(cfg.sequence_model.lstm_hidden))
    pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=False)
    return pol.to(device).eval()


def _load_budget_lstm(ckpt, cfg, device):
    import torch
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy
    enc  = GraphSAGEEncoder(21, int(cfg.encoder.hidden_dim),
                            int(cfg.encoder.n_layers), float(getattr(cfg.encoder, "dropout", 0.0)))
    lstm = EpisodeLSTM(graph_dim=int(cfg.encoder.hidden_dim),
                       lstm_hidden=int(cfg.sequence_model.lstm_hidden),
                       n_layers=int(cfg.sequence_model.lstm_n_layers))
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=int(cfg.encoder.hidden_dim),
                                 context_dim=int(cfg.sequence_model.lstm_hidden))
    pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True), strict=True)
    return pol.to(device).eval()


def _print_revenue_table(logger, results_k, sweep_k, c, label):
    cols = ["k=" + str(k) for k in sweep_k]
    hdr  = "{:<22}".format("Method") + "".join("{:>9}".format(c_) for c_ in cols)
    logger.info(f"\nREVENUE TABLE  [{label}  c={c}  k=B/c]")
    logger.info("=" * len(hdr))
    logger.info(hdr)
    logger.info("-" * len(hdr))
    for m in METHOD_ORDER:
        line = "{:<22}".format(m)
        for k in sweep_k:
            r = results_k.get(f"k={k}", {}).get(m, {})
            line += "{:>9.1f}".format(_get_val(r, "revenue"))
        logger.info(line)


def _print_audit_table(logger, results_k, sweep_k, c):
    """Print min_B, final_B, n_subsidized, balance check per method per k."""
    cols = ["k=" + str(k) for k in sweep_k]
    hdr  = "{:<28}".format("Metric  /  Method") + "".join("{:>8}".format(c_) for c_ in cols)
    logger.info("\nAUDIT TABLE  [final_B = k*c - c*n_acc + rev  (±1e-3)]")
    logger.info("-" * len(hdr))
    for m in METHOD_ORDER:
        r_list = [results_k.get(f"k={k}", {}).get(m, {}) for k in sweep_k]
        if not any(r_list):
            continue
        line_minB = "{:<28}".format(f"  min_B    ({m[:16]})")
        line_finB = "{:<28}".format(f"  final_B  ({m[:16]})")
        line_nsub = "{:<28}".format(f"  n_subsid ({m[:16]})")
        line_ok   = "{:<28}".format(f"  balance  ({m[:16]})")
        for k, r in zip(sweep_k, r_list):
            B0 = k * c
            rev  = _get_val(r, "revenue")
            nacc = _get_val(r, "n_accepted")
            finB = _get_val(r, "final_budget")
            minB = _get_val(r, "min_budget", "mean") if r.get("min_budget") else finB
            nsub = _get_val(r, "n_subsidized", "mean") if r.get("n_subsidized") else 0.0
            expected = B0 - c * nacc + rev
            ok   = "✓" if abs(finB - expected) < 1e-1 else "✗"
            line_minB += "{:>8.2f}".format(minB)
            line_finB += "{:>8.2f}".format(finB)
            line_nsub += "{:>8.0f}".format(nsub)
            line_ok   += "{:>8}".format(ok)
        logger.info(line_minB)
        logger.info(line_finB)
        logger.info(line_nsub)
        logger.info(line_ok)


def _find_knee_crossover(results_k, sweep_k, c):
    """Return Greedy's knee index (max slope drop) and LSTM-Idea3/Greedy crossover k."""
    greedy_revs = [_get_val(results_k.get(f"k={k}", {}).get("Greedy+Budget", {}), "revenue")
                   for k in sweep_k]
    idea3_revs  = [_get_val(results_k.get(f"k={k}", {}).get("LSTM-Idea3", {}), "revenue")
                   for k in sweep_k]
    # Knee: first k where incremental gain drops below 5% of peak gain
    deltas      = [greedy_revs[i+1] - greedy_revs[i] for i in range(len(greedy_revs)-1)]
    peak_delta  = max(deltas) if deltas else 1.0
    knee_k      = None
    for i, d in enumerate(deltas):
        if d < 0.05 * peak_delta:
            knee_k = sweep_k[i+1]
            break
    # Crossover: first k where LSTM-Idea3 revenue < Greedy revenue (Greedy passes LSTM)
    crossover_k = None
    for k, g, l3 in zip(sweep_k, greedy_revs, idea3_revs):
        if g > l3 and l3 > 0:
            crossover_k = k
            break
    return knee_k, crossover_k, greedy_revs, idea3_revs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/experiments/budget_constrained.yaml")
    parser.add_argument("--lstm_ckpt",   default="results/checkpoints/rev_gnn_lstm.pt")
    parser.add_argument("--budget_ckpt", default=None)
    parser.add_argument("--n_trials",    type=int, default=3)
    parser.add_argument("--out_dir",     default="results/logs")
    parser.add_argument("--fig_dir",     default="results/figures/budget")
    args = parser.parse_args()

    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name="budget_eval_v2")
    ensure_dir(args.out_dir); ensure_dir(args.fig_dir)

    c      = float(cfg.budget_constrained.production_cost)
    sweep_k = list(cfg.budget_constrained.sweep_k)
    B_list  = [round(k * c, 8) for k in sweep_k]
    b_val   = float(cfg.influence.b)
    w_strong = float(getattr(cfg.budget_constrained, "weight_high_strong", 2.0))
    w_weak   = float(getattr(cfg.budget_constrained, "weight_high_weak",   1.0))

    logger.info(f"Budget Eval v2  c={c}  sweep_k={sweep_k}  n_trials={args.n_trials}")
    logger.info(f"B values: {B_list}")

    # ── Load policies ──────────────────────────────────────────────────────────
    lstm_pol   = None
    budget_pol = None
    from src.evaluation.paper_eval import load_lstm
    if os.path.exists(args.lstm_ckpt):
        try:
            lstm_pol = load_lstm(args.lstm_ckpt, cfg, device)
            logger.info(f"  LSTM-Idea1 loaded")
        except Exception as e:
            logger.info(f"  [SKIP] LSTM-Idea1: {e}")
    if args.budget_ckpt and os.path.exists(args.budget_ckpt):
        try:
            budget_pol = _load_budget_lstm(args.budget_ckpt, cfg, device)
            logger.info(f"  LSTM-Idea3 loaded: {args.budget_ckpt}")
        except Exception as e:
            logger.info(f"  [SKIP] LSTM-Idea3: {e}")

    graph = generate_forest_fire(1000, float(cfg.graph.p), float(cfg.graph.pb),
                                 seed=cfg.project.seed)
    logger.info(f"Graph: n={graph.number_of_nodes()}  m={graph.number_of_edges()}")

    # ── Run both panels ────────────────────────────────────────────────────────
    all_panels = {}
    for panel_label, w_high in [("strong_U02", w_strong), ("weak_U01", w_weak)]:
        logger.info(f"\n{'─'*60}")
        logger.info(f"Panel: {panel_label}  weight_high={w_high}")
        panel_res = {}
        for k, B in zip(sweep_k, B_list):
            logger.info(f"  k={k}  B={B:.3f} ...")
            panel_res[f"k={k}"] = run_budget_comparison(
                graph, B=B, c=c, b=b_val, n_trials=args.n_trials,
                lstm_policy=lstm_pol, im_policy=None,
                budget_policy=budget_pol, device=device,
                weight_high=w_high,
            )
        all_panels[panel_label] = panel_res

    # ── Revenue tables ────────────────────────────────────────────────────────
    for label, panel_res in all_panels.items():
        _print_revenue_table(logger, panel_res, sweep_k, c, label)

    # ── Audit tables ──────────────────────────────────────────────────────────
    for label, panel_res in all_panels.items():
        logger.info(f"\nAUDIT  [{label}]")
        _print_audit_table(logger, panel_res, sweep_k, c)

    # ── Report: knee + crossover ───────────────────────────────────────────────
    logger.info("\nKEY FINDINGS:")
    for label, panel_res in all_panels.items():
        knee_k, xover_k, g_revs, l3_revs = _find_knee_crossover(panel_res, sweep_k, c)
        logger.info(f"  [{label}]  Greedy knee≈k={knee_k}  "
                    f"LSTM-Idea3 vs Greedy crossover≈k={xover_k}")
        # Check if Greedy knee is near k≈15 and crossover near k≈5
        for k, gv, lv in zip(sweep_k, g_revs, l3_revs):
            if lv > 0:
                logger.info(f"    k={k:3d}  Greedy={gv:.1f}  LSTM-Idea3={lv:.1f}  "
                            f"{'← LSTM wins' if lv>gv else '← Greedy wins'}")

    # ── Save results ────────────────────────────────────────────────────────────
    for label, panel_res in all_panels.items():
        fname = "budget_eval_weak_weights.json" if "weak" in label else "budget_eval_c0.3.json"
        path  = os.path.join(args.out_dir, fname)
        with open(path, "w") as f:
            json.dump(panel_res, f, indent=2, default=str)
        logger.info(f"Saved {label} → {path}")

    # ── Figure ─────────────────────────────────────────────────────────────────
    try:
        from src.utils.visualization import plot_budget_revenue_panels
        fig_path = os.path.join(args.fig_dir, "fig_b1_v2.pdf")
        ensure_dir(args.fig_dir)
        plot_budget_revenue_panels(
            all_panels["strong_U02"], all_panels["weak_U01"],
            sweep_k, c, fig_path, logger=logger,
        )
        logger.info(f"Figure → {fig_path}")
    except Exception as e:
        logger.info(f"  [SKIP] figure: {e}")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
