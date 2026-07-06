"""experiments/run_tc_training.py — Train Rev-GNN-LSTM-TC with TC reward.

Warm-starts from Idea 1 LSTM checkpoint, then applies TC-REINFORCE Phase 2
with multi-checkpoint reward R = Σ w_i × Revenue(τ_i).
All training logic in src/training/tc_reinforce_trainer.py.
"""

import argparse, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.env.graph_generators import generate_forest_fire
from src.evaluation.paper_eval import load_lstm
from src.training.tc_reinforce_trainer import TCREINFORCETrainer
from src.evaluation.tc_baselines import run_tc_comparison_multi_graph
from src.evaluation.tc_evaluation import make_latex_table
from src.utils.logging import ExperimentLogger


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",    default="configs/experiments/time_critical.yaml")
    parser.add_argument("--lstm_ckpt", default="results/checkpoints/rev_gnn_lstm.pt",
                        help="Idea 1 checkpoint to warm-start from")
    parser.add_argument("--out_ckpt",  default="results/checkpoints/rev_gnn_lstm_tc.pt")
    parser.add_argument("--n_trains",  type=int, default=5)
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    # Force CPU: Beta.rsample() → Dirichlet sampling is not yet supported on MPS.
    # (Eval with greedy=True uses Beta.mean — no sampling — so MPS works there.)
    device = torch.device("cpu")
    logger = ExperimentLogger(cfg, run_name="tc_training")
    logger.info("TC Training (Rev-GNN-LSTM-TC) — sequential model, no IC cascade")

    p, pb = float(cfg.graph.p), float(cfg.graph.pb)

    # ── Load Idea 1 LSTM checkpoint (warm-start) ──────────────────────────────
    if not os.path.exists(args.lstm_ckpt):
        logger.info(f"ERROR: Idea 1 checkpoint not found: {args.lstm_ckpt}")
        logger.info("Run experiments/run_rev_gnn_lstm.py first.")
        return

    policy = load_lstm(args.lstm_ckpt, cfg, device)
    logger.info(f"Warm-start from: {args.lstm_ckpt}")

    # ── Build training graphs ─────────────────────────────────────────────────
    train_graphs = [generate_forest_fire(1000, p, pb, seed=s)
                    for s in range(args.n_trains)]
    logger.info(f"Training graphs: {len(train_graphs)} × FF n=1000")

    # ── TC-REINFORCE Phase 2 ──────────────────────────────────────────────────
    trainer = TCREINFORCETrainer(policy, cfg, logger, device)
    n_epochs = int(cfg.training.reinforce_epochs_phase2)
    logger.info(f"TC-REINFORCE: {n_epochs} epochs | "
                f"checkpoints={list(cfg.time_critical.training_checkpoints)} | "
                f"weights={list(cfg.time_critical.training_weights)}")

    history = trainer.train(train_graphs, n_epochs=n_epochs)
    logger.info(f"Best TC reward: {history['best_reward']:.4f}")

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ensure_dir(os.path.dirname(args.out_ckpt))
    torch.save(policy.state_dict(), args.out_ckpt)
    logger.info(f"TC checkpoint saved → {args.out_ckpt}")

    # ── Quick eval at TC deadline checkpoints ─────────────────────────────────
    if not args.skip_eval:
        logger.info("\nRunning quick TC evaluation (3 graphs × 3 trials)...")
        checkpoints = list(cfg.time_critical.checkpoints)
        from src.evaluation.paper_eval import load_lstm as _load
        tc_pol = load_lstm(args.out_ckpt, cfg, device)
        from src.evaluation.paper_eval import load_im
        base_pol = load_lstm(args.lstm_ckpt, cfg, device)  # Idea 1 for comparison

        graph_fn = lambda s: generate_forest_fire(1000, p, pb, seed=s + 99)
        tc_res = run_tc_comparison_multi_graph(
            graph_fn, cfg, checkpoints, n_graphs=3,
            lstm_pol=tc_pol, device=device, n_trials=3,
        )
        base_res = run_tc_comparison_multi_graph(
            graph_fn, cfg, checkpoints, n_graphs=3,
            lstm_pol=base_pol, device=device, n_trials=3,
        )

        logger.info("TC Improvement (LSTM-TC vs LSTM base):")
        for tau in checkpoints:
            base_v = base_res.get("Rev-GNN-LSTM", {}).get("checkpoints", {}).get(tau, 0.0)
            tc_v   = tc_res.get("Rev-GNN-LSTM",   {}).get("checkpoints", {}).get(tau, 0.0)
            delta  = (tc_v - base_v) / max(base_v, 1e-6) * 100
            logger.info(f"  τ={tau:4d}  base={base_v:.1f}  TC={tc_v:.1f}  Δ={delta:+.1f}%")

    logger.info("TC training complete.")


if __name__ == "__main__":
    main()
