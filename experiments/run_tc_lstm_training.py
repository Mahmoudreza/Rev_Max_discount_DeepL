#!/usr/bin/env python
"""
experiments/run_tc_lstm_training.py — Fine-tune LSTM for time-critical objective.

Strategy: warm-start from Idea 1 LSTM checkpoint, then TC-REINFORCE fine-tune.
No Phase 1 imitation — the LSTM already learned good seed selection.
Only fine-tune Phase A choices to front-load cascade revenue (multi-horizon reward).

Saved to: results/checkpoints/rev_gnn_lstm_tc_best.pt

Usage:
    python experiments/run_tc_lstm_training.py \\
        --config configs/experiments/time_critical.yaml
"""

import argparse, os, sys, time, copy
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_forest_fire
from src.evaluation.paper_eval import load_lstm
from src.training.tc_reinforce_trainer import TCREINFORCETrainer

LSTM_CKPT     = "results/checkpoints/rev_gnn_lstm.pt"
TC_LSTM_CKPT  = "results/checkpoints/rev_gnn_lstm_tc_best.pt"
LOG_DIR       = "results/logs"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",        default="configs/experiments/time_critical.yaml")
    parser.add_argument("--n-train-graphs",type=int, default=10)
    parser.add_argument("--epochs",        type=int, default=None,
                        help="Override cfg.training.reinforce_epochs_phase2")
    args = parser.parse_args()

    ensure_dir(LOG_DIR)
    ensure_dir("results/checkpoints")
    cfg    = load_config_with_base(args.config)
    if args.epochs:
        from omegaconf import OmegaConf
        OmegaConf.update(cfg, "training.reinforce_epochs_phase2", args.epochs)

    set_seed(cfg.project.seed)
    device = torch.device("cpu")

    logger = ExperimentLogger(
        experiment_name="tc_lstm_training",
        log_dir=LOG_DIR,
        cfg=cfg,
    )
    logger.info("TC LSTM fine-tuning: warm-start + TC-REINFORCE")
    logger.info(f"  Source checkpoint: {LSTM_CKPT}")
    logger.info(f"  Target checkpoint: {TC_LSTM_CKPT}")

    # Load base LSTM policy
    policy = load_lstm(LSTM_CKPT, cfg, device)
    logger.info(f"  Parameters: {sum(p.numel() for p in policy.parameters()):,}")

    # Build training graphs (FF n ∈ [300, 600])
    p_ff, pb_ff = cfg.graph.p, cfg.graph.pb
    train_graphs = []
    rng = __import__("numpy").random.default_rng(cfg.project.seed)
    for i in range(args.n_train_graphs):
        n = int(rng.integers(300, 601))
        train_graphs.append(generate_forest_fire(n, p_ff, pb_ff, seed=42 + i))
    logger.info(f"  Training graphs: {args.n_train_graphs} × FF n∈[300,600]")

    # TC-REINFORCE trainer
    trainer = TCREINFORCETrainer(policy, cfg, logger, device)
    t0 = time.time()
    policy.train()
    result  = trainer.train(train_graphs, n_epochs=args.epochs)
    elapsed = time.time() - t0

    # Save TC-trained checkpoint
    torch.save({
        "model_state_dict": policy.state_dict(),
        "best_reward":      result["best_reward"],
        "cfg":              dict(cfg),
        "epochs_trained":   len(result["losses"]),
    }, TC_LSTM_CKPT)

    logger.info(f"\nTC fine-tuning done in {elapsed/60:.1f} min")
    logger.info(f"  Best TC reward: {result['best_reward']:.4f}")
    logger.info(f"  Saved: {TC_LSTM_CKPT}")

    print(f"\n{'='*55}")
    print(f"TC LSTM training complete ({elapsed/60:.1f} min)")
    print(f"  Best TC reward: {result['best_reward']:.4f}")
    print(f"  Checkpoint:     {TC_LSTM_CKPT}")
    print(f"\nNext: run experiments/run_time_critical.py to compare all methods")


if __name__ == "__main__":
    main()
