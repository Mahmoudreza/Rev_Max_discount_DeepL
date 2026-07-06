#!/usr/bin/env python
"""
experiments/eval_idea1.py  —  Idea-1 robustness & generalisation evaluation suite.

Runs 4 tasks sequentially using the saved Rev-GNN-LSTM checkpoint.
No retraining — pure evaluation.

Usage:
    python experiments/eval_idea1.py --config configs/experiments/rev_gnn_lstm.yaml
"""
import argparse, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger
from src.evaluation.idea1_eval import (
    load_lstm_policy, load_im_policy,
    task1_robustness, task2_generalisation,
    task3_nonmonotone, task4_ablation,
    print_and_save_summary,
)

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LSTM_CKPT = "results/checkpoints/rev_gnn_lstm.pt"
IM_CKPT   = "results/checkpoints/rev_gnn_im_rl.pt"


def main():
    parser = argparse.ArgumentParser(description="Idea-1 evaluation suite")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_lstm.yaml")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--seeds-t1", type=int, default=20, help="Seeds for Task 1 & 4")
    parser.add_argument("--seeds-t2", type=int, default=5,  help="Seeds for Task 2 & 3")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = torch.device("cpu")   # CPU for reproducibility; no MPS for no_grad evals
    logger = ExperimentLogger(cfg, run_name="idea1_eval")
    ensure_dir("results/logs")

    logger.info(f"Checkpoints: {LSTM_CKPT}  |  {IM_CKPT}")
    logger.info(f"Device: {device}  |  Seeds T1/T4={args.seeds_t1}  T2/T3={args.seeds_t2}")

    # Load policies once — reused across all tasks
    logger.info("Loading Rev-GNN-LSTM checkpoint...")
    lstm_policy = load_lstm_policy(LSTM_CKPT, cfg, device)
    logger.info(f"LSTM policy: {sum(p.numel() for p in lstm_policy.parameters()):,} params")

    logger.info("Loading Rev-GNN-IM-RL checkpoint...")
    im_policy = load_im_policy(IM_CKPT, cfg, device)
    logger.info(f"IM policy:   {sum(p.numel() for p in im_policy.parameters()):,} params")

    # ── Task 1 ──────────────────────────────────────────────────────────────
    t1 = task1_robustness(
        lstm_policy, cfg, device, logger,
        n_seeds=args.seeds_t1,
        out_path="results/logs/robustness_20seeds.csv",
    )

    # ── Task 2 ──────────────────────────────────────────────────────────────
    t2 = task2_generalisation(
        lstm_policy, cfg, device, logger,
        n_seeds=args.seeds_t2,
        data_dir=args.data_dir,
        out_path="results/logs/generalization_eval.csv",
    )

    # ── Task 3 ──────────────────────────────────────────────────────────────
    t3 = task3_nonmonotone(
        lstm_policy, cfg, device, logger,
        n_seeds=args.seeds_t2,
        out_path="results/logs/nonmonotone_eval.csv",
    )

    # ── Task 4 ──────────────────────────────────────────────────────────────
    t4 = task4_ablation(
        lstm_policy, im_policy, cfg, device, logger,
        n_seeds=args.seeds_t1,
        out_path="results/logs/ablation_lstm_vs_no_lstm.csv",
    )

    # ── Summary ─────────────────────────────────────────────────────────────
    print_and_save_summary(t1, t2, t3, t4, logger,
                           out_path="results/logs/idea1_evaluation_summary.json")
    logger.finish()


if __name__ == "__main__":
    main()
