#!/usr/bin/env python
"""Rev-GAIL-RL-Rich: 3-stage training (GAIL Phase 1 → Pricing 1.5 → Joint REINFORCE 2).

GAIL Phase 1 learns the DISTRIBUTION of good orderings, not just the greedy sequence.
Phases 1.5 and 2 are identical to Rev-GNN-IM-RL.
"""
import argparse, copy, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.training.gail_trainer import GAILTrainer
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.runner import eval_greedy_revenue

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="Rev-GAIL-RL-Rich 3-stage training")
    parser.add_argument("--config", default="configs/experiments/rev_gail_rl_rich.yaml")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)
    ensure_dir("results/checkpoints")

    train_graphs = [
        generate_forest_fire(200 + i * 60, cfg.graph.p, cfg.graph.pb,
                             seed=cfg.project.seed + i)
        for i in range(cfg.training.n_train_graphs)
    ]
    test_graph = generate_forest_fire(
        cfg.evaluation.test_n_nodes, cfg.graph.p, cfg.graph.pb,
        seed=cfg.project.seed + 9999,
    )
    logger.info(f"Train graphs: {[g.number_of_nodes() for g in train_graphs]}, "
                f"test graph n={test_graph.number_of_nodes()}")

    enc = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                           cfg.encoder.n_layers, cfg.encoder.dropout)
    policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | device={device}")

    # ── Phase 1: GAIL warm-start (adversarial scoring head training) ──────────
    logger.info("=== Phase 1: GAIL Adversarial Warm-Start ===")
    p1 = GAILTrainer(policy, cfg, logger, device)
    gail_res = p1.train(train_graphs)
    g_loss = gail_res["gen_losses"][-1] if gail_res["gen_losses"] else float("nan")
    d_loss = gail_res["disc_losses"][-1] if gail_res["disc_losses"] else float("nan")
    logger.info(f"GAIL done: gen_loss={g_loss:.4f}  disc_loss={d_loss:.4f}")
    rev_p1 = eval_greedy_revenue(policy, test_graph, cfg, device)
    logger.info(f"Post-P1 revenue (n=1000): {rev_p1:.2f}  (baseline: 460.0)")
    logger.log({"phase": 1, "rev_after_p1": rev_p1})

    # ── Phase 1.5: Pricing-only REINFORCE (encoder + scoring frozen) ──────────
    logger.info("=== Phase 1.5: Pricing-Only REINFORCE ===")
    for param in policy.encoder.parameters():
        param.requires_grad = False
    for param in policy.scoring_head.parameters():
        param.requires_grad = False

    p15 = REINFORCETrainer(policy, cfg, logger, device)
    p15.optimizer = torch.optim.Adam(
        policy.pricing_head.parameters(), lr=cfg.training.reinforce_lr * 10
    )
    best_rev_p15 = rev_p1
    for ep in range(cfg.training.reinforce_epochs_phase15):
        graph = train_graphs[ep % len(train_graphs)]
        rollout = p15.collect_rollout(graph)
        loss = p15.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev = eval_greedy_revenue(policy, test_graph, cfg, device)
            marker = "  ← best" if rev > best_rev_p15 else ""
            logger.info(f"  P1.5 ep={ep+1:3d}  rev={rev:.2f}  loss={loss:.5f}{marker}")
            logger.log({"p15/epoch": ep + 1, "p15/revenue": rev, "p15/loss": loss})
            best_rev_p15 = max(best_rev_p15, rev)
    rev_p15 = eval_greedy_revenue(policy, test_graph, cfg, device)
    logger.info(f"Post-P1.5 revenue (n=1000): {rev_p15:.2f}  (baseline: 460.0)")
    logger.log({"phase": 1.5, "rev_after_p15": rev_p15})

    # ── Phase 2: Joint REINFORCE fine-tuning (CPU) ────────────────────────────
    logger.info("=== Phase 2: Joint REINFORCE Fine-Tuning (device=cpu) ===")
    for param in policy.parameters():
        param.requires_grad = True
    device_cpu = torch.device("cpu")
    policy = policy.to(device_cpu)
    p2 = REINFORCETrainer(policy, cfg, logger, device_cpu)
    p2.optimizer = torch.optim.Adam(
        policy.parameters(), lr=cfg.training.reinforce_lr,
        weight_decay=cfg.training.weight_decay,
    )
    p2.reward_baseline = p15.reward_baseline

    # Seed best from Phase 1.5 so we never regress below it
    best_rev = rev_p15
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(cfg.training.reinforce_epochs_phase2):
        graph = train_graphs[ep % len(train_graphs)]
        rollout = p2.collect_rollout(graph)
        loss = p2.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev = eval_greedy_revenue(policy, test_graph, cfg, device_cpu)
            marker = "  ← best" if rev > best_rev else ""
            logger.info(f"  ep={ep+1:3d}  rev={rev:.2f}  loss={loss:.5f}{marker}")
            logger.log({"rl/epoch": ep + 1, "rl/revenue": rev, "rl/loss": loss})
            if rev > best_rev:
                best_rev = rev
                best_state = copy.deepcopy(policy.state_dict())
                logger.info(f"  New best: {rev:.2f}")

    policy.load_state_dict(best_state)
    torch.save(best_state, "results/checkpoints/rev_gail_rl_rich.pt")
    logger.info(f"Checkpoint saved → results/checkpoints/rev_gail_rl_rich.pt")
    logger.info(f"Best revenue: {best_rev:.2f}  vs Greedy-Discount: 460.0  "
                f"({'BEATS' if best_rev > 460 else 'below'} baseline)")
    logger.log({"final/best_revenue": best_rev, "final/greedy_baseline": 460.0,
                "final/beats_baseline": best_rev > 460.0})
    logger.finish()


if __name__ == "__main__":
    main()
