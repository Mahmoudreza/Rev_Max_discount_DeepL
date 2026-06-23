"""
experiments/run_rev_gnn_im_rl.py  (<80 lines)

Train Rev-GNN-IM-RL: Phase 1 (Imitation) + Phase 2 (REINFORCE).
Experiment: GraphSAGE encoder with joint pricing head.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_graph
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.training.imitation_trainer import ImitationTrainer
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.evaluate import evaluate_policy


def main(cfg_path: str = "configs/experiments/rev_gnn_im_rl.yaml") -> None:
    cfg = load_config_with_base(cfg_path)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)

    # Build train graphs
    train_graphs = [
        generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes, seed=cfg.project.seed + i)
        for i in range(cfg.training.n_train_graphs)
    ]
    test_graphs = [
        generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes, seed=1000 + i)
        for i in range(cfg.evaluation.n_test_graphs)
    ]

    # Build model
    encoder = GraphSAGEEncoder(
        in_dim=cfg.graph.node_feature_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    policy = JointPolicy(encoder, cfg).to(device)
    logger.info(f"Policy params: {sum(p.numel() for p in policy.parameters()):,}")

    # Phase 1: Imitation
    logger.info("=== Phase 1: Imitation Learning ===")
    im_trainer = ImitationTrainer(policy, cfg, logger, device)
    im_trainer.train(train_graphs)

    # Phase 2: REINFORCE
    logger.info("=== Phase 2: REINFORCE Fine-tuning ===")
    rl_trainer = REINFORCETrainer(policy, cfg, logger, device)
    rl_trainer.train(train_graphs)

    # Evaluation
    results = evaluate_policy(policy, test_graphs, cfg, device)
    logger.log({"eval/mean_revenue": results["mean_revenue"],
                "eval/std_revenue": results["std_revenue"]})
    logger.info(f"Test revenue: {results['mean_revenue']:.4f} ± {results['std_revenue']:.4f}")

    logger.finish()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/rev_gnn_im_rl.yaml"
    main(cfg_path)
