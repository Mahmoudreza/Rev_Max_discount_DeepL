"""
experiments/run_rev_gail.py  (<80 lines)

Train Rev-GAIL-RL-Rich: GAIL pre-training + REINFORCE.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_graph
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.training.gail_trainer import GAILTrainer
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.evaluate import evaluate_policy


def main(cfg_path: str = "configs/experiments/rev_gail_rl_rich.yaml") -> None:
    cfg = load_config_with_base(cfg_path)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)

    train_graphs = [
        generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes, seed=cfg.project.seed + i)
        for i in range(cfg.training.n_train_graphs)
    ]
    test_graphs = [
        generate_graph(cfg, graph_type="ba", n=cfg.graph.n_nodes, seed=1000 + i)
        for i in range(cfg.evaluation.n_test_graphs)
    ]

    encoder = GraphSAGEEncoder(
        in_dim=cfg.graph.node_feature_dim,
        hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers,
        dropout=cfg.encoder.dropout,
    ).to(device)
    policy = JointPolicy(encoder, cfg).to(device)

    logger.info("=== Phase 1: GAIL ===")
    gail_trainer = GAILTrainer(policy, cfg, logger, device)
    gail_trainer.train(train_graphs)

    logger.info("=== Phase 2: REINFORCE ===")
    rl_trainer = REINFORCETrainer(policy, cfg, logger, device)
    rl_trainer.train(train_graphs)

    results = evaluate_policy(policy, test_graphs, cfg, device)
    logger.log({"eval/mean_revenue": results["mean_revenue"]})
    logger.finish()


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/rev_gail_rl_rich.yaml"
    main(cfg_path)
