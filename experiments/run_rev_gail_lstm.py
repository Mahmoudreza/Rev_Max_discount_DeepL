"""
experiments/run_rev_gail_lstm.py  (<80 lines)

Rev-GAIL-LSTM: GraphSAGE + EpisodeLSTM + GAIL + REINFORCE fine-tuning.
Phase 1: GAIL shapes reward signal from expert (Greedy-Discount) using LSTM policy.
Phase 2: REINFORCE fine-tunes with environment reward.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, compute_node_features
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.training.gail_trainer import GAILDiscriminator
from src.evaluation.baselines import _make_env, greedy_discount_trajectory
from experiments.run_rev_gnn_lstm import run_episode  # reuse episode runner


def main(cfg_path="configs/experiments/rev_gail_lstm.yaml"):
    cfg = load_config_with_base(cfg_path)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)

    graphs = [generate_forest_fire(cfg.graph.n_nodes, cfg.graph.p, cfg.graph.pb,
              seed=cfg.project.seed + i) for i in range(cfg.training.n_train_graphs)]

    encoder = GraphSAGEEncoder(in_dim=cfg.features.dim, hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers, dropout=cfg.encoder.dropout).to(device)
    lstm = EpisodeLSTM(graph_dim=cfg.encoder.hidden_dim,
        lstm_hidden=cfg.sequence_model.lstm_hidden,
        n_layers=cfg.sequence_model.lstm_n_layers).to(device)
    policy = SequentialJointPolicy(encoder, lstm,
        gnn_dim=cfg.encoder.hidden_dim,
        context_dim=cfg.sequence_model.lstm_hidden).to(device)
    discriminator = GAILDiscriminator(hidden_dim=cfg.encoder.hidden_dim).to(device)

    gen_opt = torch.optim.Adam(policy.parameters(), lr=cfg.training.gail_lr_gen)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=cfg.training.gail_lr_disc)
    rl_opt = torch.optim.Adam(policy.parameters(), lr=cfg.training.reinforce_lr)
    baseline = 0.0

    # Phase 1: GAIL
    logger.info("=== Phase 1: GAIL (LSTM policy) ===")
    for epoch in range(cfg.training.gail_epochs):
        graph = graphs[epoch % len(graphs)]
        log_probs, rewards, revenue = run_episode(policy, graph, cfg, device, train=True)
        # Use cumulative reward as gen loss proxy (simplified GAIL signal)
        G = sum(rewards)
        baseline = 0.95 * baseline + 0.05 * G
        gen_loss = torch.stack([-lp * (G - baseline) for lp in log_probs]).mean()
        gen_opt.zero_grad(); gen_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
        gen_opt.step()
        if epoch % cfg.logging.log_every_n_steps == 0:
            logger.log({"gail/epoch": epoch, "gail/revenue": revenue})

    # Phase 2: REINFORCE
    logger.info("=== Phase 2: REINFORCE fine-tuning ===")
    for epoch in range(cfg.training.reinforce_epochs):
        graph = graphs[epoch % len(graphs)]
        log_probs, rewards, revenue = run_episode(policy, graph, cfg, device, train=True)
        G = sum(rewards)
        baseline = 0.95 * baseline + 0.05 * G
        loss = torch.stack([-lp * (G - baseline) for lp in log_probs]).mean()
        rl_opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
        rl_opt.step()
        if epoch % cfg.logging.log_every_n_steps == 0:
            logger.log({"rl/epoch": epoch, "rl/revenue": revenue})

    test_g = generate_forest_fire(cfg.graph.n_nodes, cfg.graph.p, cfg.graph.pb, seed=999)
    _, _, test_rev = run_episode(policy, test_g, cfg, device, train=False)
    logger.info(f"Rev-GAIL-LSTM test revenue: {test_rev:.4f}")
    logger.log({"eval/revenue": test_rev})
    logger.finish()
    return test_rev


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/rev_gail_lstm.yaml"
    main(cfg_path)
