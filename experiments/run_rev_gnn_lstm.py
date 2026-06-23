"""
experiments/run_rev_gnn_lstm.py  (<80 lines)

Rev-GNN-LSTM: GraphSAGE + EpisodeLSTM + REINFORCE.
Sequence model maintains episode-level context across the n selling steps.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, compute_node_features
from src.env.graph_generators import generate_forest_fire, build_graph_from_config
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.evaluation.baselines import _make_env
from src.evaluation.evaluate import evaluate_policy


def run_episode(policy, graph, cfg, device, train=True):
    """Run one episode with SequentialJointPolicy, return rollout data."""
    env = _make_env(graph, cfg)
    obs = env.reset()
    policy.reset_episode(device)
    static = compute_static_features(graph)
    n = graph.number_of_nodes()
    nodes = list(graph.nodes())
    log_probs, rewards = [], []
    for _ in range(n):
        available = env.available_nodes
        if not available:
            break
        feats = compute_node_features(graph=graph, static_features=static,
            S=frozenset(env.S), offered=frozenset(env.offered),
            t=env.t, n=n, k=n, env=env)
        data = graph_to_pyg_data(graph, feats, device)
        mask = get_available_mask(n, frozenset(env.offered), nodes, device)
        node_idx, discount, log_prob = policy.select_and_price(
            data.x, data.edge_index, mask, greedy=not train)
        if node_idx not in available:
            node_idx = available[0]
        _, reward, done, info = env.step(node_idx, discount)
        accepted = info.get("accepted", reward > 0)
        policy.update_sequence_state(discount, accepted, float(reward))
        log_probs.append(log_prob)
        rewards.append(reward)
        if done:
            break
    return log_probs, rewards, env.total_revenue


def main(cfg_path="configs/experiments/rev_gnn_lstm.yaml"):
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
    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.reinforce_lr)
    baseline = 0.0

    for epoch in range(cfg.training.reinforce_epochs):
        graph = graphs[epoch % len(graphs)]
        log_probs, rewards, revenue = run_episode(policy, graph, cfg, device)
        G = sum(rewards)
        baseline = 0.95 * baseline + 0.05 * G
        loss = torch.stack([-lp * (G - baseline) for lp in log_probs]).mean()
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
        optimizer.step()
        if epoch % cfg.logging.log_every_n_steps == 0:
            logger.log({"epoch": epoch, "revenue": revenue, "loss": float(loss)})

    test_g = generate_forest_fire(cfg.graph.n_nodes, cfg.graph.p, cfg.graph.pb, seed=999)
    _, _, test_rev = run_episode(policy, test_g, cfg, device, train=False)
    logger.info(f"Rev-GNN-LSTM test revenue: {test_rev:.4f}")
    logger.log({"eval/revenue": test_rev})
    logger.finish()
    return test_rev


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "configs/experiments/rev_gnn_lstm.yaml"
    main(cfg_path)
