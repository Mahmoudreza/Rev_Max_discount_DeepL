#!/usr/bin/env python
"""Evaluate Rev-GNN-IM-RL v4 checkpoint on the rice_facebook real-world graph.

Loads: results/checkpoints/rev_gnn_im_rl_best.pt
Graph: data/processed/rice_facebook.pkl (1,262 nodes, 3,441 edges after preprocessing)
"""
import os, sys, pickle, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.joint_policy import JointPolicy
from src.evaluation.baselines import _make_env, run_all_babaei

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CHECKPOINT = "results/checkpoints/rev_gnn_im_rl_best.pt"
GRAPH_PKL  = "data/processed/rice_facebook.pkl"
CONFIG     = "configs/experiments/rev_gnn_im_rl.yaml"


def eval_policy_greedy(policy, graph, cfg, device):
    """Run greedy episode and return total revenue."""
    env = _make_env(graph, cfg)
    env.reset()
    if hasattr(policy, "reset_episode"):
        policy.reset_episode(device)
    policy.eval()
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    with torch.no_grad():
        for _ in range(n):
            available = env.available_nodes
            if not available:
                break
            feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                offered=frozenset(env.offered), t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available:
                nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            if hasattr(policy, "update_sequence_state"):
                policy.update_sequence_state(disc, bool(rew > 0), float(rew))
            if done:
                break
    return env.total_revenue


def main():
    cfg = load_config_with_base(CONFIG)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name="rice_facebook_eval")

    # Load rice_facebook graph
    with open(GRAPH_PKL, "rb") as f:
        graph = pickle.load(f)
    logger.info(f"rice_facebook: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Run Babaei baselines on rice_facebook for comparison
    logger.info("Running Babaei baselines on rice_facebook (n_trials=3)...")
    babaei = run_all_babaei(graph, cfg, n_trials=3)
    for k, v in babaei.items():
        logger.info(f"  {k:22s}: {v:.2f}")
    logger.log({f"rice_fb/baseline/{k}": v for k, v in babaei.items()})

    # Find checkpoint (look for both naming conventions)
    ckpt_path = None
    for cand in [CHECKPOINT,
                 "results/checkpoints/rev_gnn_im_rl_best.pt",
                 "results/checkpoints/rev_gnn_im_rl.pt"]:
        if Path(cand).exists():
            ckpt_path = cand
            break
    if ckpt_path is None:
        logger.info("ERROR: No checkpoint found. Expected: " + CHECKPOINT)
        logger.info("Run: python experiments/run_rev_gnn_im_rl.py first.")
        return

    # Load policy
    enc = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                           cfg.encoder.n_layers, cfg.encoder.dropout)
    policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
    state = torch.load(ckpt_path, map_location=device)
    policy.load_state_dict(state)
    logger.info(f"Loaded checkpoint: {ckpt_path}")

    # Evaluate 3 times (different seeds for env stochasticity)
    revenues = []
    for trial in range(3):
        set_seed(cfg.project.seed + trial)
        rev = eval_policy_greedy(policy, graph, cfg, device)
        revenues.append(rev)
        logger.info(f"  Trial {trial+1}: rev={rev:.2f}")
    mean_rev = sum(revenues) / len(revenues)
    greedy_disc = babaei.get("greedy_discount", 0.0)

    logger.info(f"\n{'='*50}")
    logger.info(f"Rev-GNN-IM-RL on rice_facebook: {mean_rev:.2f}")
    logger.info(f"Greedy-Discount on rice_facebook: {greedy_disc:.2f}")
    logger.info(f"IE-Strategy on rice_facebook:    {babaei.get('ie_strategy', 0):.2f}")
    logger.info(f"µ-Discount on rice_facebook:      {babaei.get('mu_discount', 0):.2f}")
    rel = 100 * (mean_rev - greedy_disc) / max(greedy_disc, 1e-9)
    logger.info(f"Rev-GNN vs Greedy-Discount: {rel:+.1f}%")
    logger.log({"rice_fb/rev_gnn_im_rl": mean_rev, "rice_fb/greedy_discount": greedy_disc})
    logger.finish()


if __name__ == "__main__":
    main()
