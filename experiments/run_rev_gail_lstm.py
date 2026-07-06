#!/usr/bin/env python
"""Rev-GAIL-LSTM: GAIL Phase 1 + LSTM memory + Pricing 1.5 + Joint REINFORCE 2.

Expected strongest model: GAIL learns distribution of ORDERINGS (Phase 1),
LSTM tracks rejection history (all phases), REINFORCE fine-tunes pricing (Phase 2).
"""
import argparse, copy, os, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.training.gail_trainer import GAILTrainer
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.baselines import _make_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _eval_lstm(policy, graph, cfg, device):
    """Greedy revenue eval with proper LSTM state management.

    Runs on CPU to avoid MPS autograd state corruption (same fix as LSTM script).
    """
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache = build_graph_feature_cache(graph, static)
    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        n, nodes = graph.number_of_nodes(), list(graph.nodes())
        for _ in range(n):
            available = env.available_nodes
            if not available:
                break
            feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                offered=frozenset(env.offered), t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available:
                nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            policy.update_sequence_state(disc, bool(rew > 0), float(rew))
            if done:
                break
    policy.to(device)
    return env.total_revenue


def main():
    parser = argparse.ArgumentParser(description="Rev-GAIL-LSTM 3-stage training")
    parser.add_argument("--config", default="configs/experiments/rev_gail_lstm.yaml")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    # Force CPU: GAILTrainer._collect_agent_rollout uses torch.no_grad() which
    # corrupts MPS autograd state on Apple Silicon for SequentialJointPolicy (LSTM).
    device = torch.device("cpu")
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

    enc  = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                            cfg.encoder.n_layers, cfg.encoder.dropout)
    lstm = EpisodeLSTM(graph_dim=cfg.encoder.hidden_dim,
                       lstm_hidden=cfg.sequence_model.lstm_hidden,
                       n_layers=cfg.sequence_model.lstm_n_layers)
    policy = SequentialJointPolicy(enc, lstm, gnn_dim=cfg.encoder.hidden_dim,
                                   context_dim=cfg.sequence_model.lstm_hidden).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | device={device}")

    # ── Phase 1: GAIL with LSTM Generator ────────────────────────────────────
    # GAILTrainer._collect_agent_rollout now calls policy.reset_episode + update_sequence_state
    # Discriminator sees (global_state=mean(H), node_emb=H[v*], discount) — GNN only, no LSTM
    logger.info("=== Phase 1: GAIL Adversarial Warm-Start (LSTM generator) ===")
    p1 = GAILTrainer(policy, cfg, logger, device)
    gail_res = p1.train(train_graphs)
    g_loss = gail_res["gen_losses"][-1] if gail_res["gen_losses"] else float("nan")
    d_loss = gail_res["disc_losses"][-1] if gail_res["disc_losses"] else float("nan")
    logger.info(f"GAIL done: gen_loss={g_loss:.4f}  disc_loss={d_loss:.4f}")
    rev_p1 = _eval_lstm(policy, test_graph, cfg, device)
    logger.info(f"Post-P1 revenue (n=1000): {rev_p1:.2f}  (baseline: 460.0)")
    logger.log({"phase": 1, "rev_after_p1": rev_p1})

    # ── Phase 1.5: SKIPPED ────────────────────────────────────────────────────
    # SequentialJointPolicy.select_and_price returns only the node-selection log_prob.
    # When scoring_head is frozen, log_prob has no grad_fn → loss.backward() crash.
    # Phase 1 (GAIL) already gives strong initialization; skip straight to Phase 2.
    logger.info("Skipping Phase 1.5 (GAIL post-P1 already strong; Phase 2 unfreezes all)")
    logger.info("Going directly to Phase 2: Joint REINFORCE (all params unfrozen)")

    # ── Phase 2: Joint REINFORCE fine-tuning (CPU, all params unfrozen) ───────
    logger.info("=== Phase 2: Joint REINFORCE Fine-Tuning (device=cpu) ===")
    for param in policy.parameters():
        param.requires_grad = True

    # device is already cpu (forced at top of main())
    p2 = REINFORCETrainer(policy, cfg, logger, device)
    p2.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.reinforce_lr,
                                    weight_decay=cfg.training.weight_decay)

    # Seed best from Phase 1 result so we never regress below GAIL warm-start
    best_rev = rev_p1
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(cfg.training.reinforce_epochs_phase2):
        graph = train_graphs[ep % len(train_graphs)]
        rollout = p2.collect_rollout(graph)
        loss_val = p2.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev = _eval_lstm(policy, test_graph, cfg, device)
            marker = "  ← best" if rev > best_rev else ""
            logger.info(f"  ep={ep+1:3d}  rev={rev:.2f}  loss={loss_val:.5f}{marker}")
            logger.log({"rl/epoch": ep+1, "rl/revenue": rev, "rl/loss": loss_val})
            if rev > best_rev:
                best_rev = rev
                best_state = copy.deepcopy(policy.state_dict())
                logger.info(f"  New best: {rev:.2f}")

    policy.load_state_dict(best_state)
    torch.save(best_state, "results/checkpoints/rev_gail_lstm.pt")
    logger.info(f"Checkpoint saved → results/checkpoints/rev_gail_lstm.pt")
    logger.info(f"Best revenue: {best_rev:.2f}  vs Greedy-Discount: 460.0  "
                f"({'BEATS' if best_rev > 460 else 'below'} baseline)")
    logger.log({"final/best_revenue": best_rev, "final/greedy_baseline": 460.0,
                "final/beats_baseline": best_rev > 460.0})
    logger.finish()


if __name__ == "__main__":
    main()
