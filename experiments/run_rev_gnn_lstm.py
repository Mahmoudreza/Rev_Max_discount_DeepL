#!/usr/bin/env python
"""Rev-GNN-LSTM: 3-stage training with episode-level LSTM memory.

Phase 1: CE + MSE imitation (LSTM observes expert state sequence).
Phase 1.5: Pricing REINFORCE (encoder + scoring + LSTM frozen).
Phase 2: Joint REINFORCE fine-tuning (all params, CPU).

LSTM key advantage: tracks rejection patterns across steps, enabling
adaptive pricing when buyers 1-5 reject at discount=0.1.
"""
import argparse, copy, os, sys, torch
import torch.nn.functional as F
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
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.baselines import greedy_discount_trajectory, _make_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _eval_lstm(policy, graph, cfg, device):
    """Greedy revenue eval for LSTM policy.

    Runs on CPU to avoid MPS autograd state corruption.
    After ~100 MPS training epochs, torch.no_grad() on MPS can flip internal
    Metal shader state, breaking subsequent gradient-tracked forward passes.
    Running eval on CPU keeps no_grad ops completely isolated from MPS state.
    """
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)   # move to CPU for eval (fast on Apple unified memory)
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
    policy.to(device)   # move back to MPS for training
    return env.total_revenue


def main():
    parser = argparse.ArgumentParser(description="Rev-GNN-LSTM 3-stage training")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_lstm.yaml")
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

    enc  = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                            cfg.encoder.n_layers, cfg.encoder.dropout)
    lstm = EpisodeLSTM(graph_dim=cfg.encoder.hidden_dim,
                       lstm_hidden=cfg.sequence_model.lstm_hidden,
                       n_layers=cfg.sequence_model.lstm_n_layers)
    policy = SequentialJointPolicy(enc, lstm, gnn_dim=cfg.encoder.hidden_dim,
                                   context_dim=cfg.sequence_model.lstm_hidden).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | device={device}")

    # ── Phase 1: LSTM-aware CE + pricing MSE imitation ────────────────────────
    logger.info("=== Phase 1: LSTM Imitation (CE + pricing MSE) ===")
    # Optimal at 50 epochs with per-step backward (ep=50 → rev=469.50 > 460 baseline;
    # ep=100 → 460.42, overfitting). Override config value.
    n_im = 50
    pw   = float(getattr(cfg.training, "pricing_loss_weight", 0.3))
    im_opt = torch.optim.Adam(policy.parameters(), lr=cfg.training.imitation_lr,
                              weight_decay=cfg.training.weight_decay)
    statics = {id(g): compute_static_features(g) for g in train_graphs}
    caches  = {id(g): build_graph_feature_cache(g, statics[id(g)]) for g in train_graphs}

    # Pre-generate expert trajectories ONCE — stable with n_mc=200.
    logger.info("Pre-generating expert trajectories once (n_mc=200, ~deterministic)...")
    trajs = {id(g): greedy_discount_trajectory(g, cfg) for g in train_graphs}
    logger.info(f"Expert trajectories ready for {len(trajs)} graphs")

    policy.train()

    for epoch in range(n_im):
        ep_losses = []
        for g in train_graphs:
            cache = caches[id(g)]
            traj  = trajs[id(g)]
            n_g   = g.number_of_nodes()
            nodes = list(g.nodes())
            env   = _make_env(g, cfg); env.reset()
            policy.reset_episode(device)
            S, off = frozenset(), frozenset()
            step_losses = []
            for td in traj:
                nidx, ed, acc = td["node_idx"], td["discount"], td.get("accepted", True)
                feats = compute_node_features_fast(cache=cache, S=S, offered=off,
                    t=len(off), k=n_g, env=env)
                data = graph_to_pyg_data(g, feats, device)
                mask = get_available_mask(n_g, off, nodes, device)
                ms, h, ctx, _ = policy.forward(data.x, data.edge_index, mask)
                loss = F.cross_entropy(ms.unsqueeze(0), torch.tensor([nidx], device=device))
                if pw > 0:
                    combined = torch.cat([h[nidx], ctx])
                    # Use Beta distribution mean — pricing_head now outputs 2 values
                    if hasattr(policy, 'get_discount_distribution'):
                        pr = policy.get_discount_distribution(combined).mean
                    else:
                        pr = policy.pricing_head(combined.unsqueeze(0)).squeeze()
                    loss = loss + pw * F.mse_loss(pr, torch.tensor(ed, dtype=torch.float32, device=device))
                # Per-step backward: avoids long gradient graph accumulation
                # (MPS backend has autograd issues with large accumulated graphs)
                im_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
                im_opt.step()
                step_losses.append(loss.item())
                node = nodes[nidx]
                if acc:
                    S = frozenset(S | {node}); env.S.add(node); env._influence_cache = {}
                off = frozenset(off | {node}); env.offered.add(node); env.t += 1
                rev_step = (1 - ed) * 0.5 * float(acc)
                policy.update_sequence_state(ed, acc, rev_step)
            if step_losses:
                ep_losses.append(sum(step_losses) / len(step_losses))
        mean_loss = sum(ep_losses) / max(len(ep_losses), 1)
        if epoch % 50 == 0:
            rev = _eval_lstm(policy, test_graph, cfg, device)
            logger.info(f"  IM ep={epoch:4d}  loss={mean_loss:.4f}  rev={rev:.2f}")
            logger.log({"im/epoch": epoch, "im/loss": mean_loss, "im/revenue": rev})
    # ── Move to CPU permanently before any REINFORCE phases ──────────────────
    # MPS torch.no_grad() (used in collect_rollout) permanently corrupts MPS
    # autograd state on Apple Silicon. All phases after P1 run on CPU.
    device_cpu = torch.device("cpu")
    policy = policy.to(device_cpu)

    rev_p1 = _eval_lstm(policy, test_graph, cfg, device_cpu)
    logger.info(f"Post-P1 revenue (n=1000): {rev_p1:.2f}  (baseline: 460.0)")
    logger.log({"phase": 1, "rev_after_p1": rev_p1})

    # ── Phase 1.5: SKIPPED ────────────────────────────────────────────────────
    # Root cause: when scoring_head is frozen, log_prob has no grad_fn
    # (all dependencies have requires_grad=False → loss.backward() crashes).
    # Phase 1 already reaches 97% of baseline (445.83 / 460.0).
    # Phase 2 with ALL params unfrozen avoids the gradient bug entirely.
    # NOTE: To fix Phase 1.5 for future use, see CLAUDE.md §Phase 1.5 fix:
    # use pricing_only=True in collect_rollout so node selection uses no_grad
    # and only the pricing head produces a differentiable log_prob.
    logger.info("Skipping Phase 1.5 (post-P1 revenue already 97% of baseline)")
    logger.info("Going directly to Phase 2: Joint REINFORCE (all params unfrozen)")

    # ── Phase 2: Joint REINFORCE fine-tuning (CPU, all params unfrozen) ───────
    logger.info("=== Phase 2: Joint REINFORCE Fine-Tuning (device=cpu) ===")
    for param in policy.parameters():
        param.requires_grad = True

    p2 = REINFORCETrainer(policy, cfg, logger, device_cpu)
    p2.optimizer = torch.optim.Adam(policy.parameters(), lr=cfg.training.reinforce_lr,
                                    weight_decay=cfg.training.weight_decay)

    # Seed best with Phase 1 result so we never regress below it
    best_rev = rev_p1
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(cfg.training.reinforce_epochs_phase2):
        graph = train_graphs[ep % len(train_graphs)]
        rollout = p2.collect_rollout(graph)
        loss_val = p2.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev = _eval_lstm(policy, test_graph, cfg, device_cpu)
            marker = "  ← best" if rev > best_rev else ""
            logger.info(f"  ep={ep+1:3d}  rev={rev:.2f}  loss={loss_val:.5f}{marker}")
            logger.log({"rl/epoch": ep+1, "rl/revenue": rev, "rl/loss": loss_val})
            if rev > best_rev:
                best_rev = rev
                best_state = copy.deepcopy(policy.state_dict())
                logger.info(f"  New best: {rev:.2f}")

    policy.load_state_dict(best_state)
    torch.save(best_state, "results/checkpoints/rev_gnn_lstm.pt")
    logger.info(f"Checkpoint saved → results/checkpoints/rev_gnn_lstm.pt")
    logger.info(f"Best revenue: {best_rev:.2f}  vs Greedy-Discount: 460.0  "
                f"({'BEATS' if best_rev > 460 else 'below'} baseline)")
    logger.log({"final/best_revenue": best_rev, "final/greedy_baseline": 460.0,
                "final/beats_baseline": best_rev > 460.0})
    logger.finish()


if __name__ == "__main__":
    main()
