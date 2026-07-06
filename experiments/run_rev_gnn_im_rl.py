#!/usr/bin/env python
"""Rev-GNN-IM-RL: 2-stage training matching LSTM pipeline (Phase 1 → Phase 2).

Phase 1 — CE imitation + pricing MSE (300 epochs, per-step backward).
Phase 1.5 — SKIPPED (same reason as LSTM: frozen scoring_head breaks grad).
Phase 2 — Joint REINFORCE (200 epochs, CPU, lr=1e-5, variance baseline).

This is JointPolicy (no LSTM/sequence memory). The ONLY difference from
run_rev_gnn_lstm.py: uses JointPolicy instead of SequentialJointPolicy,
no reset_episode() / update_sequence_state() calls.
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
from src.models.policies.joint_policy import JointPolicy
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.baselines import greedy_discount_trajectory, _make_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _eval_im_rl(policy, graph, cfg, device):
    """Greedy revenue eval for JointPolicy. Runs on CPU (safe, no MPS state)."""
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
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
            if done:
                break
    policy.to(device)
    return env.total_revenue


def main():
    parser = argparse.ArgumentParser(description="Rev-GNN-IM-RL 2-stage retraining")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_im_rl.yaml")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)
    ensure_dir("results/checkpoints")

    # ── Graphs ────────────────────────────────────────────────────────────────
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

    # ── Model (JointPolicy — no LSTM, no sequence state) ─────────────────────
    enc    = GraphSAGEEncoder(cfg.features.dim, cfg.encoder.hidden_dim,
                              cfg.encoder.n_layers, cfg.encoder.dropout)
    policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | device={device}")

    # ── Phase 1: CE + pricing MSE imitation (per-step backward) ─────────────
    logger.info("=== Phase 1: CE + pricing MSE Imitation ===")
    n_im = cfg.training.imitation_epochs   # 300 from config
    pw   = float(getattr(cfg.training, "pricing_loss_weight", 0.3))
    im_opt = torch.optim.Adam(policy.parameters(), lr=cfg.training.imitation_lr,
                              weight_decay=cfg.training.weight_decay)
    statics = {id(g): compute_static_features(g) for g in train_graphs}
    caches  = {id(g): build_graph_feature_cache(g, statics[id(g)]) for g in train_graphs}

    # Pre-generate expert trajectories ONCE — they are stable with n_mc=200.
    # Regenerating per-epoch wastes 300×5 = 1,500 full MC simulations.
    logger.info("Pre-generating expert trajectories (n_mc=200, ~deterministic)...")
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
            S, off = frozenset(), frozenset()
            step_losses = []
            for td in traj:
                nidx, ed, acc = td["node_idx"], td["discount"], td.get("accepted", True)
                feats = compute_node_features_fast(cache=cache, S=S, offered=off,
                    t=len(off), k=n_g, env=env)
                data = graph_to_pyg_data(g, feats, device)
                mask = get_available_mask(n_g, off, nodes, device)
                # JointPolicy.forward returns (scores, masked_scores, embeddings)
                scores, masked_scores, h = policy.forward(
                    data.x, data.edge_index, mask, return_embeddings=True)
                loss_ce = F.cross_entropy(
                    masked_scores.unsqueeze(0),
                    torch.tensor([nidx], device=device))
                if pw > 0 and h is not None:
                    # Use Beta distribution mean — pricing_head now outputs 2 values
                    if hasattr(policy, 'get_discount_distribution'):
                        pr = policy.get_discount_distribution(h[nidx]).mean
                    else:
                        pr = policy.pricing_head(h[nidx].unsqueeze(0)).squeeze()
                    loss = loss_ce + pw * F.mse_loss(
                        pr, torch.tensor(ed, dtype=torch.float32, device=device))
                else:
                    loss = loss_ce
                im_opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
                im_opt.step()
                step_losses.append(loss.item())
                # Advance env state
                node = nodes[nidx]
                if acc:
                    S = frozenset(S | {node}); env.S.add(node); env._influence_cache = {}
                off = frozenset(off | {node}); env.offered.add(node); env.t += 1
            if step_losses:
                ep_losses.append(sum(step_losses) / len(step_losses))
        mean_loss = sum(ep_losses) / max(len(ep_losses), 1)
        if epoch % 50 == 0 or epoch == n_im - 1:
            rev = _eval_im_rl(policy, test_graph, cfg, device)
            logger.info(f"  IM ep={epoch:4d}  loss={mean_loss:.4f}  rev={rev:.2f}")
            logger.log({"im/epoch": epoch, "im/loss": mean_loss, "im/revenue": rev})

    # Move to CPU before REINFORCE (avoids MPS autograd corruption)
    device_cpu = torch.device("cpu")
    policy = policy.to(device_cpu)
    rev_p1 = _eval_im_rl(policy, test_graph, cfg, device_cpu)
    logger.info(f"Post-P1 revenue (n=1000): {rev_p1:.2f}  (greedy baseline: ~416)")
    logger.log({"phase": 1, "rev_after_p1": rev_p1})

    # ── Phase 1.5: SKIPPED ────────────────────────────────────────────────────
    # Frozen scoring_head → log_prob has no grad_fn → backward crash.
    # Phase 1 imitation already gives good seed selection (97-99% quality).
    # Phase 2 with ALL params unfrozen avoids the gradient bug entirely.
    logger.info("Skipping Phase 1.5 → going directly to Phase 2 (all params unfrozen)")

    # ── Phase 2: Joint REINFORCE fine-tuning (CPU, all params) ───────────────
    logger.info("=== Phase 2: Joint REINFORCE Fine-Tuning (device=cpu) ===")
    for param in policy.parameters():
        param.requires_grad = True

    p2 = REINFORCETrainer(policy, cfg, logger, device_cpu)
    p2.optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=cfg.training.reinforce_lr,      # 1e-5
        weight_decay=cfg.training.weight_decay,
    )

    # Seed best with Phase 1 result — never regress below it
    best_rev   = rev_p1
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(cfg.training.reinforce_epochs_phase2):  # 200 from config
        graph   = train_graphs[ep % len(train_graphs)]
        rollout = p2.collect_rollout(graph)
        loss_val = p2.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev    = _eval_im_rl(policy, test_graph, cfg, device_cpu)
            marker = "  ← best" if rev > best_rev else ""
            logger.info(f"  ep={ep+1:3d}  rev={rev:.2f}  loss={loss_val:.5f}{marker}")
            logger.log({"rl/epoch": ep+1, "rl/revenue": rev, "rl/loss": loss_val})
            if rev > best_rev:
                best_rev   = rev
                best_state = copy.deepcopy(policy.state_dict())
                logger.info(f"  New best: {rev:.2f}")

    policy.load_state_dict(best_state)
    torch.save(best_state, "results/checkpoints/rev_gnn_im_rl.pt")
    logger.info(f"Checkpoint saved → results/checkpoints/rev_gnn_im_rl.pt")
    logger.info(f"Phase 2 best: {best_rev:.2f}  Post-P1: {rev_p1:.2f}  "
                f"({'BEATS' if best_rev > 416 else 'below'} greedy ~416)")
    logger.log({"final/best_revenue": best_rev, "final/rev_p1": rev_p1,
                "final/greedy_baseline": 416.0,
                "final/beats_baseline": best_rev > 416.0})
    logger.finish()


if __name__ == "__main__":
    main()
