#!/usr/bin/env python
"""Rev-GNN-Transformer: GraphSAGE + ALiBi Sliding-Window Transformer.

Mirrors run_rev_gnn_lstm.py exactly — ONLY the sequence module changes.

Phase 1: CE + pricing MSE imitation (300 epochs, per-step backward).
Phase 1.5: SKIPPED (same reason as LSTM: gradient bug when heads frozen).
Phase 2: Joint REINFORCE (200 epochs, lr=1e-5, Welford, CPU for MPS safety).

Hypothesis: Transformer >= LSTM on large n (n=2000) and OOD topologies
(Modular FF, Rice-Facebook), due to direct attention to early rejections.

Checkpoint: results/checkpoints/rev_gnn_transformer.pt
Log:       /tmp/rev_gnn_transformer.log (when run in background)

Usage:
  cd revmax-aaai2027 && source venv/bin/activate
  python -u experiments/run_rev_gnn_transformer.py \
    --config configs/experiments/rev_gnn_transformer.yaml
"""

import argparse, copy, os, sys, torch
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import (load_config_with_base, set_seed, get_device, ensure_dir,
                                graph_to_pyg_data, get_available_mask)
from src.utils.logging import ExperimentLogger
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.models.policies.transformer_joint_policy import TransformerJointPolicy
from src.training.reinforce_trainer import REINFORCETrainer
from src.evaluation.baselines import greedy_discount_trajectory, _make_env

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GREEDY_BASELINE = 460.0   # same as LSTM baseline


# ── Helpers ────────────────────────────────────────────────────────────────────

def _eval_transformer(policy, graph, cfg, device):
    """Greedy revenue on a single graph (no sampling)."""
    from src.env.revenue_env import RevenueEnv
    policy.eval()
    n     = graph.number_of_nodes()
    nodes = list(graph.nodes())
    statics = compute_static_features(graph)
    cache   = build_graph_feature_cache(graph, statics)
    env = _make_env(graph, cfg); env.reset()
    policy.reset_episode(device)
    S, off = frozenset(), frozenset()
    revenue = 0.0
    with torch.no_grad():
        for _ in range(n):
            feats = compute_node_features_fast(cache=cache, S=S, offered=off,
                t=len(off), k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, off, nodes, device)
            if mask.sum() == 0:
                break
            ni, disc, _ = policy.select_and_price(data.x, data.edge_index, mask,
                                                   greedy=True)
            node = nodes[int(ni)]
            val  = env._true_valuation(node)
            d    = float(disc)
            price = val * (1 - d)
            acc   = val >= price
            off = frozenset(off | {node}); env.offered.add(node); env.t += 1
            if acc:
                S = frozenset(S | {node}); env.S.add(node)
                env._influence_cache = {}
                revenue += price
            policy.update_sequence_state(d, acc, price if acc else 0.0)
    policy.train()
    return revenue


def _sanity_gate(policy, graph_small, device, logger):
    """10 greedy steps on n=50: check discounts are varied in (0,1).

    Returns True if gate passes (no discount collapse), False otherwise.
    Runs BEFORE Phase 1 and AFTER Phase 1 to verify training progress.
    """
    import numpy as np
    policy.eval()
    n     = graph_small.number_of_nodes()
    nodes = list(graph_small.nodes())
    statics = compute_static_features(graph_small)
    cache   = build_graph_feature_cache(graph_small, statics)
    env = _make_env(graph_small, None, use_default_cfg=True)
    env.reset()
    policy.reset_episode(device)
    off = frozenset()
    discounts = []
    with torch.no_grad():
        for step in range(min(10, n)):
            feats = compute_node_features_fast(cache=cache, S=frozenset(), offered=off,
                t=len(off), k=n, env=env)
            data = graph_to_pyg_data(graph_small, feats, device)
            mask = get_available_mask(n, off, nodes, device)
            if mask.sum() == 0:
                break
            _, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            discounts.append(float(disc))
            off = frozenset(off | {nodes[0]})   # dummy step
    policy.train()
    if not discounts:
        return False
    d_arr = np.array(discounts)
    varied = d_arr.std() > 1e-3 and 0 < d_arr.mean() < 1
    logger.info(f"  Sanity gate discounts: {[f'{d:.3f}' for d in discounts]}")
    logger.info(f"  mean={d_arr.mean():.3f}  std={d_arr.std():.4f}  "
                f"{'PASS ✓' if varied else 'FAIL — collapsed'}")
    return varied


def main():
    parser = argparse.ArgumentParser(description="Rev-GNN-Transformer training")
    parser.add_argument("--config", default="configs/experiments/rev_gnn_transformer.yaml")
    args = parser.parse_args()

    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name=cfg.project.experiment_name)
    ensure_dir("results/checkpoints")
    logger.info(f"Window={cfg.transformer.attention_window} steps  "
                f"(O(T²) capped at T={cfg.transformer.attention_window})")

    # ── Build graphs ──────────────────────────────────────────────────────────
    train_graphs = [
        generate_forest_fire(200 + i * 60, cfg.graph.p, cfg.graph.pb,
                             seed=cfg.project.seed + i)
        for i in range(cfg.training.n_train_graphs)
    ]
    test_graph   = generate_forest_fire(
        cfg.evaluation.test_n_nodes, cfg.graph.p, cfg.graph.pb,
        seed=cfg.project.seed + 9999,
    )
    sanity_graph = generate_forest_fire(50, cfg.graph.p, cfg.graph.pb,
                                         seed=cfg.project.seed + 1)
    logger.info(f"Train: {[g.number_of_nodes() for g in train_graphs]}, "
                f"test n={test_graph.number_of_nodes()}")

    # ── Build policy ──────────────────────────────────────────────────────────
    enc = GraphSAGEEncoder(
        int(cfg.features.dim), int(cfg.encoder.hidden_dim),
        int(cfg.encoder.n_layers), float(cfg.encoder.dropout),
    )
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    policy = TransformerJointPolicy(
        enc, tfm,
        gnn_dim=int(cfg.encoder.hidden_dim),
        context_dim=tfm.context_dim,
    ).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | "
                f"device={device} | window={tfm.window}")

    # ── Sanity gate: fresh-weights discounts must be varied ───────────────────
    logger.info("\n=== Sanity Gate (fresh weights, n=50) ===")
    _sanity_gate(policy, sanity_graph, device, logger)

    # ── Phase 1: Transformer-aware CE + pricing MSE imitation ────────────────
    logger.info("\n=== Phase 1: Transformer Imitation (CE + pricing MSE) ===")
    n_im = int(cfg.training.imitation_epochs)   # 50 (overrideable via config)
    pw   = float(getattr(cfg.training, "pricing_loss_weight", 0.3))
    im_opt = torch.optim.Adam(policy.parameters(), lr=float(cfg.training.imitation_lr),
                              weight_decay=float(cfg.training.weight_decay))
    statics = {id(g): compute_static_features(g) for g in train_graphs}
    caches  = {id(g): build_graph_feature_cache(g, statics[id(g)]) for g in train_graphs}

    logger.info("Pre-generating expert trajectories (n_mc=200)...")
    trajs = {id(g): greedy_discount_trajectory(g, cfg) for g in train_graphs}
    logger.info(f"Trajectories ready for {len(trajs)} graphs")

    policy.train()
    for epoch in range(n_im):
        ep_losses = []
        for g in train_graphs:
            cache = caches[id(g)]; traj = trajs[id(g)]
            n_g = g.number_of_nodes(); nodes = list(g.nodes())
            env = _make_env(g, cfg); env.reset()
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
                loss = F.cross_entropy(ms.unsqueeze(0),
                                       torch.tensor([nidx], device=device))
                if pw > 0:
                    combined = torch.cat([h[nidx], ctx])
                    if hasattr(policy, "get_discount_distribution"):
                        pr = policy.get_discount_distribution(combined).mean
                    else:
                        pr = policy.pricing_head(combined.unsqueeze(0)).squeeze()
                    loss = loss + pw * F.mse_loss(
                        pr, torch.tensor(ed, dtype=torch.float32, device=device))
                im_opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
                im_opt.step()
                step_losses.append(loss.item())
                node = nodes[nidx]
                if acc:
                    S = frozenset(S | {node}); env.S.add(node)
                    env._influence_cache = {}
                off = frozenset(off | {node}); env.offered.add(node); env.t += 1
                policy.update_sequence_state(ed, acc, (1 - ed) * 0.5 * float(acc))
            if step_losses:
                ep_losses.append(sum(step_losses) / len(step_losses))
        mean_loss = sum(ep_losses) / max(len(ep_losses), 1)
        if epoch % 10 == 0:
            rev = _eval_transformer(policy, test_graph, cfg, device)
            logger.info(f"  IM ep={epoch:4d}  loss={mean_loss:.4f}  rev={rev:.2f}")
            logger.log({"im/epoch": epoch, "im/loss": mean_loss, "im/revenue": rev})

    # ── Post-Phase-1 sanity gate ──────────────────────────────────────────────
    logger.info("\n=== Post-P1 Sanity Gate ===")
    _sanity_gate(policy, sanity_graph, device, logger)
    rev_p1 = _eval_transformer(policy, test_graph, cfg, device)
    logger.info(f"Post-P1 revenue (n=1000): {rev_p1:.2f}  (baseline: {GREEDY_BASELINE})")
    if rev_p1 < 300:
        logger.info("WARNING: Post-P1 revenue below 300 — check training stability")
    logger.log({"phase": 1, "rev_after_p1": rev_p1})

    # ── Move to CPU (MPS corrupts autograd in no_grad context, same as LSTM) ──
    device_cpu = torch.device("cpu")
    policy     = policy.to(device_cpu)

    # ── Phase 1.5: SKIPPED ────────────────────────────────────────────────────
    logger.info("Skipping Phase 1.5 (same gradient-bug reason as LSTM)")
    logger.info("Going directly to Phase 2: Joint REINFORCE (all params, CPU)")

    # ── Phase 2: Joint REINFORCE (CPU, all params) ────────────────────────────
    logger.info("\n=== Phase 2: Joint REINFORCE Fine-Tuning (CPU) ===")
    for param in policy.parameters():
        param.requires_grad = True

    p2 = REINFORCETrainer(policy, cfg, logger, device_cpu)
    p2.optimizer = torch.optim.Adam(
        policy.parameters(),
        lr=float(cfg.training.reinforce_lr),
        weight_decay=float(cfg.training.weight_decay),
    )

    best_rev   = rev_p1
    best_state = copy.deepcopy(policy.state_dict())

    for ep in range(int(cfg.training.reinforce_epochs_phase2)):
        graph    = train_graphs[ep % len(train_graphs)]
        rollout  = p2.collect_rollout(graph)
        loss_val = p2.update(graph, rollout)
        if (ep + 1) % 10 == 0:
            rev    = _eval_transformer(policy, test_graph, cfg, device_cpu)
            marker = "  ← best" if rev > best_rev else ""
            logger.info(f"  ep={ep+1:3d}  rev={rev:.2f}  loss={loss_val:.5f}{marker}")
            logger.log({"rl/epoch": ep+1, "rl/revenue": rev, "rl/loss": loss_val})
            if rev > best_rev:
                best_rev   = rev
                best_state = copy.deepcopy(policy.state_dict())

    policy.load_state_dict(best_state)
    torch.save(best_state, "results/checkpoints/rev_gnn_transformer.pt")
    logger.info(f"Checkpoint → results/checkpoints/rev_gnn_transformer.pt")
    logger.info(f"Best revenue: {best_rev:.2f}  vs baseline: {GREEDY_BASELINE}  "
                f"({'BEATS' if best_rev > GREEDY_BASELINE else 'below'} baseline)")
    logger.log({"final/best_revenue": best_rev,
                "final/greedy_baseline": GREEDY_BASELINE,
                "final/beats_baseline": best_rev > GREEDY_BASELINE})
    logger.finish()


if __name__ == "__main__":
    main()
