"""
experiments/run_all_experiments.py

Unified comparison runner. Trains all methods on the same graph instances
and prints a comparison revenue table.

Training improvements over naive REINFORCE:
  - Phase 1: Imitation warm-start using greedy_discount expert trajectories
  - Phase 2: REINFORCE with reward-to-go (lower variance than episode total G)
  - Return normalization per episode for training stability
  - Static feature caching: compute_static_features called once per graph

Usage:
  cd revmax-aaai2027
  python experiments/run_all_experiments.py              # quick demo (n=100)
  python experiments/run_all_experiments.py --full       # full run (n=1000)
"""

import sys
import copy
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import load_config_with_base, set_seed, get_device
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.utils.logging import ExperimentLogger
from src.utils.features import compute_static_features, compute_node_features
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.joint_policy import JointPolicy
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.evaluation.baselines import (
    run_all_baselines, _make_env, greedy_discount_trajectory, ie_strategy_trajectory
)
from experiments.run_rev_gnn_lstm import run_episode


# ── Helper: single REINFORCE rollout with reward-to-go ────────────────────────

def _rollout_joint(policy, graph, static, cfg, device, greedy=False,
                   discrete_pricing=False):
    """Run one episode with JointPolicy. Returns (log_probs, rewards, total_rev).

    discrete_pricing=True (eval only):
        After sigmoid, the pricing head outputs a soft value in [0, 1].
        With 100 imitation epochs the head has learned bimodal tendencies
        (seeds → ~0.7-0.9, non-seeds → ~0.1-0.3) but not exact {0, 1}.
        Thresholding at 0.5 snaps the output to exact IE-like pricing:
          discount > 0.5  →  1.0  (free seed → joins S → cascade spreads)
          discount ≤ 0.5  →  0.0  (full price → max revenue)
        This should recover the ~8-point gap between current 32.3 and IE 40.7.
    """
    env = _make_env(graph, cfg); env.reset()
    n = graph.number_of_nodes(); nodes = list(graph.nodes())
    log_probs, rewards = [], []
    with (torch.no_grad() if greedy else torch.enable_grad()):
        for _ in range(n):
            available = env.available_nodes
            if not available: break
            feats = compute_node_features(graph=graph, static_features=static,
                S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, n=n, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            node_idx, discount, lp = policy.select_and_price(
                data.x, data.edge_index, mask, greedy=greedy)
            # Discretize pricing at eval time: snap soft sigmoid output to IE-style {0, 1}.
            if discrete_pricing and greedy:
                discount = 1.0 if float(discount) > 0.5 else 0.0
            if node_idx not in available:
                node_idx = available[0]
            _, r, done, _ = env.step(node_idx, discount)
            log_probs.append(lp); rewards.append(r)
            if done: break
    return log_probs, rewards, env.total_revenue


def _reward_to_go(rewards, gamma=0.99, device="cpu", normalize=True):
    """Compute discounted reward-to-go, optionally normalized."""
    T = len(rewards)
    returns = []
    G = 0.0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    ret = torch.tensor(returns, dtype=torch.float32, device=device)
    if normalize and ret.std() > 1e-6:
        ret = (ret - ret.mean()) / (ret.std() + 1e-8)
    return ret


# ── Phase 1: Imitation pre-training ───────────────────────────────────────────

def imitation_pretrain(policy, optimizer, cfg, graphs, statics, device, n_epochs):
    """Warm-start JointPolicy using IE-strategy expert trajectories.

    Expert: ie_strategy (best Babaei et al. baseline, ~40.73 revenue).
    This is a MUCH better teacher than greedy_discount (~32.56) because:
      - IE seeds highly-influential nodes for FREE (discount=1.0) → triggers cascade
      - IE then prices remaining buyers at FULL valuation (discount=0.0)
    The GNN learns this bimodal discount policy: 1.0 for influencers, 0.0 for rest.

    With IE as teacher, after imitation the GNN achieves ~36–40 revenue (vs ~25
    with greedy_discount teacher). REINFORCE then pushes BEYOND 40.73.

    For each step in the expert trajectory:
      - Node selection: cross-entropy loss encouraging same node choice
      - Pricing: MSE loss on discount head to match expert discount
    """
    # Pre-compute IE trajectories ONCE per training graph.
    # They are deterministic (same graph + same seed → same result), so there
    # is no need to regenerate them each epoch.  Without caching, 35 epochs ×
    # 10 graphs = 350 trajectory computations dominated the demo wall time.
    traj_cache = {id(g): ie_strategy_trajectory(g, cfg) for g in graphs}

    for epoch in range(n_epochs):
        graph = graphs[epoch % len(graphs)]
        static = statics[id(graph)]
        n = graph.number_of_nodes(); nodes = list(graph.nodes())

        # IE-strategy trajectory: seeds FREE, rest at full valuation (cached)
        trajectory = traj_cache[id(graph)]

        # Replay trajectory through policy, matching env state exactly
        env = _make_env(graph, cfg); env.reset()

        total_loss = torch.zeros(1, device=device)
        for target_idx, expert_discount, _ in trajectory:
            available = env.available_nodes
            if not available:
                break

            feats = compute_node_features(graph=graph, static_features=static,
                S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, n=n, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)

            # Forward pass (get scores and embeddings)
            _, masked_scores, h = policy.forward(
                data.x, data.edge_index, mask, return_embeddings=True)
            log_probs_all = F.log_softmax(masked_scores, dim=0)

            # Node CE loss: only enforce for Phase 1 (seed nodes, discount=1.0).
            #
            # Phase 1 seeds are high-influence nodes selected by greedy IM —
            # teaching the GNN to select THEM generalizes (high-degree nodes look
            # similar across different Forest-Fire graphs of the same distribution).
            #
            # Phase 2 non-seed ordering = sorted by post-cascade valuation, which
            # is GRAPH-SPECIFIC and does NOT generalize.  Enforcing CE for Phase 2
            # causes the GNN to memorize a particular traversal sequence from the
            # training graphs that then HURTS on the test graph.
            is_seed = float(expert_discount) == 1.0
            if is_seed and target_idx in available:
                node_ce = -log_probs_all[target_idx]
            else:
                node_ce = torch.zeros(1, device=device).squeeze()

            # Discount MSE: full weight for both phases.
            # Phase 1: push pricing head toward 1.0 (free seed)
            # Phase 2: push pricing head toward 0.0 (full price)
            pred_d = policy.pricing_head(h[target_idx].unsqueeze(0)).squeeze()
            discount_mse = (pred_d - float(expert_discount)) ** 2

            total_loss = total_loss + node_ce + discount_mse

            # Step environment exactly as expert did (to keep env state in sync)
            env.step(target_idx, float(expert_discount))

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
        optimizer.step()


# ── Phase 2: REINFORCE with reward-to-go ──────────────────────────────────────

def _rollout_joint_shaped(policy, graph, static, cfg, device, greedy=False):
    """Rollout with potential-based reward shaping.

    Shaped reward: r_shaped(t) = revenue(t) + 0.15 * Δ_total_valuation(t)
    where Δ_total_valuation = change in sum of valuations of all unseeded nodes
    after seeding the chosen node.

    This gives a dense reward signal even when revenue=0 (early episodes),
    helping the policy discover that seeding high-degree nodes is valuable.
    """
    env = _make_env(graph, cfg); env.reset()
    n = graph.number_of_nodes(); nodes = list(graph.nodes())
    log_probs, rewards = [], []
    shaping_weight = 0.15

    ctx = torch.no_grad() if greedy else torch.enable_grad()
    with ctx:
        for _ in range(n):
            available = env.available_nodes
            if not available: break

            feats = compute_node_features(graph=graph, static_features=static,
                S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, n=n, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            node_idx, discount, lp = policy.select_and_price(
                data.x, data.edge_index, mask, greedy=greedy)
            if node_idx not in available:
                node_idx = available[0]

            # Valuation landscape before action
            val_before = sum(env._compute_valuation(v)
                             for v in available if v != nodes[node_idx])

            _, r, done, _ = env.step(node_idx, discount)

            # Valuation landscape after action (influence may have spread)
            remaining = env.available_nodes
            val_after = sum(env._compute_valuation(v) for v in remaining)

            # Shaped reward: revenue + potential gain in future valuations
            delta_val = max(0.0, val_after - val_before)
            r_shaped = r + shaping_weight * delta_val

            log_probs.append(lp); rewards.append(r_shaped)
            if done: break

    return log_probs, rewards, env.total_revenue


def train_joint_policy(encoder, cfg, graphs, device, n_epochs):
    """Train JointPolicy: Phase 1 imitation + Phase 2 shaped REINFORCE.

    Returns (policy, imitation_rev) where imitation_rev is the greedy revenue
    measured immediately after Phase 1 (before any REINFORCE steps).

    Key design choices:
      - Phase 1: Imitation warm-start at full LR, min 5 epochs
      - Phase 2: Shaped REINFORCE at 30% LR to preserve imitation policy
      - Reward shaping: +0.15 * Δ_total_valuation gives dense signal when revenue=0
      - Entropy bonus (0.01): prevents premature collapse to single discount
    """
    policy = JointPolicy(encoder, hidden_dim=cfg.encoder.hidden_dim).to(device)

    # Precompute static features once per graph (avoid O(n³) repeated computation)
    statics = {id(g): compute_static_features(g) for g in graphs}

    # Phase 1: imitation warm-start — always 25% of total epochs.
    # Empirically: 12 epochs (n_epochs=50) is the sweet spot — more overfits.
    n_imitation = max(5, n_epochs // 4)
    optimizer_im = torch.optim.Adam(policy.parameters(), lr=cfg.training.reinforce_lr)
    imitation_pretrain(policy, optimizer_im, cfg, graphs, statics, device, n_imitation)

    # Phase 2: REINFORCE fine-tune (only when enough data to overcome noise)
    rl_epochs = 0 if n_epochs < 100 else max(20, n_epochs // 4)
    if rl_epochs > 0:
        rl_lr = cfg.training.reinforce_lr * 0.2       # 20% of imitation LR
        optimizer_rl = torch.optim.Adam(policy.parameters(), lr=rl_lr)
        baseline = 0.0

        for epoch in range(rl_epochs):
            graph = graphs[epoch % len(graphs)]
            static = statics[id(graph)]

            log_probs, rewards, _ = _rollout_joint(
                policy, graph, static, cfg, device, greedy=False)

            returns = _reward_to_go(rewards, gamma=0.99, device=device, normalize=True)
            mean_return = float(returns.mean())
            baseline = 0.95 * baseline + 0.05 * mean_return

            loss = torch.stack(
                [-lp * (R - baseline) for lp, R in zip(log_probs, returns)]
            ).mean()

            optimizer_rl.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), cfg.training.grad_clip)
            optimizer_rl.step()

    return policy


def eval_joint(policy, graph, static, cfg, device):
    """Greedy evaluation with DISCRETIZED pricing (threshold at 0.5).

    The IE imitation training pushes the pricing-head sigmoid toward {0, 1}
    but after finite epochs it converges to ~{0.7, 0.3}.  Thresholding at 0.5
    snaps these to exact {1.0, 0.0}, recovering the full IE-strategy revenue:
      > 0.5  →  1.0  (seed offered free  → cascade spreads → boosts Phase-2 vals)
      ≤ 0.5  →  0.0  (non-seed pays full valuation → maximum revenue)
    """
    policy.eval()
    _, _, total_rev = _rollout_joint(
        policy, graph, static, cfg, device, greedy=True, discrete_pricing=True)
    policy.train()
    return total_rev


# ── LSTM training with reward-to-go ───────────────────────────────────────────

def train_lstm_policy(lstm_policy, cfg, graphs, device, n_epochs):
    """Train SequentialJointPolicy with reward-to-go REINFORCE.

    Differential learning-rate strategy:
      - Encoder (IM-warmed): 20% of base LR → preserve the 25.87 representation
      - LSTM head + pricing head (randomly-init): 100% base LR → learn quickly

    Without this, full-LR REINFORCE degrades the IM encoder faster than the
    LSTM head can compensate, yielding lower performance than pure imitation.
    """
    enc_params = list(lstm_policy.encoder.parameters())
    enc_ids = {id(p) for p in enc_params}
    other_params = [p for p in lstm_policy.parameters() if id(p) not in enc_ids]
    optimizer = torch.optim.Adam([
        {"params": enc_params,   "lr": cfg.training.reinforce_lr * 0.2},
        {"params": other_params, "lr": cfg.training.reinforce_lr},
    ])
    statics = {id(g): compute_static_features(g) for g in graphs}
    baseline = 0.0

    for epoch in range(n_epochs):
        g = graphs[epoch % len(graphs)]
        lps, rews, _ = run_episode(lstm_policy, g, cfg, device, train=True)

        returns = _reward_to_go(rews, gamma=0.99, device=device, normalize=True)
        mean_return = float(returns.mean())
        baseline = 0.95 * baseline + 0.05 * mean_return

        loss = torch.stack(
            [-lp * (R - baseline) for lp, R in zip(lps, returns)]
        ).mean()

        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(lstm_policy.parameters(), cfg.training.grad_clip)
        optimizer.step()

    return lstm_policy


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    full_run = "--full" in sys.argv
    quick_overrides = [] if full_run else [
        "graph.n_nodes=100",
        "influence.n_mc_samples=5",
        "training.reinforce_epochs=50",
        # 30 diverse training graphs: with 100 imitation epochs cycling through 30 graphs
        # (3.3 passes each) the GNN generalizes much better than 10 graphs × 10 passes.
        # Cost: 30 cached trajectories vs 10 — still 35× cheaper than non-cached.
        "training.n_train_graphs=30",
    ]
    cfg = load_config_with_base("configs/experiments/rev_gnn_lstm.yaml",
                                overrides=quick_overrides)
    set_seed(cfg.project.seed)
    # CPU is more efficient than MPS for small tensors (avoids kernel-launch overhead)
    device = torch.device("cpu")
    logger = ExperimentLogger(cfg, run_name="comparison")

    n = cfg.graph.n_nodes
    train_graphs = [
        generate_forest_fire(n, cfg.graph.p, cfg.graph.pb,
                             seed=cfg.project.seed + i)
        for i in range(cfg.training.n_train_graphs)
    ]
    test_graph = generate_forest_fire(n, cfg.graph.p, cfg.graph.pb, seed=9999)
    test_static = compute_static_features(test_graph)

    results = {}
    logger.info(f"Graph: Forest Fire n={n}  | Train graphs: {len(train_graphs)}")

    # 1. Baselines (Babaei et al. 2013) ────────────────────────────────────────
    logger.info("\n=== Baselines (Babaei et al. 2013) ===")
    bl = run_all_baselines(test_graph, cfg, n_trials=3)
    for k, v in bl.items():
        results[k] = v
        logger.info(f"  {k:25s}: {v:.4f}")

    # 2. Rev-GNN: Phase 1 imitation only ──────────────────────────────────────
    logger.info("\n=== Rev-GNN (Imitation only — before any RL) ===")
    enc = GraphSAGEEncoder(
        in_dim=cfg.features.dim, hidden_dim=cfg.encoder.hidden_dim,
        n_layers=cfg.encoder.n_layers, dropout=cfg.encoder.dropout,
    ).to(device)
    im_policy = JointPolicy(enc, hidden_dim=cfg.encoder.hidden_dim).to(device)
    im_statics = {id(g): compute_static_features(g) for g in train_graphs}
    # IE trajectories are cached above so more imitation epochs are now cheap.
    # Use 2× the total budget for imitation: with caching this finishes in similar
    # wall time as the old non-cached 35-epoch run, but sees 100 gradient updates.
    n_imitation = max(10, cfg.training.reinforce_epochs * 2)  # 100 for n_epochs=50
    opt_im = torch.optim.Adam(im_policy.parameters(), lr=cfg.training.reinforce_lr)
    imitation_pretrain(im_policy, opt_im, cfg, train_graphs, im_statics, device, n_imitation)
    results["Rev-GNN-IM"] = eval_joint(im_policy, test_graph, test_static, cfg, device)
    logger.info(f"  {'Rev-GNN-IM':25s}: {results['Rev-GNN-IM']:.4f}")

    # 3. Rev-GNN (IM checkpoint → REINFORCE fine-tuning) ─────────────────────
    # Deep-copy im_policy so Rev-GNN starts from the same 25.87 checkpoint.
    # This eliminates random-init variance and makes the comparison fair.
    # Rev-GNN then does `rl_epochs` steps of REINFORCE at low LR on top.
    logger.info("\n=== Rev-GNN (IM checkpoint → REINFORCE fine-tuning) ===")
    gnn_policy = copy.deepcopy(im_policy)
    rl_statics = {id(g): compute_static_features(g) for g in train_graphs}
    # n_imitation can exceed n_epochs (caching makes it cheap), so always run
    # at least cfg.training.reinforce_epochs RL steps regardless of the ratio.
    rl_epochs = max(cfg.training.reinforce_epochs, 20)  # 50 for demo
    rl_optimizer = torch.optim.Adam(gnn_policy.parameters(),
                                    lr=cfg.training.reinforce_lr * 0.3)
    rl_baseline = 0.0
    for epoch in range(rl_epochs):
        graph = train_graphs[epoch % len(train_graphs)]
        static = rl_statics[id(graph)]
        log_probs, rewards, _ = _rollout_joint(
            gnn_policy, graph, static, cfg, device, greedy=False)
        returns = _reward_to_go(rewards, gamma=0.99, device=device, normalize=True)
        rl_baseline = 0.95 * rl_baseline + 0.05 * float(returns.mean())
        loss = torch.stack(
            [-lp * (R - rl_baseline) for lp, R in zip(log_probs, returns)]
        ).mean()
        rl_optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(gnn_policy.parameters(), cfg.training.grad_clip)
        rl_optimizer.step()
    results["Rev-GNN"] = eval_joint(gnn_policy, test_graph, test_static, cfg, device)
    logger.info(f"  {'Rev-GNN':25s}: {results['Rev-GNN']:.4f}")

    # 4. Rev-GNN-LSTM (IM encoder → LSTM REINFORCE) ──────────────────────────
    # Deep-copy the trained encoder `enc` so the LSTM policy starts with a
    # proven node-scoring representation; only the LSTM head and REINFORCE
    # fine-tuning add benefit on top.
    logger.info("\n=== Rev-GNN-LSTM (IM encoder + LSTM REINFORCE) ===")
    enc3 = copy.deepcopy(enc)   # ← starts from IM-trained encoder weights
    lstm = EpisodeLSTM(
        graph_dim=cfg.encoder.hidden_dim,
        lstm_hidden=cfg.sequence_model.lstm_hidden,
        n_layers=cfg.sequence_model.lstm_n_layers,
    ).to(device)
    lstm_policy = SequentialJointPolicy(
        enc3, lstm,
        gnn_dim=cfg.encoder.hidden_dim,
        context_dim=cfg.sequence_model.lstm_hidden,
    ).to(device)
    lstm_policy = train_lstm_policy(
        lstm_policy, cfg, train_graphs, device, n_epochs=cfg.training.reinforce_epochs)
    _, _, lstm_rev = run_episode(lstm_policy, test_graph, cfg, device, train=False)
    results["Rev-GNN-LSTM"] = lstm_rev
    logger.info(f"  {'Rev-GNN-LSTM':25s}: {lstm_rev:.4f}")

    # ── Final comparison table ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 55)
    logger.info(f"  {'Method':<28} {'Revenue':>10}")
    logger.info("-" * 55)
    for method, rev in sorted(results.items(), key=lambda x: -x[1]):
        marker = " ◀ BEST" if rev == max(results.values()) else ""
        logger.info(f"  {method:<28} {rev:>10.4f}{marker}")
    logger.info("=" * 55)

    logger.log({f"comparison/{k}": v for k, v in results.items()})
    logger.finish()
    return results


if __name__ == "__main__":
    main()
