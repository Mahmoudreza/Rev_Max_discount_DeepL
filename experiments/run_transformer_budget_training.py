"""experiments/run_transformer_budget_training.py — Stage 2: Budget-Aware Transformer.

Fine-tunes Rev-GNN-Transformer (trained in Stage 1) on BudgetRevenueEnv.
Mirrors run_budget_training.py exactly — only policy type changes.

Key changes vs unconstrained Transformer:
  1. Episodes run in BudgetRevenueEnv (cost c, budget B)
  2. Features 20-dim → 21-dim (add budget_fraction via compute_budget_node_features)
  3. Encoder input projection extended: (64, 20) → (64, 21)
     Old weights preserved; budget dim initialised Xavier uniform
  4. Budget randomly sampled per episode from budget_levels
  5. Per-budget Welford advantage normalisation (same as LSTM budget)
  6. best-ckpt restore at end

Checkpoint: results/checkpoints/rev_gnn_transformer_budget.pt

Usage:
  cd revmax-aaai2027 && source venv/bin/activate
  python experiments/run_transformer_budget_training.py \\
    --config configs/experiments/budget_constrained.yaml \\
    --warm_start results/checkpoints/rev_gnn_transformer.pt
"""
import argparse, copy, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np

from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.utils.logging import ExperimentLogger


def _extend_input_dim(state_dict: dict, new_input_dim: int = 21) -> dict:
    """Extend encoder input projection from 20 → new_input_dim.

    Copies existing weights; new dim initialised with Xavier uniform.
    Returns modified state_dict (does NOT modify in-place).
    """
    import math
    state_dict = {k: v.clone() for k, v in state_dict.items()}

    proj_key = None
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] < new_input_dim:
            if "encoder" in k.lower() and "weight" in k:
                proj_key = k
                break

    if proj_key is None:
        print("  [WARN] Could not find encoder input projection — skipping extension")
        return state_dict

    old_weight = state_dict[proj_key]          # (hidden_dim, old_input_dim)
    old_dim    = old_weight.shape[1]
    hidden_dim = old_weight.shape[0]
    extra_dims = new_input_dim - old_dim

    if extra_dims <= 0:
        print(f"  {proj_key} already {tuple(old_weight.shape)} — no extension needed")
        return state_dict

    # Xavier uniform init for new dim(s)
    fan_in  = new_input_dim
    fan_out = hidden_dim
    bound   = math.sqrt(6.0 / (fan_in + fan_out))
    new_cols = torch.empty(hidden_dim, extra_dims).uniform_(-bound, bound)

    state_dict[proj_key] = torch.cat([old_weight, new_cols], dim=1)
    print(f"  Extended {proj_key}: {old_dim} → {new_input_dim} (kept {old_dim} dims, "
          f"Xavier-init {extra_dims} new)")
    return state_dict


def main():
    parser = argparse.ArgumentParser(description="Transformer budget fine-tuning")
    parser.add_argument("--config",      default="configs/experiments/budget_constrained.yaml",
                        help="Budget config (budget_constrained, training.reinforce_lr, etc.)")
    parser.add_argument("--config-tfm",  default="configs/experiments/rev_gnn_transformer_300ep.yaml",
                        help="Transformer config (transformer: section)")
    parser.add_argument("--warm_start",  default="results/checkpoints/rev_gnn_transformer.pt",
                        help="Unconstrained Transformer checkpoint to warm-start from")
    parser.add_argument("--out_ckpt",    default="results/checkpoints/rev_gnn_transformer_budget.pt")
    parser.add_argument("--n_epochs",    type=int, default=200)
    parser.add_argument("--budget_levels", default="2,5,20,50")
    parser.add_argument("--log_every",   type=int, default=10)
    parser.add_argument("--save_every",  type=int, default=25)
    parser.add_argument("--dry_run",     action="store_true",
                        help="Run 2 epochs only (smoke-test)")
    args = parser.parse_args()

    from omegaconf import OmegaConf
    cfg_budget = load_config_with_base(args.config)
    cfg_tfm    = load_config_with_base(args.config_tfm)
    # Merge: budget config as base, overlay transformer section from tfm config
    cfg = OmegaConf.merge(cfg_budget, OmegaConf.create({"transformer": OmegaConf.to_container(cfg_tfm.transformer)}))
    set_seed(cfg_budget.project.seed)
    device = torch.device("cpu")   # CPU for safety (same as LSTM)
    logger = ExperimentLogger(cfg_budget, run_name="rev_gnn_transformer_budget")
    ensure_dir("results/checkpoints")
    ensure_dir("results/logs")

    logger.info(f"Stage 2: Transformer Budget Fine-Tuning ({args.n_epochs} epochs)")
    logger.info(f"Warm-start: {args.warm_start}  →  {args.out_ckpt}")

    # ── Build 21-dim Transformer policy ──────────────────────────────────────
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy

    enc = GraphSAGEEncoder(
        21, int(cfg.encoder.hidden_dim),               # 21-dim input
        int(cfg.encoder.n_layers), float(cfg.encoder.dropout),
    )
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    policy = TransformerJointPolicy(
        enc, tfm,
        gnn_dim=int(cfg.encoder.hidden_dim),
        context_dim=tfm.context_dim,
    ).to(device)
    logger.info(f"Policy: {sum(p.numel() for p in policy.parameters()):,} params | "
                f"input_dim=21 | window={tfm.window}")

    # ── Load warm-start + extend 20→21 ────────────────────────────────────────
    logger.info(f"Loading warm-start checkpoint: {args.warm_start}")
    raw_state = torch.load(args.warm_start, map_location=device, weights_only=True)
    extended  = _extend_input_dim(raw_state, new_input_dim=21)
    missing, unexpected = policy.load_state_dict(extended, strict=False)
    if missing:
        logger.info(f"  Missing keys: {missing[:5]}")
    if unexpected:
        logger.info(f"  Unexpected keys: {unexpected[:5]}")
    logger.info("  Warm-start loaded with extended input projection")

    # ── Training setup ────────────────────────────────────────────────────────
    from src.env.graph_generators import generate_forest_fire
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.budget_features import compute_budget_node_features
    from src.utils.features import compute_static_features

    p, pb         = float(cfg.graph.p), float(cfg.graph.pb)
    c             = float(cfg.budget_constrained.production_cost)
    budget_levels = [float(x) for x in args.budget_levels.split(",")]
    n_epochs      = 2 if args.dry_run else args.n_epochs
    lr            = float(cfg.training.reinforce_lr)
    entropy_coef  = float(cfg.training.entropy_coef)
    grad_clip     = float(cfg.training.grad_clip)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    train_graphs = [generate_forest_fire(1000, p, pb, seed=i) for i in range(5)]
    static_feats = [compute_static_features(g) for g in train_graphs]

    best_reward  = -float("inf")
    best_state   = None
    rng_episode  = np.random.default_rng(cfg.project.seed)

    # ── Per-budget Welford running statistics ─────────────────────────────────
    welford: dict = {}

    def _welford_get(stats):
        n_w = stats.get("n", 0)
        if n_w < 2:
            return stats.get("mean", 0.0), 1.0
        return stats["mean"], float(np.sqrt(stats["M2"] / (n_w - 1)))

    def _welford_update(stats, x):
        stats["n"]    = stats.get("n", 0) + 1
        delta         = x - stats.get("mean", 0.0)
        stats["mean"] = stats.get("mean", 0.0) + delta / stats["n"]
        delta2        = x - stats["mean"]
        stats["M2"]   = stats.get("M2", 0.0) + delta * delta2

    logger.info(f"Budget-REINFORCE: {n_epochs} epochs | budget_levels={budget_levels}")
    logger.info("Per-budget Welford advantage normalisation: ON")

    policy.train()
    for epoch in range(n_epochs):
        graph_idx = epoch % len(train_graphs)
        graph     = train_graphs[graph_idx]
        sfeats    = static_feats[graph_idx]
        B         = float(rng_episode.choice(budget_levels))

        env_cfg = BudgetEnvConfig(
            budget_B=B, production_cost=c, seed=epoch,
            influence_model="monotone", n_mc_samples=50,
        )
        env = BudgetRevenueEnv(graph, env_cfg)
        env.reset()
        policy.reset_episode(device)

        # Build edge_index once per episode
        _edges = list(graph.edges())
        if _edges:
            _src = [u for u, v in _edges] + [v for u, v in _edges]
            _dst = [v for u, v in _edges] + [u for u, v in _edges]
            edge_index = torch.tensor([_src, _dst], dtype=torch.long, device=device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long, device=device)
        n = graph.number_of_nodes()

        policy_losses, entropies = [], []
        total_rev  = 0.0
        max_steps  = min(env.n, 300)          # cap for Transformer O(T²)
        steps_ep   = 0

        while len(env.offered) < env.n and not env._check_bankrupt():
            feats = compute_budget_node_features(
                graph, sfeats, env.S, env.offered, env.t, n, k=0, env=env)
            x    = torch.tensor(feats, dtype=torch.float32, device=device)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            for idx in env.available_nodes:
                mask[idx] = True

            if mask.sum() == 0:
                break

            node_idx_t, disc_t, log_prob = policy.select_and_price(
                x, edge_index, mask, greedy=False)
            node_idx = int(node_idx_t)
            discount = float(disc_t)

            node     = env.nodes[node_idx]
            max_disc = env.max_affordable_discount(node)
            if max_disc >= 0:
                discount = min(discount, max_disc)

            _, reward, done, info = env.step(node_idx, discount)
            total_rev += reward
            policy.update_sequence_state(discount, info.get("accepted", False), reward)

            policy_losses.append(log_prob)
            entropies.append(log_prob)

            steps_ep += 1
            if done or steps_ep >= max_steps:
                break

        if not policy_losses:
            continue

        # ── Per-budget Welford advantage ──────────────────────────────────────
        B_key = str(round(B, 1))
        if B_key not in welford:
            welford[B_key] = {}
        w_mean, w_std = _welford_get(welford[B_key])
        # Clamp std to >= 1.0 to avoid advantage explosion when Welford has few
        # samples and revenues happen to be nearly equal (M2 ≈ 0 → std ≈ 0).
        # 1.0 matches the σ returned by _welford_get before n >= 2, so
        # normalization is continuous across the warmup period.
        advantage_val = (total_rev - w_mean) / max(w_std, 1.0)
        _welford_update(welford[B_key], total_rev)

        advantages = [advantage_val] * len(policy_losses)

        # ── REINFORCE loss ────────────────────────────────────────────────────
        loss = torch.stack([
            -lp * torch.tensor(adv, dtype=torch.float32, device=device)
            for lp, adv in zip(policy_losses, advantages)
        ]).mean()
        loss -= entropy_coef * torch.stack(policy_losses).mean().abs()

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), grad_clip)
        optimizer.step()

        if advantage_val > best_reward:
            best_reward = advantage_val
            best_state  = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == n_epochs - 1:
            n_budget_eps = welford[B_key].get("n", 0)
            logger.info(
                f"Budget-REINFORCE ep {epoch:4d}/{n_epochs} | "
                f"B={B:4.0f} | rev={total_rev:6.1f} | "
                f"adv={advantage_val:+.3f} "
                f"(μ_B={w_mean:6.1f}, σ_B={w_std:5.1f}, n_B={n_budget_eps}) | "
                f"best_adv={best_reward:+.3f}"
            )
            logger.log({"budget/epoch": epoch, "budget/B": float(B),
                        "budget/rev": float(total_rev), "budget/adv": float(advantage_val)})

        # Save best checkpoint periodically
        if best_state is not None and (epoch + 1) % args.save_every == 0:
            torch.save(best_state, args.out_ckpt)

    # ── Final save ────────────────────────────────────────────────────────────
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in policy.state_dict().items()}
    torch.save(best_state, args.out_ckpt)
    logger.info(f"Checkpoint → {args.out_ckpt}")
    logger.info(f"Best advantage: {best_reward:.4f}")
    logger.finish()


if __name__ == "__main__":
    main()
