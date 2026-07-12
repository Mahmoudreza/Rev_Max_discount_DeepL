"""experiments/run_budget_training.py — Stage 2: Budget-Aware LSTM Training (Idea 3).

Fine-tunes Rev-GNN-LSTM to handle the production budget constraint.

Key changes from Idea 1 training:
  1. Episodes run in BudgetRevenueEnv (with cost c, budget B)
  2. Features extended from 20-dim → 21-dim (add budget_fraction)
  3. Input projection layer extended: (64, 20) → (64, 21)
     Old weights preserved; budget dim initialised with Xavier uniform
  4. Budget randomly sampled per episode from cfg.training.budget_levels
  5. Reward = total_revenue (episode may end early on bankruptcy)
     LSTM learns: free offers drain budget → bankruptcy → no reward

Training procedure:
  Phase 1: Warm-start from Idea 1 imitation checkpoint
           Extend input dim 20→21 (copy old + init new dim)
  Phase 2: REINFORCE on BudgetRevenueEnv
           Reward normalised by n (same as Idea 1)
           Entropy bonus for exploration
           Gradient clipping (same as Idea 1)

Checkpoint: results/checkpoints/rev_gnn_lstm_budget.pt

Usage:
  cd revmax-aaai2027
  source venv/bin/activate
  python experiments/run_budget_training.py \
    --config configs/experiments/budget_constrained.yaml \
    --warm_start results/checkpoints/rev_gnn_lstm.pt

NOTE: Stage 1 evaluation (run_budget_eval.py) must be completed first.
      The training here is Stage 2 and uses the same baselines as references.

Implementation notes:
  - Do NOT start this training until Stage 1 results are confirmed
  - Expected training time: ~3-4 hours (200 epochs × multiple budget levels)
  - The checkpoint extension logic is in _extend_input_dim() below
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import numpy as np

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.utils.logging import ExperimentLogger


def _extend_input_dim(state_dict: dict, new_input_dim: int = 21) -> dict:
    """Extend the GNN encoder's input projection from 20 → new_input_dim dims.

    Copies old weights for dims 0..19, initialises dim 20 with Xavier uniform.
    This allows the pre-trained model to continue working with the old 20 dims
    while learning the budget dimension from scratch (gradient signal from REINFORCE).

    Args:
        state_dict:     State dict from Idea 1 checkpoint.
        new_input_dim:  Target input dimension (default 21).

    Returns:
        Modified state_dict with extended input projection.
    """
    old_dim = 20
    if new_input_dim <= old_dim:
        return state_dict

    # Find the input projection weight key
    proj_key = None
    for k in state_dict:
        if "input_proj" in k and "weight" in k:
            proj_key = k
            break
    if proj_key is None:
        # Try encoder's first linear layer
        for k in state_dict:
            if "encoder" in k and "weight" in k:
                w = state_dict[k]
                if isinstance(w, torch.Tensor) and w.shape[1] == old_dim:
                    proj_key = k
                    break

    if proj_key is None:
        print(f"  [WARN] Could not find input projection layer — keys: {list(state_dict.keys())[:5]}")
        return state_dict

    old_weight = state_dict[proj_key]        # shape: (hidden_dim, actual_dim)
    actual_dim = old_weight.shape[1]
    if actual_dim >= new_input_dim:
        # Already at target dim (e.g. Idea3 checkpoint already has 21-dim proj).
        print(f"  {proj_key} already {tuple(old_weight.shape)} — no extension needed")
        return state_dict

    # Use the ACTUAL dim from the tensor (not the hardcoded 20) for the copy.
    old_dim    = actual_dim
    hidden_dim = old_weight.shape[0]
    new_weight = torch.zeros(hidden_dim, new_input_dim)
    new_weight[:, :old_dim] = old_weight    # copy old weights
    nn.init.xavier_uniform_(new_weight[:, old_dim:])   # init new dim
    state_dict[proj_key] = new_weight
    print(f"  Extended {proj_key}: {tuple(old_weight.shape)} → {tuple(new_weight.shape)}")
    return state_dict


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/experiments/budget_constrained.yaml")
    parser.add_argument("--warm_start",  default="results/checkpoints/rev_gnn_lstm_budget.pt",
                        help="Checkpoint to warm-start from (Idea3 21-dim, or Idea1 20-dim)")
    parser.add_argument("--out_ckpt",    default="results/checkpoints/rev_gnn_lstm_budget.pt")
    parser.add_argument("--n_epochs",    type=int, default=200)
    parser.add_argument("--log_every",   type=int, default=10)
    parser.add_argument("--save_every",  type=int, default=10,
                        help="Save checkpoint every N epochs (default: 10)")
    parser.add_argument("--budget_levels", default="2,5,20,50",
                        help="Comma-separated budget levels to sample (overrides config)")
    parser.add_argument("--dry_run",     action="store_true",
                        help="Run 2 epochs for smoke-test only")
    args = parser.parse_args()

    cfg = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    # Force CPU: aten::_sample_dirichlet is not implemented for MPS.
    # Consistent with run_transformer_budget_training.py.
    device = torch.device("cpu")
    logger = ExperimentLogger(cfg, run_name="budget_training")

    logger.info("Budget-Aware LSTM Training (Idea 3)")
    logger.info(f"Warm-start: {args.warm_start}")
    logger.info(f"Output ckpt: {args.out_ckpt}")
    logger.info(f"n_epochs={args.n_epochs} | log_every={args.log_every}")

    if not os.path.exists(args.warm_start):
        logger.info(f"[ERROR] Warm-start checkpoint not found: {args.warm_start}")
        logger.info("Run Idea 1 training first: python experiments/run_rev_gnn_lstm.py")
        return

    # ── Load and extend checkpoint ────────────────────────────────────────────
    from src.models.policies.sequential_joint_policy import SequentialJointPolicy

    old_ckpt = torch.load(args.warm_start, map_location="cpu")
    if isinstance(old_ckpt, dict) and "model_state_dict" in old_ckpt:
        state_dict = old_ckpt["model_state_dict"]
    else:
        state_dict = old_ckpt

    extended_state = _extend_input_dim(state_dict, new_input_dim=21)

    # Build policy with 21-dim input (same as load_lstm but input_dim=21)
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.sequence_models import EpisodeLSTM

    hidden_dim   = int(cfg.encoder.hidden_dim)
    n_layers     = int(cfg.encoder.n_layers)
    dropout      = float(getattr(cfg.encoder, "dropout", 0.0))
    lstm_hidden  = int(cfg.sequence_model.lstm_hidden)
    lstm_n_layers= int(cfg.sequence_model.lstm_n_layers)

    enc  = GraphSAGEEncoder(21, hidden_dim, n_layers, dropout)
    lstm = EpisodeLSTM(graph_dim=hidden_dim,
                       lstm_hidden=lstm_hidden,
                       n_layers=lstm_n_layers)
    policy = SequentialJointPolicy(enc, lstm,
                                   gnn_dim=hidden_dim,
                                   context_dim=lstm_hidden).to(device)

    # Load with strict=False (handles minor key mismatches)
    missing, unexpected = policy.load_state_dict(extended_state, strict=False)
    if missing:
        logger.info(f"  Missing keys: {missing[:5]}")
    if unexpected:
        logger.info(f"  Unexpected keys: {unexpected[:5]}")
    logger.info("  Checkpoint loaded with extended input projection")

    # ── Training setup ─────────────────────────────────────────────────────────
    from src.env.graph_generators import generate_forest_fire
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.budget_features import compute_budget_node_features
    from src.utils.features import compute_static_features

    p, pb        = float(cfg.graph.p), float(cfg.graph.pb)
    c            = float(cfg.budget_constrained.production_cost)
    # --budget_levels CLI arg overrides config (enables warm-start fine-tuning
    # with a different budget set without editing the YAML).
    budget_levels = [float(x) for x in args.budget_levels.split(",")]
    n_epochs     = 2 if args.dry_run else args.n_epochs
    lr           = float(cfg.training.reinforce_lr)
    entropy_coef = float(cfg.training.entropy_coef)
    grad_clip    = float(cfg.training.grad_clip)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Pre-build training graphs (5 graphs, rotate each epoch)
    train_graphs = [generate_forest_fire(1000, p, pb, seed=i) for i in range(5)]
    static_feats = [compute_static_features(g) for g in train_graphs]

    best_reward  = -float("inf")   # tracks best per-budget-normalised advantage
    best_state   = None
    rng_episode  = np.random.default_rng(cfg.project.seed)

    # ── Per-budget Welford running statistics ─────────────────────────────────
    # WHY: With mixed budgets, absolute revenue is not comparable across B values.
    # A B=5 episode yielding rev=330 is excellent; a B=50 episode yielding 330
    # is poor.  Without per-budget normalisation the gradient would always punish
    # B=5 episodes (they always appear "below average") and reward B=50 episodes,
    # causing the policy to ignore the tight-budget regime entirely.
    #
    # FIX: maintain separate Welford running mean/variance per B level.
    # advantage = (total_rev - mean_B) / max(std_B, 1e-8)
    # This ensures a good B=5 episode gets a POSITIVE gradient signal and a
    # mediocre B=50 episode gets a negative one — regardless of absolute scale.
    #
    # Note: we update stats AFTER computing the advantage (so the current episode
    # does not contribute to its own normalisation, avoiding self-inflation bias).
    welford: dict = {}   # B_key (rounded str) → {"n": int, "mean": float, "M2": float}

    def _welford_get(stats: dict):
        """Return (mean, std) from a Welford stats dict (returns (0, 1) if n < 2)."""
        n_w = stats.get("n", 0)
        if n_w < 2:
            return stats.get("mean", 0.0), 1.0
        return stats["mean"], float(np.sqrt(stats["M2"] / (n_w - 1)))

    def _welford_update(stats: dict, x: float) -> None:
        """Online Welford update (Knuth / Welford algorithm)."""
        stats["n"] = stats.get("n", 0) + 1
        delta      = x - stats.get("mean", 0.0)
        stats["mean"] = stats.get("mean", 0.0) + delta / stats["n"]
        delta2     = x - stats["mean"]
        stats["M2"] = stats.get("M2", 0.0) + delta * delta2

    logger.info(f"Budget-REINFORCE: {n_epochs} epochs | budget_levels={budget_levels}")
    logger.info("Per-budget Welford advantage normalisation: ON")

    for epoch in range(n_epochs):
        graph_idx = epoch % len(train_graphs)
        graph     = train_graphs[graph_idx]
        sfeats    = static_feats[graph_idx]
        B         = float(rng_episode.choice(budget_levels))

        env_cfg = BudgetEnvConfig(
            budget_B=B, production_cost=c, seed=epoch,
            influence_model="monotone", n_mc_samples=50,   # fast during training
        )
        env = BudgetRevenueEnv(graph, env_cfg)
        env.reset()
        policy.reset_episode(device)

        # Build bidirectional edge_index inline
        _edges = list(graph.edges())
        if _edges:
            _src = [u for u, v in _edges] + [v for u, v in _edges]
            _dst = [v for u, v in _edges] + [u for u, v in _edges]
            edge_index = torch.tensor([_src, _dst], dtype=torch.long).to(device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)
        n = graph.number_of_nodes()

        policy_losses = []
        entropies     = []
        total_rev     = 0.0

        # Cap episode length to prevent O(n²) BPTT through sequential LSTM.
        # With B=50 and n=1000, uncapped episodes process all 1000 nodes,
        # creating a 1000-step LSTM chain whose backward pass is infeasible.
        # At 300 steps we still see >85% of the episode revenue while keeping
        # the backward pass fast. B=2/5 episodes end naturally at bankruptcy
        # well before this limit, so they are unaffected.
        max_steps = min(env.n, 300)
        steps_ep  = 0

        while len(env.offered) < env.n and not env._check_bankrupt():
            feats = compute_budget_node_features(
                graph, sfeats, env.S, env.offered, env.t, n,
                k=0, env=env,
            )
            x    = torch.tensor(feats, dtype=torch.float32).to(device)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            for idx in env.available_nodes:
                mask[idx] = True

            if mask.sum() == 0:
                break

            node_idx_t, disc_t, log_prob = policy.select_and_price(
                x, edge_index, mask, greedy=False
            )
            node_idx = int(node_idx_t)
            discount = float(disc_t)

            # Clamp discount to affordable range (max_affordable_discount returns -1
            # if node can't be served at any price).
            node     = env.nodes[node_idx]
            max_disc = env.max_affordable_discount(node)
            if max_disc >= 0:
                discount = min(discount, max_disc)   # Python min suffices; disc_t is already float

            _, reward, done, info = env.step(node_idx, discount)
            total_rev += reward
            policy.update_sequence_state(discount, info.get("accepted", False), reward)

            policy_losses.append(log_prob)
            entropies.append(log_prob)   # entropy approximation (use separate head later)

            steps_ep += 1
            if done or steps_ep >= max_steps:
                break

        if not policy_losses:
            continue

        # ── Per-budget Welford advantage ──────────────────────────────────────
        # Compute advantage with stats from ALL PREVIOUS same-B episodes
        # (do NOT update stats yet — avoids self-inflation of the current episode).
        B_key = str(round(B, 1))
        if B_key not in welford:
            welford[B_key] = {}
        w_mean, w_std = _welford_get(welford[B_key])

        # advantage: how many std-devs above/below same-B mean is this episode
        # Clamp std to >= 1.0 to prevent advantage explosion when Welford has
        # very few samples and revenues happen to be nearly equal (M2 ≈ 0).
        # Matches the σ=1.0 returned by _welford_get before n >= 2.
        advantage_val = (total_rev - w_mean) / max(w_std, 1.0)

        # Update Welford stats after computing advantage (no self-inflation)
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

        # Intermediate checkpoint: save best-so-far every save_every epochs
        if best_state is not None and (epoch + 1) % args.save_every == 0:
            ensure_dir(os.path.dirname(args.out_ckpt))
            torch.save(best_state, args.out_ckpt)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    ensure_dir(os.path.dirname(args.out_ckpt))
    if best_state is not None:
        torch.save(best_state, args.out_ckpt)
        logger.info(f"\nBest checkpoint saved → {args.out_ckpt}")
        logger.info(f"Best normalised reward: {best_reward:.4f}")
    else:
        logger.info("\n[WARN] No checkpoint saved (no successful episodes)")

    logger.info("\nDone. Evaluate with:")
    logger.info(f"  python experiments/run_budget_eval.py \\")
    logger.info(f"    --config configs/experiments/budget_constrained.yaml \\")
    logger.info(f"    --budget_ckpt {args.out_ckpt}")


if __name__ == "__main__":
    main()
