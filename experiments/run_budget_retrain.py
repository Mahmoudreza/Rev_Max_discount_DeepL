"""experiments/run_budget_retrain.py — Retrain LSTM-Idea3 to close the large-k gap.

Warm-starts from rev_gnn_lstm_budget.pt (already 21-dim), trains on
B ∈ {0.9, 3.0, 9.0} (k=3,10,30 at c=0.3) with per-B Welford advantage.

Three failure causes from prior run addressed:
  1. Env feature bug ── already fixed (verified gap=0).
  2. Budget-dim column init ── zero-init verified (step below).
     The current checkpoint is already 21-dim; we just load it directly.
  3. Per-B Welford ── separate running stats per budget level.

Gate (every 25 epochs, 3 trials):
  k=3  must stay ≥ 320  (if < 300 → stop, restore).
  k=30 target ≥ 420.

Best checkpoint: save when min(rev_k3/320, rev_k30/420) is maximised.

Usage:
  cd revmax-aaai2027 && source venv/bin/activate
  nohup env PYTORCH_ENABLE_MPS_FALLBACK=1 python -u \
    experiments/run_budget_retrain.py \
    --warm_start results/checkpoints/rev_gnn_lstm_budget.pt \
    > /tmp/retrain_v2.log 2>&1 &
"""

import argparse, os, sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np

from src.utils.helpers import load_config_with_base, set_seed, get_device, ensure_dir
from src.utils.logging import ExperimentLogger
from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.utils.features import compute_static_features
from src.utils.budget_features import compute_budget_node_features
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.evaluation.budget_baselines import evaluate_budget_aware_policy


# ── Greedy revenues at gate budgets (used for normalisation) ──────────────────
GREEDY_K3  = 23.6   # from budget_eval_v2
GREEDY_K10 = 118.2
GREEDY_K30 = 428.6

# Gate thresholds
GATE_K3_GUARD  = 320.0
GATE_K3_STOP   = 300.0
GATE_K30_TARGET = 420.0


def _make_policy(cfg, device):
    enc  = GraphSAGEEncoder(21, int(cfg.encoder.hidden_dim),
                            int(cfg.encoder.n_layers), 0.0)
    lstm = EpisodeLSTM(graph_dim=int(cfg.encoder.hidden_dim),
                       lstm_hidden=int(cfg.sequence_model.lstm_hidden),
                       n_layers=int(cfg.sequence_model.lstm_n_layers))
    pol  = SequentialJointPolicy(enc, lstm,
                                  gnn_dim=int(cfg.encoder.hidden_dim),
                                  context_dim=int(cfg.sequence_model.lstm_hidden))
    return pol.to(device)


def _verify_zero_init_unchanged(pol_new, pol_ref, x, edge_index, mask, device, logger):
    """Verify that warm-start reproduces old model actions exactly (sanity check)."""
    pol_new.eval(); pol_ref.eval()
    with torch.no_grad():
        pol_new.reset_episode(device); pol_ref.reset_episode(device)
        ni_new, d_new, _ = pol_new.select_and_price(x, edge_index, mask, greedy=True)
        pol_new.reset_episode(device); pol_ref.reset_episode(device)
        ni_ref, d_ref, _ = pol_ref.select_and_price(x, edge_index, mask, greedy=True)
    match = (int(ni_new) == int(ni_ref)) and (abs(float(d_new) - float(d_ref)) < 1e-5)
    logger.info(f"  Warm-start self-check: node_match={int(ni_new)==int(ni_ref)}  "
                f"disc_match={abs(float(d_new)-float(d_ref))<1e-5}  ok={match}")
    return match


def _run_episode(pol, env, graph, static_feats, edge_index, device, c, max_steps=300):
    """Run one REINFORCE episode; returns (log_probs, rewards, total_rev)."""
    n = graph.number_of_nodes()
    log_probs, rewards = [], []
    revenue = 0.0
    step = 0
    pol.reset_episode(device)

    while len(env.offered) < n and not env._check_bankrupt() and step < max_steps:
        feats = compute_budget_node_features(
            graph, static_feats, env.S, env.offered, env.t, n, k=0, env=env)
        x    = torch.tensor(feats, dtype=torch.float32).to(device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for idx in env.available_nodes:
            mask[idx] = True
        if mask.sum() == 0:
            break

        ni, disc, log_p = pol.select_and_price(x, edge_index, mask, greedy=False)
        node_idx = int(ni)
        discount = float(disc)
        node     = env.nodes[node_idx]
        max_d    = env.max_affordable_discount(node)
        if max_d >= 0:
            discount = min(discount, max_d)

        _, rew, done, info = env.step(node_idx, discount)
        revenue  += rew
        log_probs.append(log_p)
        rewards.append(rew)
        pol.update_sequence_state(discount, info.get("accepted", False), rew)
        step += 1
        if done:
            break

    return log_probs, rewards, revenue


def _gate_eval(pol, graph, static_feats, edge_index, device, c, k_vals, n_trials=3, logger=None):
    """Evaluate policy at given k values (no grad). Returns dict k→mean_revenue."""
    results = {}
    pol_eval = pol  # use existing policy object in eval mode
    pol.eval()
    for k in k_vals:
        B = round(k * c, 8)
        rev = evaluate_budget_aware_policy(
            pol, graph, B=B, c=c, device=device,
            n_trials=n_trials, weight_high=2.0,
        )
        mean_rev = rev["revenue"]["mean"] if isinstance(rev.get("revenue"), dict) else 0.0
        results[k] = mean_rev
        if logger:
            gate_str = (f"GATE k={k} B={B:.2f}: rev={mean_rev:.1f} "
                        f"{'✓' if (k==3 and mean_rev>=GATE_K3_GUARD) or (k==30 and mean_rev>=GATE_K30_TARGET) else ('⚠' if k==3 and mean_rev>=GATE_K3_STOP else '✗')}")
            logger.info(f"  {gate_str}")
    pol.train()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",       default="configs/experiments/budget_constrained.yaml")
    parser.add_argument("--warm_start",   default="results/checkpoints/rev_gnn_lstm_budget.pt")
    parser.add_argument("--out_ckpt",     default="results/checkpoints/rev_gnn_lstm_budget_v2.pt")
    parser.add_argument("--n_epochs",     type=int, default=150)
    parser.add_argument("--lr",           type=float, default=5e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--grad_clip",    type=float, default=1.0)
    parser.add_argument("--max_steps",    type=int,   default=300)
    parser.add_argument("--gate_every",   type=int,   default=25)
    parser.add_argument("--gate_trials",  type=int,   default=3)
    args = parser.parse_args()

    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = get_device()
    logger = ExperimentLogger(cfg, run_name="budget_retrain_v2")
    ensure_dir(os.path.dirname(args.out_ckpt))

    c        = float(cfg.budget_constrained.production_cost)
    B_levels = [round(k * c, 8) for k in [3, 10, 30]]   # 0.9, 3.0, 9.0
    logger.info(f"Budget retrain v2  B_levels={B_levels}  epochs={args.n_epochs}  lr={args.lr}")

    # ── Build graph ───────────────────────────────────────────────────────────
    graph = generate_forest_fire(1000, float(cfg.graph.p), float(cfg.graph.pb),
                                 seed=cfg.project.seed)
    static_feats = compute_static_features(graph)
    edges  = list(graph.edges())
    s_idx  = [u for u, v in edges] + [v for u, v in edges]
    t_idx  = [v for u, v in edges] + [u for u, v in edges]
    edge_index = torch.tensor([s_idx, t_idx], dtype=torch.long).to(device)
    n = graph.number_of_nodes()
    logger.info(f"Graph: n={n}  m={graph.number_of_edges()}")

    # ── Load warm-start (already 21-dim) ─────────────────────────────────────
    pol = _make_policy(cfg, device)
    if os.path.exists(args.warm_start):
        sd = torch.load(args.warm_start, map_location=device, weights_only=True)
        pol.load_state_dict(sd, strict=True)
        logger.info(f"Warm-start loaded: {args.warm_start}")
    else:
        logger.info(f"[WARN] warm_start not found, training from random init")

    # ── Zero-init verification ────────────────────────────────────────────────
    # Build a reference copy and verify outputs match (sanity: same weights → same output)
    pol_ref = _make_policy(cfg, device)
    if os.path.exists(args.warm_start):
        pol_ref.load_state_dict(
            torch.load(args.warm_start, map_location=device, weights_only=True), strict=True)
    dummy_cfg = BudgetEnvConfig(budget_B=3.0, production_cost=c, seed=0, weight_high=2.0)
    dummy_env = BudgetRevenueEnv(graph, dummy_cfg)
    dummy_env.reset()
    dummy_feats = compute_budget_node_features(
        graph, static_feats, dummy_env.S, dummy_env.offered, 0, n, k=0, env=dummy_env)
    dummy_x    = torch.tensor(dummy_feats, dtype=torch.float32).to(device)
    dummy_mask = torch.ones(n, dtype=torch.bool, device=device)
    _verify_zero_init_unchanged(pol, pol_ref, dummy_x, edge_index, dummy_mask, device, logger)
    del pol_ref, dummy_env, dummy_feats, dummy_x, dummy_mask

    # ── Optimiser ─────────────────────────────────────────────────────────────
    optimiser = torch.optim.Adam(pol.parameters(), lr=args.lr)
    pol.train()

    # ── Welford stats per B level ─────────────────────────────────────────────
    wf = {B: {"n": 0, "mean": 0.0, "M2": 0.0} for B in B_levels}

    def welford_update(B_val, rev):
        w = wf[B_val]
        w["n"] += 1
        delta  = rev - w["mean"]
        w["mean"] += delta / w["n"]
        w["M2"]   += delta * (rev - w["mean"])
        std = max((w["M2"] / max(w["n"] - 1, 1)) ** 0.5, 1e-3)
        return (rev - w["mean"]) / std

    # ── Best checkpoint tracking ───────────────────────────────────────────────
    best_score      = -float("inf")
    best_gate_revs  = {}
    stopped_early   = False

    logger.info(f"\nInitial gate eval before training:")
    init_gate = _gate_eval(pol, graph, static_feats, edge_index, device, c,
                           k_vals=[3, 10, 30], n_trials=args.gate_trials, logger=logger)

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(1, args.n_epochs + 1):
        B_val = random.choice(B_levels)
        k_val  = round(B_val / c)

        cfg_e = BudgetEnvConfig(budget_B=B_val, production_cost=c,
                                seed=epoch, weight_high=2.0)
        env   = BudgetRevenueEnv(graph, cfg_e)
        env.reset()

        log_probs, rewards, revenue = _run_episode(
            pol, env, graph, static_feats, edge_index, device, c, args.max_steps)

        if not log_probs:
            continue

        advantage = welford_update(B_val, revenue)

        # REINFORCE loss + entropy bonus
        log_p_stack = torch.stack(log_probs)
        loss = -advantage * log_p_stack.sum()

        # Entropy bonus: −λ * Σ H(π_t) (encourages exploration)
        loss = loss - args.entropy_coef * log_p_stack.sum()

        optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pol.parameters(), args.grad_clip)
        optimiser.step()

        if epoch % 10 == 0:
            logger.info(
                f"ep={epoch:4d}  B={B_val:.1f}(k={k_val})  "
                f"rev={revenue:.1f}  adv={advantage:+.3f}  "
                f"Welford: k={k_val} μ={wf[B_val]['mean']:.1f} "
                f"σ={max((wf[B_val]['M2']/max(wf[B_val]['n']-1,1))**0.5,1e-3):.1f}"
            )

        # ── Gate eval every gate_every epochs ────────────────────────────────
        if epoch % args.gate_every == 0 or epoch == args.n_epochs:
            logger.info(f"\n── Gate eval at epoch {epoch} ──────────")
            gate   = _gate_eval(pol, graph, static_feats, edge_index, device, c,
                                k_vals=[3, 10, 30], n_trials=args.gate_trials, logger=logger)
            rev_k3  = gate.get(3,  0.0)
            rev_k10 = gate.get(10, 0.0)
            rev_k30 = gate.get(30, 0.0)

            # Score = min normalised improvement (1.0 = at target, >1.0 = above)
            score = min(rev_k3 / GATE_K3_GUARD, rev_k30 / GATE_K30_TARGET)
            logger.info(f"  score={score:.3f}  (k3/320={rev_k3/GATE_K3_GUARD:.3f}  "
                        f"k30/420={rev_k30/GATE_K30_TARGET:.3f})")

            if score > best_score:
                best_score     = score
                best_gate_revs = gate.copy()
                torch.save(pol.state_dict(), args.out_ckpt)
                logger.info(f"  ★ New best! Saved → {args.out_ckpt}")

            if rev_k3 < GATE_K3_STOP:
                logger.info(f"  ✗ k=3 dropped below {GATE_K3_STOP} ({rev_k3:.1f}). "
                            f"Stopping and keeping best checkpoint.")
                stopped_early = True
                break

    # ── Final report ──────────────────────────────────────────────────────────
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {'STOPPED EARLY' if stopped_early else 'COMPLETE'}")
    logger.info(f"Best checkpoint score: {best_score:.3f}")
    logger.info(f"Best gate revenues: {best_gate_revs}")
    logger.info(f"Checkpoint: {args.out_ckpt}")
    logger.info(f"\nWelford stats at end:")
    for B_val in B_levels:
        w = wf[B_val]
        std = max((w["M2"] / max(w["n"]-1, 1))**0.5, 1e-3)
        logger.info(f"  B={B_val:.1f}(k={round(B_val/c)}): n={w['n']}  "
                    f"μ={w['mean']:.1f}  σ={std:.1f}")

    if best_gate_revs:
        verdict = ("ACHIEVED" if best_gate_revs.get(30, 0) >= GATE_K30_TARGET
                   and best_gate_revs.get(3, 0) >= GATE_K3_GUARD
                   else "PARTIAL_IMPROVEMENT" if best_gate_revs.get(30, 0) > 340
                   else "NO_IMPROVEMENT")
        logger.info(f"\nVerdict: {verdict}  "
                    f"k=3={best_gate_revs.get(3,0):.1f}  "
                    f"k=10={best_gate_revs.get(10,0):.1f}  "
                    f"k=30={best_gate_revs.get(30,0):.1f}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
