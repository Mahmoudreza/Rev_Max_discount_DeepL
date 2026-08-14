#!/usr/bin/env python3
"""experiments/run_budget_unified_training.py — Unified budget model training.

ONE-SHOT training of a budget-aware GNN+LSTM policy that covers ALL budget
regimes [k=1..40] via:
  Phase 1 (300 epochs): Mixed-expert imitation from regime-appropriate champion
  Phase 2 (200 epochs): Full-range REINFORCE with per-bucket Welford advantage

Architecture: SequentialJointPolicy (GNN+LSTM), input_dim=21.
Init: rev_gnn_lstm.pt (Idea-1, 8fbc4648), input_proj extended 20→21
      with column 21 ZERO-INITIALIZED.
Feature 21: B_t / (40 * c)  — budget framed against max possible, not episode B0.

Output: results/checkpoints/rev_gnn_lstm_unified.pt (best on min-bucket reward)

DO NOT modify this file after the run has started.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.budget_features import (
    compute_budget_node_features_fast,
    build_graph_feature_cache,  # re-exported via budget_features
)
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.training.mixed_expert_trajectories import (
    build_trajectory_cache,
    generate_budget_expert_trajectory,
)

# ── Constants ──────────────────────────────────────────────────────────────────
C         = 0.3
B_MAX     = 40 * C           # = 12.0  — normalisation denominator for feature 21
CKPT_DIR  = "results/checkpoints"
LOG_DIR   = "results/logs"
BASE_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm.pt")
OUT_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")
SHA256_BASE = "8fbc4648"     # expected SHA256 prefix of rev_gnn_lstm.pt

# Training graph sizes (standard FF set, matches prior experiments)
TRAIN_SIZES = [200, 260, 320, 380, 440]
FF_P, FF_PB = 0.37, 0.32

# Phase 1
PH1_EPOCHS    = 300
K_SAMPLES_P1  = [1, 3, 5, 10, 15, 25, 40]
N_SEEDS       = 10           # seeds 0..9 per (graph, k)
PRICE_ALPHA   = 0.3          # weight of MSE pricing loss

# Phase 2
PH2_EPOCHS    = 200
PH2_LR        = 5e-5
ENTROPY_COEF  = 0.01
GRAD_CLIP     = 1.0
STD_FLOOR     = 1.0          # MUST be 1.0 — do not change to 1e-8

# Welford buckets (k ranges)
BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40)]

def _bucket_of(k: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= k <= hi:
            return i
    return len(BUCKETS) - 1


os.makedirs(CKPT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# ── Feature computation ────────────────────────────────────────────────────────

def _unified_features(cache: dict, env: BudgetRevenueEnv, k: int) -> np.ndarray:
    """21-dim features: 20 standard + feature_20 = B_t / B_MAX (NOT B_t/B_0).

    Using B_MAX = 40*c rather than B_0 gives the model cross-episode regime
    context: at episode start, k=1 → 0.025, k=40 → 1.0.

    Args:
        cache: build_graph_feature_cache output.
        env:   Live BudgetRevenueEnv.
        k:     Current episode k (for round_ratio normalisation).

    Returns:
        (n, 21) float32 array.
    """
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    n = cache["n"]
    budget_col = np.full((n, 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, budget_col], axis=1)


def _to_edge_index(graph, device) -> torch.Tensor:
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    nodes = list(graph.nodes())
    nmap = {v: i for i, v in enumerate(nodes)}
    src_i = [nmap[v] for v in src]
    dst_i = [nmap[v] for v in dst]
    ei = torch.tensor([src_i, dst_i], dtype=torch.long, device=device)
    return ei


def _avail_mask(env: BudgetRevenueEnv, n: int, device) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes:
        mask[idx] = True
    return mask


# ── Model loading + 20→21 extension ───────────────────────────────────────────

def _build_policy_20dim():
    """Build a fresh SequentialJointPolicy with 20-dim input."""
    enc = GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)


def _build_policy_21dim():
    """Build a fresh SequentialJointPolicy with 21-dim input."""
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)


def _load_and_extend_policy(ckpt_path: str, device: torch.device) -> SequentialJointPolicy:
    """Load Idea-1 checkpoint (20-dim) and zero-extend input projection to 21-dim.

    The new input_proj.weight has shape (64, 21):
      columns 0-19: copied from original (64, 20) weight
      column  20:   zeros (so dim 21 is initially ignored → preserves behavior)

    Args:
        ckpt_path: Path to rev_gnn_lstm.pt.
        device:    Target device.

    Returns:
        SequentialJointPolicy with 21-dim input, ready for training.
    """
    # ── Verify SHA256 prefix ──────────────────────────────────────────────────
    import hashlib
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    assert sha.startswith(SHA256_BASE), (
        f"ABORT: {ckpt_path} SHA256={sha[:8]} expected {SHA256_BASE}. "
        "Wrong base checkpoint."
    )
    print(f"[init] Base checkpoint verified: {sha[:16]}...")

    # ── Load original 20-dim model ────────────────────────────────────────────
    policy_20 = _build_policy_20dim().to(device)
    state_20  = torch.load(ckpt_path, map_location=device)
    if "policy_state_dict" in state_20:
        state_20 = state_20["policy_state_dict"]
    elif "model_state_dict" in state_20:
        state_20 = state_20["model_state_dict"]
    policy_20.load_state_dict(state_20, strict=True)

    # ── Build 21-dim model ────────────────────────────────────────────────────
    policy_21 = _build_policy_21dim().to(device)

    # ── Copy all weights except input_proj ────────────────────────────────────
    state_21 = policy_21.state_dict()
    for k, v in state_20.items():
        if k in state_21 and k != "encoder.input_proj.weight":
            state_21[k] = v.clone()

    # ── Extend input_proj.weight: (64,20) → (64,21) with zero column ─────────
    old_w = state_20["encoder.input_proj.weight"]   # (64, 20)
    new_w = state_21["encoder.input_proj.weight"]   # (64, 21)
    new_w[:, :20] = old_w
    new_w[:, 20]  = 0.0                              # zero-init: dim 21 ignored at start
    state_21["encoder.input_proj.weight"] = new_w

    policy_21.load_state_dict(state_21, strict=True)
    print(f"[init] 21-dim policy built: {sum(p.numel() for p in policy_21.parameters())} params")
    return policy_21, policy_20


# ── Behavior-preservation check ───────────────────────────────────────────────

@torch.no_grad()
def _behavior_preservation_check(
    policy_20: SequentialJointPolicy,
    policy_21: SequentialJointPolicy,
    graph,
    device: torch.device,
    n_steps: int = 10,
) -> bool:
    """Verify that 21-dim model with zero column 20 == 20-dim model.

    Runs both policies greedy on the same test graph for n_steps.
    Prints a comparison table. Aborts if any step differs.

    Args:
        policy_20: Original 20-dim policy.
        policy_21: Zero-extended 21-dim policy.
        graph:     Test graph.
        device:    Device.
        n_steps:   Steps to compare.

    Returns:
        True if all steps match, False if mismatch detected.
    """
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.features import compute_static_features

    print("\n── Behavior-preservation check (20-dim vs 21-dim zero-padded) ──")
    print(f"{'step':>5} | {'top-node 20dim':>14} | {'top-node 21dim':>14} | match?")
    print("-" * 55)

    n = graph.number_of_nodes()
    ei = _to_edge_index(graph, device)
    static_feats = compute_static_features(graph)

    def _run_policy(policy, use_21: bool):
        """Run policy greedy for n_steps, return list of selected node_idx."""
        from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
        cfg = BudgetEnvConfig(budget_B=12.0, production_cost=C, seed=0)
        env = BudgetRevenueEnv(graph, cfg)
        env.reset()

        cache = build_graph_feature_cache(graph, static_feats)
        policy.eval()
        policy.reset_episode(device)

        selected = []
        for _ in range(n_steps):
            if not env.available_nodes:
                break
            avail = _avail_mask(env, n, device)
            if use_21:
                x_np = _unified_features(cache, env, k=40)   # use max k; col20 ~1.0
                # Force col 20 = 0.0 to match 20-dim behavior
                x_np[:, 20] = 0.0
            else:
                from src.utils.features import compute_node_features_fast
                x_np = compute_node_features_fast(cache, env.S, env.offered, env.t, 40, env)
            x_t = torch.FloatTensor(x_np).to(device)
            scores, h, ctx, _ = policy.forward(x_t, ei, avail)
            idx = int(scores.argmax().item())
            selected.append(idx)
            # Step with dummy discount (0.5) — only care about ordering
            env.step(idx, 0.5)
            policy.update_sequence_state(0.5, True, 0.1)
        return selected

    sel_20 = _run_policy(policy_20, use_21=False)
    sel_21 = _run_policy(policy_21, use_21=True)

    all_match = True
    for t, (a, b) in enumerate(zip(sel_20, sel_21)):
        match = "✓" if a == b else "✗ MISMATCH"
        if a != b:
            all_match = False
        print(f"{t:>5} | {a:>14} | {b:>14} | {match}")

    if all_match:
        print("\n[check] PASS — all 10 steps match. Zero-init preserves behavior.\n")
    else:
        print("\n[check] FAIL — mismatch detected. Aborting.\n")
    return all_match


# ── Welford per-bucket running stats ──────────────────────────────────────────

class WelfordBucket:
    """Running mean and std estimator for one k-bucket."""
    def __init__(self):
        self.n   = 0
        self.m1  = 0.0    # running mean
        self.m2  = 0.0    # sum of squared deviations

    def update(self, x: float):
        self.n += 1
        delta  = x - self.m1
        self.m1 += delta / self.n
        delta2  = x - self.m1
        self.m2 += delta * delta2

    @property
    def mean(self) -> float:
        return self.m1

    @property
    def std(self) -> float:
        if self.n < 2:
            return STD_FLOOR
        return max(math.sqrt(self.m2 / (self.n - 1)), STD_FLOOR)

    def advantage(self, x: float) -> float:
        return (x - self.mean) / self.std


# ── Phase 1: Mixed-expert imitation ───────────────────────────────────────────

def phase1_imitation(
    policy: SequentialJointPolicy,
    training_data: List[dict],
    n_epochs: int,
    device: torch.device,
    lr: float = 1e-3,
    log_every: int = 50,
) -> dict:
    """Train policy via mixed-expert imitation.

    Args:
        policy:        21-dim SequentialJointPolicy in train mode.
        training_data: List of {graph, cache, edge_index, static_feats, n}.
        n_epochs:      Number of training epochs.
        device:        Training device.
        lr:            Learning rate.
        log_every:     Print per-k CE breakdown every this many epochs.

    Returns:
        phase1_log: dict of per-epoch losses.
    """
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    rng = np.random.default_rng(42)
    log = []

    print(f"\n── Phase 1: Imitation ({n_epochs} epochs, {len(training_data)} graphs) ──")

    for epoch in range(n_epochs):
        epoch_loss = 0.0
        epoch_ce   = 0.0
        epoch_mse  = 0.0
        epoch_steps = 0

        # Per-k accumulators (for log_every reporting)
        k_ce_sum: Dict[int, float] = {k: 0.0 for k in K_SAMPLES_P1}
        k_steps:  Dict[int, int]   = {k: 0    for k in K_SAMPLES_P1}

        for gdata in training_data:
            graph  = gdata["graph"]
            cache  = gdata["cache"]
            ei     = gdata["edge_index"]
            n      = gdata["n"]

            k      = int(rng.choice(K_SAMPLES_P1))
            seed   = int(epoch % N_SEEDS)
            traj   = generate_budget_expert_trajectory(graph, k, c=C, seed=seed)
            if not traj:
                continue

            B = k * C
            cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed)
            env = BudgetRevenueEnv(graph, cfg)
            env.reset()

            policy.train()
            policy.reset_episode(device)

            step_ce  = torch.tensor(0.0, device=device)
            step_mse = torch.tensor(0.0, device=device)
            n_valid  = 0

            for step in traj:
                expert_idx  = step["node_idx"]
                expert_disc = step["discount"]

                # Skip steps where expert node is unavailable in env (shouldn't happen)
                if expert_idx not in env.available_nodes:
                    break

                x_np = _unified_features(cache, env, k=k)
                x_t  = torch.FloatTensor(x_np).to(device)
                avail = _avail_mask(env, n, device)

                masked_scores, h, context, _ = policy.forward(x_t, ei, avail)

                # CE loss on node selection
                log_probs = F.log_softmax(masked_scores, dim=0)
                ce = -log_probs[expert_idx]
                step_ce = step_ce + ce

                # MSE loss on Beta pricing mean vs expert discount
                comb = torch.cat([h[expert_idx], context], dim=0)
                beta = policy.get_discount_distribution(comb)
                mse  = (beta.mean - expert_disc) ** 2
                step_mse = step_mse + PRICE_ALPHA * mse

                n_valid += 1

                # Teacher forcing: step env with expert action
                env.step(expert_idx, expert_disc)
                policy.update_sequence_state(
                    expert_disc, step["accepted"], step["price"]
                )

            if n_valid == 0:
                continue

            loss = (step_ce + step_mse) / n_valid
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
            optimizer.step()

            epoch_loss  += float(loss.item())
            epoch_ce    += float(step_ce.item()) / n_valid
            epoch_mse   += float(step_mse.item()) / n_valid
            epoch_steps += 1

            k_ce_sum[k] += float(step_ce.item()) / n_valid
            k_steps[k]  += 1

        n_g = max(epoch_steps, 1)
        avg_loss = epoch_loss / n_g

        log.append({"epoch": epoch, "loss": avg_loss,
                    "ce": epoch_ce / n_g, "mse": epoch_mse / n_g})

        if (epoch + 1) % log_every == 0 or epoch == 0:
            per_k = {k: (k_ce_sum[k] / max(k_steps[k], 1)) for k in K_SAMPLES_P1}
            k_str = "  ".join(f"k{k}={per_k[k]:.3f}" for k in K_SAMPLES_P1)
            print(
                f"  ep{epoch+1:>3}/{n_epochs} | loss={avg_loss:.4f}"
                f" | CE per-k: {k_str}"
            )

    print(f"[Phase 1] Done. Final loss={log[-1]['loss']:.4f}\n")
    return {"phase1": log}


# ── Phase 2: Full-range REINFORCE ─────────────────────────────────────────────

def phase2_reinforce(
    policy: SequentialJointPolicy,
    training_data: List[dict],
    n_epochs: int,
    device: torch.device,
    save_every: int = 25,
) -> dict:
    """Full-range REINFORCE with per-bucket Welford advantage normalisation.

    k sampled log-uniform over [1, 40].
    Advantage normalised within bucket; std_floor = 1.0.
    Best checkpoint: saved when min-bucket normalised reward improves.

    Args:
        policy:        Policy (continues from Phase 1 weights).
        training_data: Training graph data list.
        n_epochs:      Phase 2 epochs.
        device:        Device.
        save_every:    Save checkpoint every this many epochs.

    Returns:
        phase2_log: training history + best_epoch.
    """
    optimizer  = torch.optim.Adam(policy.parameters(), lr=PH2_LR)
    rng        = np.random.default_rng(seed=123)
    welfords   = [WelfordBucket() for _ in BUCKETS]
    log        = []
    best_min_adv = -float("inf")
    best_epoch   = -1

    print(f"── Phase 2: REINFORCE ({n_epochs} epochs) ──")
    print(f"   lr={PH2_LR}, entropy={ENTROPY_COEF}, std_floor={STD_FLOOR}")
    print(f"   Buckets: {BUCKETS}\n")

    for epoch in range(n_epochs):
        gdata    = training_data[epoch % len(training_data)]
        graph    = gdata["graph"]
        cache    = gdata["cache"]
        ei       = gdata["edge_index"]
        n        = gdata["n"]

        # Sample k log-uniform over [1, 40]
        log_lo, log_hi = math.log(1.0), math.log(40.001)
        k = max(1, min(40, int(math.exp(rng.uniform(log_lo, log_hi)))))
        b_idx = _bucket_of(k)
        B = k * C

        cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=epoch)
        env = BudgetRevenueEnv(graph, cfg)
        env.reset()

        policy.train()
        policy.reset_episode(device)

        log_probs_list = []
        entropies_list = []
        episode_revenue = 0.0

        # ── Roll out one episode ──────────────────────────────────────────────
        for _t in range(n):
            avail = env.available_nodes
            if not avail:
                break
            # Budget check
            if env._check_bankrupt():
                break

            x_np = _unified_features(cache, env, k=k)
            x_t  = torch.FloatTensor(x_np).to(device)
            avail_mask = _avail_mask(env, n, device)

            masked_scores, h, context, _ = policy.forward(x_t, ei, avail_mask)

            # Stochastic node selection
            probs    = F.softmax(masked_scores, dim=0)
            cat_dist = torch.distributions.Categorical(probs)
            node_idx = int(cat_dist.sample().item())
            lp_node  = cat_dist.log_prob(torch.tensor(node_idx, device=device))

            # Beta discount
            comb      = torch.cat([h[node_idx], context], dim=0)
            beta_dist = policy.get_discount_distribution(comb)
            discount_t = beta_dist.rsample().clamp(1e-6, 1.0 - 1e-6)
            lp_disc   = beta_dist.log_prob(discount_t)
            entropy   = beta_dist.entropy()

            discount  = float(discount_t.item())
            lp_total  = lp_node + lp_disc
            log_probs_list.append(lp_total)
            entropies_list.append(entropy)

            # Budget-SKIP enforcement: if we can't afford this, mark skip + break
            est_val      = env._estimate_valuation(env.nodes[node_idx])
            offered_price = est_val * (1.0 - discount)
            if env.B - C + offered_price < -1e-9:
                # Policy tried to overspend → stop episode (implicit SKIP signal)
                break

            obs, reward, done, info = env.step(node_idx, discount)
            episode_revenue += info["offered_price"] if info["accepted"] else 0.0
            policy.update_sequence_state(discount, info["accepted"], info.get("revenue_step", 0.0))

            if done:
                break

        if not log_probs_list:
            continue

        # ── Advantage computation ─────────────────────────────────────────────
        w     = welfords[b_idx]
        adv   = w.advantage(episode_revenue)
        w.update(episode_revenue)

        # ── Policy gradient loss ──────────────────────────────────────────────
        lp_sum = sum(log_probs_list)
        ent_sum = sum(entropies_list)
        loss   = -adv * lp_sum - ENTROPY_COEF * ent_sum

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        # Track min bucket advantage for best-checkpoint criterion
        min_adv = min(
            (w.advantage(w.mean) if w.n >= 2 else -float("inf"))
            for w in welfords
        )
        log.append({
            "epoch":   epoch,
            "k":       k,
            "bucket":  b_idx,
            "revenue": episode_revenue,
            "adv":     adv,
            "loss":    float(loss.item()),
        })

        if (epoch + 1) % save_every == 0:
            _save_checkpoint(policy, epoch, f"{OUT_CKPT.replace('.pt', f'_ep{epoch+1}.pt')}")
            b_stats = [f"B{i}({welfords[i].mean:.1f}±{welfords[i].std:.1f})"
                       for i in range(len(BUCKETS))]
            print(f"  ep{epoch+1:>3}/{n_epochs} | k={k} "
                  f"rev={episode_revenue:.2f} adv={adv:.3f} | {' '.join(b_stats)}")

        # Best-checkpoint: maximize minimum bucket mean reward
        # (Proxy: use current episode revenue as reference after phase 1 warmup)
        if epoch >= 20:  # Wait 20 eps for Welford to stabilise
            bucket_means = [welfords[i].mean if welfords[i].n >= 1 else 0.0
                            for i in range(len(BUCKETS))]
            min_mean = min(bucket_means)
            if min_mean > best_min_adv:
                best_min_adv = min_mean
                best_epoch   = epoch + 1
                torch.save(policy.state_dict(), OUT_CKPT)
                print(f"  [BEST] ep{epoch+1} min-bucket-mean={min_mean:.3f} → {OUT_CKPT}")

    print(f"\n[Phase 2] Done. Best checkpoint: ep{best_epoch}, "
          f"min-bucket-mean={best_min_adv:.3f}")

    return {"phase2": log, "best_epoch": best_epoch, "best_min_mean": best_min_adv}


def _save_checkpoint(policy, epoch, path):
    torch.save(policy.state_dict(), path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()
    print("=" * 70)
    print("run_budget_unified_training.py — Unified budget model (ONE-SHOT)")
    print("=" * 70)

    # ── Device ────────────────────────────────────────────────────────────────
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Build training graphs ─────────────────────────────────────────────────
    print(f"\nGenerating {len(TRAIN_SIZES)} training graphs (FF, p={FF_P}, pb={FF_PB})...")
    graphs = []
    for i, n in enumerate(TRAIN_SIZES):
        g = generate_forest_fire(n, p=FF_P, pb=FF_PB, seed=i)
        graphs.append(g)
        print(f"  g{i}: n={g.number_of_nodes()} m={g.number_of_edges()}")

    # ── Build feature caches ───────────────────────────────────────────────────
    training_data = []
    for i, graph in enumerate(graphs):
        static_feats = compute_static_features(graph)
        cache = build_graph_feature_cache(graph, static_feats)
        ei    = _to_edge_index(graph, device)
        training_data.append({
            "graph": graph, "cache": cache,
            "edge_index": ei, "n": graph.number_of_nodes(),
        })

    # ── Build trajectory cache ────────────────────────────────────────────────
    print("\nBuilding expert trajectory cache...")
    summary = build_trajectory_cache(
        graphs, k_list=tuple(K_SAMPLES_P1), n_seeds=N_SEEDS, c=C, verbose=True
    )
    print(f"\nTrajectory cache complete. ETA Phase1+2 ~{(PH1_EPOCHS + PH2_EPOCHS) * len(graphs) * 2 // 60} min")

    # ── Load + extend base policy ─────────────────────────────────────────────
    policy_21, policy_20 = _load_and_extend_policy(BASE_CKPT, device)

    # ── Behavior-preservation check ───────────────────────────────────────────
    test_graph = graphs[0]
    ok = _behavior_preservation_check(policy_20, policy_21, test_graph, device, n_steps=10)
    if not ok:
        print("ABORT: behavior-preservation check failed. Investigate and rerun.")
        sys.exit(1)

    # ── Phase 1: Imitation ────────────────────────────────────────────────────
    policy_21 = policy_21.to(device)
    ph1_log = phase1_imitation(
        policy_21, training_data, n_epochs=PH1_EPOCHS, device=device, lr=1e-3
    )
    # Save Phase-1 checkpoint (before Phase-2 overwrites best)
    ph1_ckpt = OUT_CKPT.replace(".pt", "_ph1end.pt")
    torch.save(policy_21.state_dict(), ph1_ckpt)
    print(f"[Phase 1 end] Checkpoint saved: {ph1_ckpt}")

    # ── Phase 2: REINFORCE ────────────────────────────────────────────────────
    ph2_log = phase2_reinforce(
        policy_21, training_data, n_epochs=PH2_EPOCHS,
        device=device, save_every=25
    )

    # ── Final: ensure best checkpoint exists ──────────────────────────────────
    if not os.path.exists(OUT_CKPT):
        torch.save(policy_21.state_dict(), OUT_CKPT)
        print(f"[final] No best checkpoint found; saving final weights → {OUT_CKPT}")

    # ── Verify final checkpoint hash ──────────────────────────────────────────
    import hashlib
    final_sha = hashlib.sha256(open(OUT_CKPT, "rb").read()).hexdigest()
    print(f"\n[final] {OUT_CKPT}  SHA256={final_sha[:16]}...")

    # ── Save training log ─────────────────────────────────────────────────────
    log_path = os.path.join(LOG_DIR, "unified_training_log.json")
    full_log = {
        "phase1": ph1_log["phase1"],
        "phase2": ph2_log["phase2"],
        "best_epoch": ph2_log["best_epoch"],
        "best_min_mean": ph2_log["best_min_mean"],
        "final_sha256": final_sha,
        "duration_s": time.time() - t0,
        "expert_summary": summary,
    }
    with open(log_path, "w") as f:
        json.dump(full_log, f, indent=2)
    print(f"[log] Training log → {log_path}")

    elapsed = time.time() - t0
    print(f"\nTotal wall time: {elapsed/60:.1f} min")
    print("DONE — run experiments/run_unified_sweep.py to evaluate.")


if __name__ == "__main__":
    main()
