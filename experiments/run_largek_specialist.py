#!/usr/bin/env python3
"""experiments/run_largek_specialist.py — Train large-k budget specialist.

Warm-starts from rev_gnn_lstm_unified.pt (ep200, sha1 a7b7081d).

Phase 1 — Imitation (150 epochs):
    Teacher: greedy_discount_budget trajectories from largek_traj_cache.
    Loss: CE(selection) + 0.3 * MSE(discount)
    Adam lr=1e-4, batch by episode.

Phase 2 — REINFORCE fine-tune (100 epochs):
    k ~ log-uniform [16, 40], graphs sampled from the 5 training sizes.
    lr=1e-5, entropy=0.01, grad_clip=1.0.
    Welford baseline per k-bucket {16-20, 21-30, 31-40}, std floor=1.0.
    Best checkpoint = max over epochs of min per-bucket normalised reward.

Output: results/checkpoints/rev_gnn_lstm_largek.pt
        results/logs/largek_specialist_train.json
Log:    /tmp/largek_specialist.log  (this script is run via nohup)
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.utils.budget_features import compute_budget_node_features_fast

# ── Constants ──────────────────────────────────────────────────────────────────
C              = 0.3
# Training graphs: same 5 graphs as unified model (zero-shot protocol).
# n=1000 eval graph is quarantined in largek_traj_cache_EVALGRAPH_DO_NOT_TRAIN/.
GRAPH_SIZES    = [200, 260, 320, 380, 440]
K_LIST_TRAIN   = [16, 20, 25, 30, 40]
GRAPH_SEED     = 42    # matches Stage-B / unified model provenance
WEIGHT_HIGH    = 2.0

# Warm-start checkpoint
WARM_CKPT      = "results/checkpoints/rev_gnn_lstm_unified.pt"
WARM_SHA1      = "a7b7081d"   # ep200 unified

# Output
CKPT_OUT       = "results/checkpoints/rev_gnn_lstm_largek.pt"
LOG_OUT        = "results/logs/largek_specialist_train.json"
CACHE_DIR      = "results/logs/largek_traj_cache"

# Phase 1 hyper-parameters
# Full-length episodes — NO truncation. The harvest phase (steps 50+, where paid
# offers dominate) is exactly where the large-k deficit lives and must dominate
# the CE signal (user-specified 2026-08-08).
# Per-epoch subsampling: 2 of 5000 trajectories, resampled each epoch.
# TIMING MEASURED 2026-08-08: epoch-0 with 300 samples/318 avg-steps took 5.7h
#   → actual per-step cost ≈ 215ms (env+GNN forward+backward, MPS n=200-440).
# FIX: PER_EPOCH_SAMPLE=2 → total Phase-1 gradient steps = 150×2×318 ≈ 95k
#   (same as epoch-0 total); spread across 150 optimizer calls (better SGD convergence).
# Expected Phase-1 wall time: 5.7h. Phase-2 (100ep × 5 rollouts, forward-only per step):
#   ~2-5h. Total ≈ 8-11h; finishes by ~06:00 overnight.
# TEACHER EQUIVALENCE VERIFIED 2026-08-08: extractor diff=0.00e+00 on all 6 seed pairs.
P1_EPOCHS        = 150
P1_LR            = 1e-4
P1_CE_W          = 1.0
P1_MSE_W         = 0.3
PER_EPOCH_SAMPLE = 2     # trajectories per epoch; total steps = 150×2×318 ≈ 95k

# Phase 2 hyper-parameters — main training; more epochs since Phase 1 is short
P2_EPOCHS      = 100
P2_LR          = 1e-5
P2_ENTROPY     = 0.01
P2_GRAD_CLIP   = 1.0
P2_WELFORD_FLOOR = 1.0   # SENTINEL: floor=1.0, never 1e-8
P2_K_MIN       = 16
P2_K_MAX       = 40
P2_EPISODES_PER_EPOCH = 5  # per-epoch rollouts; × 100 epochs = 500 total
BUCKETS        = {0: (16, 20), 1: (21, 30), 2: (31, 40)}

os.makedirs("results/checkpoints", exist_ok=True)
os.makedirs("results/logs",        exist_ok=True)


# ── Welford online mean/variance ───────────────────────────────────────────────

class WelfordBucket:
    """Online Welford running mean/variance for reward normalisation."""
    def __init__(self, floor: float = 1.0):
        assert floor >= 0.1, f"std_floor={floor} is suspiciously small; must be >= 0.1"
        self.floor = floor
        self.n     = 0
        self.mean  = 0.0
        self.M2    = 0.0

    def update(self, x: float) -> None:
        self.n    += 1
        delta      = x - self.mean
        self.mean += delta / self.n
        self.M2   += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return self.floor
        return max(self.floor, (self.M2 / (self.n - 1)) ** 0.5)

    def normalise(self, x: float) -> float:
        return (x - self.mean) / self.std


def _bucket_id(k: int) -> int:
    for bid, (lo, hi) in BUCKETS.items():
        if lo <= k <= hi:
            return bid
    return 2  # above 40 → highest bucket


# ── Model helpers ─────────────────────────────────────────────────────────────

def _build_model():
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)


def _load_warmstart(device) -> SequentialJointPolicy:
    """Load ep200 unified checkpoint; verify sha1 prefix."""
    sha1 = hashlib.sha1(open(WARM_CKPT, "rb").read()).hexdigest()
    if not sha1.startswith(WARM_SHA1):
        raise RuntimeError(
            f"Warm-start sha1 mismatch: expected prefix {WARM_SHA1}, got {sha1[:8]}. "
            "Wrong file or corrupted checkpoint."
        )
    model = _build_model()
    model.load_state_dict(torch.load(WARM_CKPT, map_location=device), strict=True)
    print(f"  Warm-start loaded: {WARM_CKPT}  sha1={sha1[:8]}  ✓")
    return model


def _edge_index(graph, device) -> torch.Tensor:
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


# ── Trajectory helpers ────────────────────────────────────────────────────────

def _load_traj(graph_hash: str, k: int, seed: int) -> List[dict]:
    path = os.path.join(CACHE_DIR, f"{graph_hash}_k{k}_s{seed}.pkl")
    if not os.path.exists(path):
        return []
    return pickle.load(open(path, "rb"))


def _graph_hash(graph) -> str:
    sig = f"{sorted(graph.nodes())}|{sorted(graph.edges())}"
    return hashlib.md5(sig.encode()).hexdigest()[:8]


# ── Phase 1: Imitation ────────────────────────────────────────────────────────

def phase1_imitation(model, graphs, device) -> List[float]:
    """Full-episode imitation (P1_EPOCHS epochs, PER_EPOCH_SAMPLE trajectories/epoch).

    No MAX_TRAJ_STEPS truncation — full episode replayed so the harvest phase
    (paid offers, steps 50+) dominates the CE signal.
    Each epoch: PER_EPOCH_SAMPLE trajectories sampled uniformly from 5000 cached.
    """
    print("\n" + "=" * 60)
    print(f"Phase 1 — Imitation ({P1_EPOCHS} epochs, {PER_EPOCH_SAMPLE} samples/epoch, full episodes)")
    print("=" * 60)

    optimizer  = optim.Adam(model.parameters(), lr=P1_LR)
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_fn     = nn.MSELoss()
    ce_curve   = []

    # Precompute graph structures indexed by graph hash
    graph_meta = {}
    for (n, graph) in graphs:
        sf    = compute_static_features(graph)
        fc    = build_graph_feature_cache(graph, sf)
        ei    = _edge_index(graph, device)
        gh    = _graph_hash(graph)
        n_val = graph.number_of_nodes()
        graph_meta[gh] = (n, graph, fc, ei, n_val)

    # Build full list of (gh, k, seed) tuples available in cache
    all_keys = []
    for gh, (n, graph, fc, ei, n_val) in graph_meta.items():
        for k in K_LIST_TRAIN:
            for seed in range(200):  # N_EPISODES=200 per (graph,k)
                path = os.path.join(CACHE_DIR, f"{gh}_k{k}_s{seed}.pkl")
                if os.path.exists(path):
                    all_keys.append((gh, k, seed))
    print(f"  Total trajectories available: {len(all_keys)}")

    rng_p1 = random.Random(77)

    for epoch in range(P1_EPOCHS):
        epoch_loss = 0.0
        n_steps    = 0
        model.train()

        # Sample PER_EPOCH_SAMPLE trajectories uniformly, shuffle each epoch
        sample_keys = rng_p1.sample(all_keys, min(PER_EPOCH_SAMPLE, len(all_keys)))

        for (gh, k, seed) in sample_keys:
            traj = _load_traj(gh, k, seed)
            if not traj:
                continue
            if gh not in graph_meta:
                continue

            n, graph, fc, ei, n_val = graph_meta[gh]
            B0 = k * C

            env = BudgetRevenueEnv(graph, BudgetEnvConfig(
                budget_B=B0, production_cost=C,
                seed=seed, weight_high=WEIGHT_HIGH,
            ))
            env.reset()
            model.reset_episode(device)

            for step in traj:  # full episode — NO truncation
                if env._check_bankrupt():
                    break
                node_exp = step["node_idx"]
                disc_exp = step["discount"]

                feats = compute_budget_node_features_fast(
                    fc, env.S, env.offered, env.t, k=n_val, env=env,
                )
                x    = torch.tensor(feats, dtype=torch.float32, device=device)
                mask = torch.zeros(n_val, dtype=torch.bool, device=device)
                for idx in env.available_nodes:
                    mask[idx] = True

                if mask.sum() == 0:
                    break

                scores, h_emb, ctx, _ = model(x, ei, mask)

                # CE: expert node local index in available subset
                avail_idx = mask.nonzero(as_tuple=True)[0]
                local_m   = (avail_idx == node_exp).nonzero(as_tuple=True)[0]
                if len(local_m) == 0:
                    break
                local_tgt     = local_m[0:1]
                masked_logits = scores[mask]
                loss_ce = ce_loss_fn(masked_logits.unsqueeze(0), local_tgt)

                # MSE: Beta mean vs expert discount
                comb     = torch.cat([h_emb[node_exp], ctx])
                dist     = model.get_discount_distribution(comb)
                disc_t   = torch.tensor([disc_exp], dtype=torch.float32, device=device)
                loss_mse = mse_fn(dist.mean.unsqueeze(0), disc_t)

                loss = P1_CE_W * loss_ce + P1_MSE_W * loss_mse
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_steps    += 1

                # Advance env with expert action (replay)
                env.step(node_exp, disc_exp)
                model.update_sequence_state(
                    disc_exp, step["accepted"],
                    step["price"] if step["accepted"] else 0.0,
                )

        mean_loss = epoch_loss / max(n_steps, 1)
        ce_curve.append(mean_loss)
        if epoch % 10 == 0 or epoch == P1_EPOCHS - 1:
            print(f"  epoch {epoch:3d}/{P1_EPOCHS}  loss={mean_loss:.4f}  steps={n_steps}")

    return ce_curve


# ── Phase 2: REINFORCE ────────────────────────────────────────────────────────

@torch.no_grad()
def _rollout_episode(model, graph, fc, ei, n_val, k, seed, device) -> float:
    """Run one greedy-eval episode, return total revenue."""
    B0  = k * C
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C,
                          seed=seed, weight_high=WEIGHT_HIGH)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    model.reset_episode(device)

    revenue = 0.0
    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(
            fc, env.S, env.offered, env.t, k=n_val, env=env,
        )
        x    = torch.tensor(feats, dtype=torch.float32, device=device)
        mask = torch.zeros(n_val, dtype=torch.bool, device=device)
        for idx in env.available_nodes:
            mask[idx] = True

        if mask.sum() == 0:
            break

        scores, h_emb, ctx, _ = model(x, ei, mask)
        ni   = int(scores.argmax().item())
        comb = torch.cat([h_emb[ni], ctx])
        d    = float(model.get_discount_distribution(comb).mean.item())

        # SKIP enforcement
        ev = env._estimate_valuation(env.nodes[ni])
        p  = ev * (1.0 - d)
        if env.B - C + p < -1e-9:
            env.offered.add(env.nodes[ni])
            env.t += 1
            env.budget_history.append(env.B)
            model.update_sequence_state(d, False, 0.0)
            continue

        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            revenue += info["offered_price"]
        model.update_sequence_state(
            d, info["accepted"],
            info["offered_price"] if info["accepted"] else 0.0,
        )
        if done:
            break
    return revenue


def _rollout_with_grad(model, graph, fc, ei, n_val, k, seed, device):
    """Policy-gradient rollout — returns (revenue, log_prob_sum, entropy_sum)."""
    B0  = k * C
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C,
                          seed=seed, weight_high=WEIGHT_HIGH)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    model.reset_episode(device)

    revenue    = 0.0
    log_probs  = []
    entropies  = []

    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(
            fc, env.S, env.offered, env.t, k=n_val, env=env,
        )
        x    = torch.tensor(feats, dtype=torch.float32, device=device)
        mask = torch.zeros(n_val, dtype=torch.bool, device=device)
        for idx in env.available_nodes:
            mask[idx] = True

        if mask.sum() == 0:
            break

        scores, h_emb, ctx, _ = model(x, ei, mask)

        # Categorical over available nodes
        logits   = scores[mask]
        cat_dist = torch.distributions.Categorical(logits=logits)
        local_a  = cat_dist.sample()
        ni       = int(mask.nonzero(as_tuple=True)[0][local_a].item())
        log_probs.append(cat_dist.log_prob(local_a))
        entropies.append(cat_dist.entropy())

        # Beta discount
        comb     = torch.cat([h_emb[ni], ctx])
        beta_dist = model.get_discount_distribution(comb)
        d_t       = beta_dist.rsample()
        d         = float(d_t.item())
        log_probs.append(beta_dist.log_prob(d_t))
        entropies.append(beta_dist.entropy())

        # SKIP enforcement (greedy eval: if infeasible, skip)
        ev = env._estimate_valuation(env.nodes[ni])
        p  = ev * (1.0 - d)
        if env.B - C + p < -1e-9:
            env.offered.add(env.nodes[ni])
            env.t += 1
            env.budget_history.append(env.B)
            model.update_sequence_state(d, False, 0.0)
            continue

        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            revenue += info["offered_price"]
        model.update_sequence_state(
            d, info["accepted"],
            info["offered_price"] if info["accepted"] else 0.0,
        )
        if done:
            break

    log_prob_sum = torch.stack(log_probs).sum() if log_probs else torch.tensor(0.0, device=device)
    entropy_sum  = torch.stack(entropies).sum() if entropies else torch.tensor(0.0, device=device)
    return revenue, log_prob_sum, entropy_sum


def phase2_reinforce(model, graphs, device) -> Tuple[List, int]:
    """100-epoch REINFORCE with per-bucket Welford baseline."""
    print("\n" + "=" * 60)
    print(f"Phase 2 — REINFORCE ({P2_EPOCHS} epochs, k~log-uniform [16,40])")
    print("=" * 60)
    print(f"  Welford std floor: {P2_WELFORD_FLOOR}  (sentinel: must be 1.0, not 1e-8)")
    assert P2_WELFORD_FLOOR == 1.0, f"SENTINEL VIOLATED: floor={P2_WELFORD_FLOOR}"

    optimizer = optim.Adam(model.parameters(), lr=P2_LR)

    # Per-bucket Welford baselines
    welfords: Dict[int, WelfordBucket] = {
        bid: WelfordBucket(floor=P2_WELFORD_FLOOR) for bid in BUCKETS
    }

    # Precompute graph structures
    graph_meta = []
    for (n, graph) in graphs:
        sf    = compute_static_features(graph)
        fc    = build_graph_feature_cache(graph, sf)
        ei    = _edge_index(graph, device)
        n_val = graph.number_of_nodes()
        graph_meta.append((n, graph, fc, ei, n_val))

    bucket_rewards_history = []  # per epoch: {bucket_id: [rewards]}
    best_min_norm_rew      = -float("inf")
    best_epoch             = -1
    best_state             = None

    rng = random.Random(99)

    for epoch in range(P2_EPOCHS):
        model.train()
        epoch_rewards = {bid: [] for bid in BUCKETS}
        epoch_loss    = 0.0

        for ep_i in range(P2_EPISODES_PER_EPOCH):
            # k ~ log-uniform [16, 40]
            log_k = rng.uniform(np.log(P2_K_MIN), np.log(P2_K_MAX))
            k     = max(P2_K_MIN, min(P2_K_MAX, int(round(np.exp(log_k)))))
            bid   = _bucket_id(k)

            # Random graph from training set
            n, graph, fc, ei, n_val = rng.choice(graph_meta)
            seed = rng.randint(0, 999)

            revenue, lp_sum, ent_sum = _rollout_with_grad(
                model, graph, fc, ei, n_val, k, seed, device,
            )

            # Update Welford baseline AFTER computing advantage
            advantage = welfords[bid].normalise(revenue)
            welfords[bid].update(revenue)

            loss = -advantage * lp_sum - P2_ENTROPY * ent_sum
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), P2_GRAD_CLIP)
            optimizer.step()

            epoch_rewards[bid].append(revenue)
            epoch_loss += loss.item()

        # Per-bucket mean normalised reward
        bucket_means = {}
        for bid, revs in epoch_rewards.items():
            if revs:
                wf  = welfords[bid]
                bucket_means[bid] = float(np.mean(
                    [(r - wf.mean) / wf.std for r in revs]
                ))
            else:
                bucket_means[bid] = 0.0

        min_norm_rew = min(bucket_means.values())
        bucket_rewards_history.append({
            "epoch": epoch,
            "bucket_means": {str(k): v for k, v in bucket_means.items()},
            "min_norm_rew": min_norm_rew,
        })

        # Best checkpoint = max min_norm_rew over epochs
        if min_norm_rew > best_min_norm_rew:
            best_min_norm_rew = min_norm_rew
            best_epoch        = epoch
            best_state        = {k: v.cpu() if hasattr(v,'cpu') else v
                                 for k, v in model.state_dict().items()}

        bm_str = "  ".join(f"b{bid}={bucket_means[bid]:+.3f}" for bid in sorted(BUCKETS))
        print(f"  epoch {epoch:3d}/{P2_EPOCHS}  {bm_str}  min={min_norm_rew:+.3f}"
              + ("  ★" if epoch == best_epoch else ""))

    print(f"\n  Best epoch: {best_epoch}  min_norm_rew={best_min_norm_rew:.3f}")
    return bucket_rewards_history, best_epoch, best_state


# ── Checkpoint helpers ────────────────────────────────────────────────────────

def _save_checkpoint(state_dict, path: str) -> str:
    """Save state_dict and return sha256 hex."""
    import io
    buf = io.BytesIO()
    torch.save(state_dict, buf)
    raw = buf.getvalue()
    sha = hashlib.sha256(raw).hexdigest()
    with open(path, "wb") as f:
        f.write(raw)
    # Append to checkpoints/README.md
    readme = "results/checkpoints/README.md"
    ts     = time.strftime("%Y-%m-%d %H:%M")
    line   = f"\n| {ts} | {os.path.basename(path)} | sha256={sha[:16]}... | large-k specialist, warm-start from ep200 unified |\n"
    with open(readme, "a") as f:
        f.write(line)
    return sha


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 68)
    print("run_largek_specialist.py — Large-k budget specialist training")
    print("=" * 68)
    print(f"Warm-start: {WARM_CKPT}  (expected sha1 prefix={WARM_SHA1})")
    print(f"Output:     {CKPT_OUT}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── Step 1: Trajectory generation (may already be cached) ─────────────────
    print("Generating/verifying large-k trajectories...")
    import subprocess
    ret = subprocess.run(
        [sys.executable, "experiments/gen_largek_trajectories.py"],
        capture_output=False, text=True
    )
    if ret.returncode != 0:
        print("ERROR: trajectory generation failed.")
        sys.exit(1)

    # ── Load training graphs ───────────────────────────────────────────────────
    print("\nLoading training graphs...")
    graphs = []
    for n in GRAPH_SIZES:
        random.seed(GRAPH_SEED); np.random.seed(GRAPH_SEED)
        g = generate_forest_fire(n=n, p=0.37, pb=0.32, seed=GRAPH_SEED)
        graphs.append((n, g))
        print(f"  n={n}  edges={g.number_of_edges()}")

    # ── Load warm-start model ─────────────────────────────────────────────────
    print("\nLoading warm-start model...")
    model = _load_warmstart(device)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # ── Phase 1: Imitation ────────────────────────────────────────────────────
    ce_curve = phase1_imitation(model, graphs, device)

    # ── Intermediate Phase-1 checkpoint (in case Phase 2 crashes) ────────────
    p1_ckpt = CKPT_OUT.replace(".pt", "_p1.pt")
    print(f"\nSaving Phase-1 checkpoint: {p1_ckpt}")
    _save_checkpoint(model.state_dict(), p1_ckpt)

    # ── Phase 2: REINFORCE ────────────────────────────────────────────────────
    p2_history, best_epoch, best_state = phase2_reinforce(model, graphs, device)

    # ── Save best checkpoint ──────────────────────────────────────────────────
    print(f"\nSaving checkpoint: {CKPT_OUT}")
    sha = _save_checkpoint(best_state, CKPT_OUT)
    print(f"  sha256: {sha}")

    # ── Save training log ─────────────────────────────────────────────────────
    log = {
        "warm_start_sha1": WARM_SHA1,
        "ckpt_sha256":     sha,
        "p1_epochs":       P1_EPOCHS,
        "p1_ce_curve":     ce_curve,
        "p2_epochs":       P2_EPOCHS,
        "p2_history":      p2_history,
        "best_p2_epoch":   best_epoch,
        "wall_min":        (time.time() - t_start) / 60.0,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Training log: {LOG_OUT}")
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")
    print("TRAINING COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
