"""experiments/run_fair_rl_training.py

Task 3 — Phase 5 Fair-RL training.
NEW FILE — does NOT modify any existing file.

Usage (chained, via shell):
  python run_fair_rl_training.py --lam 0.2 > /tmp/fair_rl_l02.log 2>&1
  python run_fair_rl_training.py --lam 0.5 > /tmp/fair_rl_l05.log 2>&1
  python run_fair_rl_training.py --lam 1.0 > /tmp/fair_rl_l10.log 2>&1

Architecture: SequentialJointPolicy warm-started from rev_gnn_lstm.pt.
Feature index 15 (0-based, 16th feature) = node group label (0/1) — active for THIS model only.
"""
from __future__ import annotations
import sys, json, os, argparse, hashlib, time, math
sys.path.insert(0, ".")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf

from src.env.sbm_generators import two_block_graph
from src.evaluation.baselines import _make_env, _compute_normalized_infl
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.encoders.sequence_models import EpisodeLSTM
from src.utils.features import compute_node_features, compute_static_features

# ── Hyperparameters ───────────────────────────────────────────────────────────
LR             = 1e-5
ENTROPY_COEFF  = 0.01
GRAD_CLIP      = 1.0
WELFORD_FLOOR  = 1.0          # std never below this value
N_EPOCHS       = 100
TRAIN_NS       = [200, 300, 400]
TRAIN_HS       = [0.7, 0.9]
FRAC_MIN       = 0.3
WARM_CKPT      = "results/checkpoints/rev_gnn_lstm.pt"
CKPT_DIR       = "results/checkpoints"
CKPT_README    = "results/checkpoints/README.md"
FEAT_DIM       = 20           # same as rev_gnn_lstm
HIDDEN         = 64

BASE_CFG_YAML = """
project:
  name: fair-rl
  seed: 0
features:
  dim: 20
encoder:
  hidden_dim: 64
  n_layers: 2
  dropout: 0.0
influence:
  model: monotone
  n_mc_samples: 10
  b: 1.0
  weight_low: 0.0
  weight_high: 2.0
reward:
  type: revenue
  gamma: 1.0
env:
  k: 400
  budget: 0.0
"""


def _shasum(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def _backup_and_register(ckpt_path: str, lam: float) -> str:
    """shasum + append to checkpoints/README.md."""
    sha = _shasum(ckpt_path)
    entry = (f"| {os.path.basename(ckpt_path)} | {sha} | "
             f"Fair-RL λ={lam}, warm-start rev_gnn_lstm.pt, "
             f"REINFORCE {N_EPOCHS} epochs, "
             f"reward=rev/n+λ*min_rho, lr={LR} |\n")
    with open(CKPT_README, "a") as f:
        f.write(entry)
    print(f"  Registered {os.path.basename(ckpt_path)} sha={sha} in {CKPT_README}")
    return sha


def _load_policy_from_warm_start() -> SequentialJointPolicy:
    """Load SequentialJointPolicy warm-started from rev_gnn_lstm.pt.

    EpisodeLSTM: graph_dim=64, lstm_hidden=64 (takes token_proj internally).
    SequentialJointPolicy: (encoder, sequence_model, gnn_dim=64, context_dim=64).
    available_mask must be bool (policy uses ~mask).
    """
    enc = GraphSAGEEncoder(in_dim=FEAT_DIM, hidden_dim=HIDDEN, n_layers=2, dropout=0.0)
    seq = EpisodeLSTM(graph_dim=HIDDEN, lstm_hidden=HIDDEN)
    policy = SequentialJointPolicy(enc, seq, gnn_dim=HIDDEN, context_dim=HIDDEN)

    state = torch.load(WARM_CKPT, map_location="cpu")
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = policy.load_state_dict(state, strict=False)
    print(f"  Warm-start loaded from {WARM_CKPT}")
    if missing:    print(f"    Missing keys:    {missing[:5]}")
    if unexpected: print(f"    Unexpected keys: {unexpected[:5]}")
    return policy


# ── Welford online stats ──────────────────────────────────────────────────────

class WelfordBaseline:
    def __init__(self, floor: float = 1.0):
        self.n    = 0
        self.mean = 0.0
        self.M2   = 0.0
        self.floor = floor

    def update(self, x: float):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self) -> float:
        if self.n < 2:
            return self.floor
        return max(self.floor, math.sqrt(self.M2 / (self.n - 1)))

    def advantage(self, x: float) -> float:
        return (x - self.mean) / self.std


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(graph, cfg, labels, policy):
    """Run one REINFORCE episode.

    Fair term uses AUC-style early coverage:
      fair_term = mean over K in {n/8, n/4, n/2, 3n/4} of min_g rho_g(K)
    Rationale: on dense graphs final adoption saturates for all methods;
    fairness lives in WHO IS SERVED EARLY (Rice-FB: sub_share_B=0 at ALL K
    under Greedy, so early coverage captures the structural disparity).

    Returns:
        rev (float): raw revenue
        fair_term (float): AUC early coverage fairness signal
        min_rho_final (float): final min_rho (for logging only)
        log_probs (Tensor): stacked log-probs of selected nodes
        entropy (Tensor): mean entropy over steps
    """
    env = _make_env(graph, cfg)
    env.reset()
    n   = env.n
    b   = float(cfg.influence.b)
    lw  = env._link_weights
    nodes = list(graph.nodes())

    edges = list(graph.edges())
    src = [e[0] for e in edges] + [e[1] for e in edges]
    dst = [e[1] for e in edges] + [e[0] for e in edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    static_feats = compute_static_features(graph)
    offered: set = set()
    log_probs_list = []
    entropies_list = []
    rev = 0.0
    nA  = int((labels == 0).sum())
    nB  = int((labels == 1).sum())
    acc_g = {0: 0, 1: 0}
    # Checkpoints for AUC fair term: steps n//8, n//4, n//2, 3n//4
    auc_ks = {n // 8, n // 4, n // 2, 3 * n // 4}
    auc_vals = []

    for _ in range(n):
        remaining = [v for v in nodes if v not in offered]
        if not remaining:
            break

        feats = compute_node_features(
            graph=graph, static_features=static_feats,
            S=frozenset(env.S), offered=frozenset(offered),
            t=len(offered), n=n, k=n, env=env,
        )
        # Group label is feature index 15 (0-based 20-dim vector) = ACTIVE for Fair-RL
        feats[:, 15] = labels.astype(np.float32)

        x = torch.tensor(feats, dtype=torch.float32)
        # available_mask: True = available, False = offered (policy uses ~mask → -inf)
        avail_mask = torch.ones(n, dtype=torch.bool)
        for v in offered:
            avail_mask[env.node_to_idx[v]] = False

        out = policy(x, edge_index, avail_mask)
        if isinstance(out, (list, tuple)):
            scores = out[0]   # (n,) logits
        else:
            scores = out

        # Softmax over available nodes only
        avail_idx   = [env.node_to_idx[v] for v in remaining]
        avail_logits = scores[avail_idx]
        probs       = torch.softmax(avail_logits, dim=0)
        dist        = torch.distributions.Categorical(probs=probs)
        local_idx   = dist.sample()
        log_p       = dist.log_prob(local_idx)
        ent         = dist.entropy()

        log_probs_list.append(log_p)
        entropies_list.append(ent)

        target   = remaining[local_idx.item()]
        node_idx = env.node_to_idx[target]
        group    = int(labels[node_idx])

        infl     = _compute_normalized_infl(graph, target, env.S, lw)
        if infl < 2.0 / 6.0:
            from src.evaluation.baselines import _rayleigh_price
            price = 0.0
        elif infl < 4.0 / 6.0:
            from src.evaluation.baselines import _rayleigh_price
            price = _rayleigh_price(2.0 / 6.0, b)
        else:
            from src.evaluation.baselines import _rayleigh_price
            price = _rayleigh_price(4.0 / 6.0, b)

        true_val = env._true_valuation(target)
        if price == 0.0 or true_val >= price:
            env.S.add(target)
            env._influence_cache = {}
            acc_g[group] += 1
            rev += price

        offered.add(target)
        env.offered.add(target)
        env.t += 1

        # Record AUC checkpoint
        t_step = len(offered)
        if t_step in auc_ks:
            rho_a = acc_g[0] / max(1, nA)
            rho_b = acc_g[1] / max(1, nB)
            auc_vals.append(min(rho_a, rho_b))

    rho_A = acc_g[0] / max(1, nA)
    rho_B = acc_g[1] / max(1, nB)
    min_rho_final = min(rho_A, rho_B)

    # AUC fair term: mean min_rho at {n/8, n/4, n/2, 3n/4}
    fair_term = sum(auc_vals) / len(auc_vals) if auc_vals else min_rho_final

    log_probs_t = torch.stack(log_probs_list) if log_probs_list else torch.tensor(0.0)
    entropy_t   = torch.stack(entropies_list).mean() if entropies_list else torch.tensor(0.0)
    return rev, fair_term, min_rho_final, log_probs_t, entropy_t


# ── Training loop ─────────────────────────────────────────────────────────────

def train(lam: float):
    tag = f"l{int(lam*10):02d}"
    ckpt_out = os.path.join(CKPT_DIR, f"rev_gnn_lstm_fair_{tag}.pt")
    log_path = f"/tmp/fair_rl_{tag}.log"

    print(f"\n{'='*70}")
    print(f"  Fair-RL training  λ={lam}  →  {ckpt_out}")
    print(f"  lr={LR}  entropy={ENTROPY_COEFF}  grad_clip={GRAD_CLIP}  epochs={N_EPOCHS}")
    print(f"  Welford floor={WELFORD_FLOOR}  graphs: n={TRAIN_NS} × h={TRAIN_HS}")
    print(f"{'='*70}\n")

    cfg = OmegaConf.create(BASE_CFG_YAML)

    policy = _load_policy_from_warm_start()
    optimizer = optim.Adam(policy.parameters(), lr=LR)
    baseline = WelfordBaseline(floor=WELFORD_FLOOR)

    # Pre-generate training graphs (fresh per epoch)
    graph_configs = [(n, h) for n in TRAIN_NS for h in TRAIN_HS]

    best_reward = -1e9
    os.makedirs(CKPT_DIR, exist_ok=True)

    for epoch in range(1, N_EPOCHS + 1):
        # Sample one graph per epoch (cycle through configs)
        n_g, h_g = graph_configs[(epoch - 1) % len(graph_configs)]
        graph, labels = two_block_graph(
            n=n_g, frac_minority=FRAC_MIN, avg_degree=5.0,
            homophily=h_g, seed=epoch,   # fresh graph each epoch
        )
        cfg_ep = OmegaConf.merge(cfg, OmegaConf.create({
            "graph": {"n_nodes": n_g}, "env": {"k": n_g},
        }))

        policy.train()
        # _run_episode returns (rev, fair_term, min_rho_final, log_probs, entropy)
        rev, fair_term, min_rho_final, log_probs, entropy = _run_episode(
            graph, cfg_ep, labels, policy)

        # Reward: revenue term + AUC-fair-coverage term
        total_reward = float(rev) / n_g + lam * fair_term
        baseline.update(total_reward)
        advantage = baseline.advantage(total_reward)

        # REINFORCE loss: -advantage * sum(log_probs) - entropy_coeff * entropy
        policy_loss = -advantage * log_probs.sum()
        loss        = policy_loss - ENTROPY_COEFF * entropy

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(policy.parameters(), GRAD_CLIP)
        optimizer.step()

        if total_reward > best_reward:
            best_reward = total_reward
            torch.save(policy.state_dict(), ckpt_out)

        if epoch <= 3 or epoch % 10 == 0 or epoch == N_EPOCHS:
            print(f"  epoch {epoch:3d}/{N_EPOCHS}  "
                  f"n={n_g} h={h_g}  "
                  f"rev={rev:.2f}  fair_auc={fair_term:.3f}  min_rho_final={min_rho_final:.3f}  "
                  f"reward={total_reward:.4f}  "
                  f"adv={advantage:.3f}  "
                  f"loss={loss.item():.4f}  "
                  f"bl={baseline.mean:.4f}±{baseline.std:.4f}",
                  flush=True)

    # Save final (might be different from best)
    final_path = ckpt_out.replace(".pt", "_final.pt")
    torch.save(policy.state_dict(), final_path)
    print(f"\n  Best checkpoint: {ckpt_out}")
    print(f"  Final checkpoint: {final_path}")

    # shasum + README
    sha = _backup_and_register(ckpt_out, lam)
    print(f"  sha={sha}  best_reward={best_reward:.4f}")

    return best_reward


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lam", type=float, required=True,
                        help="lambda for fairness regularization (0.2, 0.5, or 1.0)")
    args = parser.parse_args()
    lam = args.lam

    # Backup warm-start checkpoint entry
    ws_sha = _shasum(WARM_CKPT)
    print(f"[Fair-RL] Warm-start checkpoint: {WARM_CKPT}  sha={ws_sha}")
    print(f"[Fair-RL] λ={lam}  start={time.strftime('%H:%M:%S')}")

    best = train(lam)
    print(f"\n[Fair-RL] λ={lam} done  best_reward={best:.4f}  "
          f"end={time.strftime('%H:%M:%S')}")


if __name__ == "__main__":
    main()
