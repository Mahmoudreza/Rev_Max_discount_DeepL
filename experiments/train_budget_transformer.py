#!/usr/bin/env python3
"""train_budget_transformer.py — Budget-aware Transformer training.

EXACT same recipe as run_budget_unified_training.py; only sequence encoder
changes (EpisodeTransformerSliding in place of EpisodeLSTM).

Architecture:
  GraphSAGEEncoder(in_dim=CFG.features.dim+1, hidden_dim=H, n_layers=2)
  EpisodeTransformerSliding (from rev_gnn_transformer_300ep.yaml)
  TransformerJointPolicy

Feature 21: B_t / B_MAX  where B_MAX = 40*c = 12.0
  B_MAX confirmed from run_budget_unified_training.py line ~55:
  `B_MAX = 40 * C` — same constant, same normalisation.
  At κ=40: B_0=12.0, budget_col=1.0. At κ=5: B_0=1.5, budget_col=0.125.

Init: rev_gnn_transformer.pt (sha c24215b8), input_proj extended
      in_dim → in_dim+1 with ZERO-INITIALIZED extra column.

Phase 1 (300 epochs): Mixed-expert imitation, CE + 0.3*MSE
  ONE gradient step per (graph, k, seed) trajectory — ~350 steps/epoch.
Phase 2 (200 epochs): REINFORCE, k log-uniform, per-bucket Welford,
  entropy=0.01, clip=1.0, STD_FLOOR=1.0.
  Best checkpoint: max(min_bucket_normalised_advantage) across epochs.

Output: results/checkpoints/transf_budget_s{SEED}_*.pt

Usage (3 seeds in parallel on 3 GPUs):
  for S in 0 1 2; do
    nohup venv/bin/python3 -u experiments/train_budget_transformer.py \\
      --seed $S --device cuda:$S \\
      > /tmp/transf_budget_s${S}.log 2>&1 &
    echo "Seed $S PID=$!"
  done
"""
from __future__ import annotations

import argparse, hashlib, math, os, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn as nn

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.utils.helpers import set_seed, load_config_with_base
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.models.policies.transformer_joint_policy import TransformerJointPolicy
from src.training.mixed_expert_trajectories import generate_budget_expert_trajectory

_ROOT       = str(Path(__file__).parent.parent)
CFG_TFM     = os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml")
BASE_CKPT   = os.path.join(_ROOT, "results/checkpoints/rev_gnn_transformer.pt")
BASE_SHA    = "c24215b8"
CKPT_DIR    = os.path.join(_ROOT, "results/checkpoints")
README      = os.path.join(CKPT_DIR, "README.md")

C           = 0.3
B_MAX       = 40 * C        # 12.0 — same as run_budget_unified_training.py
FF_P        = 0.37
FF_PB       = 0.32
TRAIN_SIZES = [200, 260, 320, 380, 440]
W_HIGH      = 2.0
N_MC        = 200

PH1_EPOCHS  = 300
PH1_LR      = 1e-4
PRICE_ALPHA = 0.3
K_P1        = [1, 3, 5, 10, 15, 25, 40]   # per-episode k values for Phase 1
N_SEEDS_P1  = 10

PH2_EPOCHS  = 200
PH2_LR      = 5e-5
ENTROPY     = 0.01
GRAD_CLIP   = 1.0
STD_FLOOR   = 1.0   # SENTINEL: must be 1.0

BUCKETS = [(1, 2), (3, 5), (6, 10), (11, 20), (21, 40)]

def _bucket(k: int) -> int:
    for i, (lo, hi) in enumerate(BUCKETS):
        if lo <= k <= hi:
            return i
    return len(BUCKETS) - 1


# ── Welford per-bucket ────────────────────────────────────────────────────────

class WelfordBucket:
    def __init__(self):
        self.n = 0; self.m1 = 0.0; self.m2 = 0.0

    def update(self, x: float):
        self.n += 1
        d = x - self.m1
        self.m1 += d / self.n
        self.m2 += d * (x - self.m1)

    @property
    def mean(self) -> float:
        return self.m1

    @property
    def std(self) -> float:
        return max(math.sqrt(self.m2 / max(self.n - 1, 1)), STD_FLOOR)

    def normalise(self, x: float) -> float:
        return (x - self.mean) / self.std


# ── Feature helpers ───────────────────────────────────────────────────────────

def _features(cache, env: BudgetRevenueEnv, k: int) -> np.ndarray:
    """21-dim: 20 standard features + B_t / B_MAX."""
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    bcol = np.full((cache["n"], 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)

def _edge_index(G, device) -> torch.Tensor:
    edges = list(G.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    m = {v: i for i, v in enumerate(G.nodes())}
    src = [m[u] for u, _ in edges] + [m[v] for _, v in edges]
    dst = [m[v] for _, v in edges] + [m[u] for u, _ in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)

def _avail_mask(env: BudgetRevenueEnv, n: int, device) -> torch.Tensor:
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes:
        mask[idx] = True
    return mask


# ── Load and extend checkpoint ────────────────────────────────────────────────

def _load_and_extend(ckpt_path: str, device):
    """Load transformer checkpoint; extend input_proj by 1 dim (zero-init)."""
    sha = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()
    assert sha.startswith(BASE_SHA), f"ABORT: sha={sha[:8]} expected {BASE_SHA}"
    print(f"[init] base sha verified: {sha[:16]}", flush=True)

    cfg_t  = load_config_with_base(CFG_TFM)
    in_dim = int(cfg_t.features.dim)
    H      = int(cfg_t.encoder.hidden_dim)
    NL     = int(cfg_t.encoder.n_layers)
    DO     = float(cfg_t.encoder.dropout)

    enc_new = GraphSAGEEncoder(in_dim + 1, H, NL, DO)
    tfm     = EpisodeTransformerSliding.from_config(cfg_t.transformer)
    pol     = TransformerJointPolicy(enc_new, tfm,
                                     gnn_dim=H, context_dim=tfm.context_dim).to(device)

    sd_old = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd_old, dict) and "state_dict" in sd_old:
        sd_old = sd_old["state_dict"]

    sd_new = pol.state_dict()
    n_copied = 0
    for k, v in sd_old.items():
        if k in sd_new and k != "encoder.input_proj.weight":
            sd_new[k] = v.clone(); n_copied += 1

    old_w = sd_old["encoder.input_proj.weight"]   # (H, in_dim)
    new_w = sd_new["encoder.input_proj.weight"]   # (H, in_dim+1)
    new_w[:, :in_dim] = old_w
    new_w[:, in_dim]  = 0.0                        # zero-init: ignored at start
    sd_new["encoder.input_proj.weight"] = new_w
    pol.load_state_dict(sd_new, strict=True)

    n_params = sum(p.numel() for p in pol.parameters())
    print(f"[init] {in_dim}→{in_dim+1}-dim transformer built  params={n_params:,}  "
          f"copied={n_copied+1} tensors", flush=True)
    return pol


# ── Trajectory cache ──────────────────────────────────────────────────────────

def _build_traj_cache(graphs, seed: int) -> dict:
    """Build per-(graph,k,seed) expert trajectory cache. Reuse if key matches."""
    import pickle
    cache_path = os.path.join(_ROOT, f"results/logs/traj_cache_budget_s{seed}.pkl")
    key_str    = str(("budget_expert", "monotone", "flat", W_HIGH,
                       K_P1, N_SEEDS_P1,
                       [g.number_of_nodes() for g in graphs]))
    if os.path.exists(cache_path):
        try:
            saved_key, tc = pickle.load(open(cache_path, "rb"))
            if saved_key == key_str:
                print(f"[traj] Cache HIT  ({len(tc)} trajectories)", flush=True)
                return tc
        except Exception:
            pass

    print("[traj] Cache MISS — building trajectories...", flush=True)
    tc = {}
    for gi, G in enumerate(graphs):
        for k in K_P1:
            for s in range(N_SEEDS_P1):
                try:
                    traj = generate_budget_expert_trajectory(G, k, c=C, seed=s)
                    if traj:
                        tc[(gi, k, s)] = traj
                except Exception as e:
                    print(f"[traj] WARN gi={gi} k={k} s={s}: {e}", flush=True)
        print(f"[traj] graph {gi+1}/{len(graphs)} done  hits={sum(1 for (gi2,_,_) in tc if gi2==gi)}", flush=True)
    pickle.dump((key_str, tc), open(cache_path, "wb"))
    print(f"[traj] Saved {len(tc)} trajectories → {cache_path}", flush=True)
    return tc


# ── Phase 1: Imitation ────────────────────────────────────────────────────────

def phase1(pol, graphs, traj_cache: dict, device, seed: int, save_prefix: str):
    """300 epochs, CE + PRICE_ALPHA*MSE.
    One gradient step per (graph, k, env-seed) trajectory.
    ~len(graphs) × len(K_P1) × N_SEEDS_P1 ≈ 350 steps/epoch."""
    opt = torch.optim.Adam(pol.parameters(), lr=PH1_LR, weight_decay=1e-5)
    print(f"[P1-s{seed}] {PH1_EPOCHS} epochs  "
          f"~{len(graphs)*len(K_P1)*N_SEEDS_P1} grad steps/epoch", flush=True)

    for ep in range(1, PH1_EPOCHS + 1):
        epoch_loss = 0.0; epoch_steps = 0

        for gi, G in enumerate(graphs):
            cache = build_graph_feature_cache(G, compute_static_features(G))
            ei_t  = _edge_index(G, device)
            nodes = list(G.nodes()); n = len(nodes)
            nmap  = {v: i for i, v in enumerate(nodes)}
            k_order = list(K_P1); np.random.shuffle(k_order)   # local copy

            for k in k_order:
                B0 = k * C
                for s in range(N_SEEDS_P1):
                    traj = traj_cache.get((gi, k, s))
                    if not traj:
                        continue
                    pol.reset_episode(device)
                    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=s)
                    env = BudgetRevenueEnv(G, cfg); env.reset()
                    step_losses = []

                    for step in traj:
                        expert_idx  = step["node_idx"]
                        expert_disc = step["discount"]
                        if expert_idx not in env.available_nodes:
                            break
                        feats = _features(cache, env, k)
                        x  = torch.tensor(feats, dtype=torch.float32, device=device)
                        av = _avail_mask(env, n, device)
                        if not av.any():
                            break
                        scores, h, ctx, _ = pol.forward(x, ei_t, av)
                        safe  = scores.clone(); safe[~av] = -1e9
                        ce    = -torch.log_softmax(safe, dim=-1)[expert_idx]
                        pd    = pol.get_discount_distribution(
                            torch.cat([h[expert_idx], ctx]))
                        mse   = (pd.mean - torch.tensor(expert_disc, dtype=torch.float32,
                                                         device=device)) ** 2
                        step_losses.append(ce + PRICE_ALPHA * mse)
                        pol.update_sequence_state(
                            expert_disc, step["accepted"], step["price"])
                        _, _, done, _ = env.step(expert_idx, expert_disc)
                        if done:
                            break

                    if step_losses:
                        traj_loss = torch.stack(step_losses).mean()
                        opt.zero_grad()
                        traj_loss.backward()
                        nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
                        opt.step()
                        epoch_loss  += traj_loss.item()
                        epoch_steps += 1

        if ep % 50 == 0:
            avg = epoch_loss / max(epoch_steps, 1)
            print(f"[P1-s{seed}] ep={ep}/{PH1_EPOCHS}  "
                  f"avg_loss={avg:.4f}  steps={epoch_steps}", flush=True)
        if ep % 100 == 0:
            sp = save_prefix.replace("_ep.pt", f"_p1_ep{ep}.pt")
            torch.save(pol.state_dict(), sp); _append_readme(sp)
            print(f"[P1-s{seed}] saved {sp}", flush=True)

    return pol


# ── Phase 2: REINFORCE ────────────────────────────────────────────────────────

def _rollout(pol, G, cache, ei_t, B0: float, k: int, s: int, device):
    """One stochastic episode. Returns (revenue, log_probs, entropies)."""
    set_seed(s)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=s,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    pol.reset_episode(device)
    revenue = 0.0; lps = []; ents = []

    while env.available_nodes and not env._check_bankrupt():
        feats = _features(cache, env, k)
        x  = torch.tensor(feats, dtype=torch.float32, device=device)
        av = _avail_mask(env, n, device)
        if not av.any():
            break
        with torch.enable_grad():
            scores, h, ctx, _ = pol.forward(x, ei_t, av)
            safe  = scores.clone(); safe[~av] = -1e9
            probs = torch.softmax(safe, dim=-1)
            dist  = torch.distributions.Categorical(probs=probs[av])
            si    = dist.sample()
            glob  = av.nonzero(as_tuple=True)[0][si]
            lps.append(dist.log_prob(si))
            ents.append(dist.entropy())
            disc  = float(pol.get_discount_distribution(
                torch.cat([h[int(glob)], ctx])).mean.clamp(1e-4, 1 - 1e-4).detach())
        v = nodes[int(glob)]
        _, r, done, _ = env.step(env.node_to_idx[v], disc)
        revenue += r
        pol.update_sequence_state(disc, r > 0, r)
        if done:
            break

    return revenue, lps, ents


def phase2(pol, graphs, device, seed: int, save_prefix: str):
    """200 epochs REINFORCE, per-bucket Welford, k log-uniform [1,40]."""
    assert STD_FLOOR == 1.0, f"SENTINEL: STD_FLOOR={STD_FLOOR}"
    opt      = torch.optim.Adam(pol.parameters(), lr=PH2_LR, weight_decay=1e-5)
    welfords = [WelfordBucket() for _ in BUCKETS]
    rng      = np.random.default_rng(seed=seed + 123)
    best_min = -1e9; best_sd = None; best_ep = 0

    print(f"\n[P2-s{seed}] {PH2_EPOCHS} epochs  lr={PH2_LR}  "
          f"entropy={ENTROPY}  STD_FLOOR={STD_FLOOR}", flush=True)

    caches = {gi: build_graph_feature_cache(G, compute_static_features(G))
              for gi, G in enumerate(graphs)}
    eis    = {gi: _edge_index(G, device) for gi, G in enumerate(graphs)}

    for ep in range(1, PH2_EPOCHS + 1):
        adv_lp_ent = []         # list of (adv, lps, ents) per graph-episode
        bucket_revs = [[] for _ in BUCKETS]

        for gi, G in enumerate(graphs):
            k   = int(np.exp(rng.uniform(math.log(1), math.log(40))))
            k   = max(1, min(40, k))
            B0  = k * C
            s   = int(rng.integers(0, 1000))
            rev, lps, ents = _rollout(pol, G, caches[gi], eis[gi], B0, k, s, device)
            bid = _bucket(k)
            adv = welfords[bid].normalise(rev)
            adv_lp_ent.append((adv, lps, ents))
            bucket_revs[bid].append(rev)

        # Update Welford AFTER using it for advantages (no self-inflation)
        for bid, revs in enumerate(bucket_revs):
            for r in revs:
                welfords[bid].update(r)

        if not adv_lp_ent:
            continue

        pol_loss = -sum(adv * sum(lp for lp in lps)
                        for adv, lps, ents in adv_lp_ent) / len(adv_lp_ent)
        ent_loss = -ENTROPY * sum(e for _, _, ents in adv_lp_ent
                                  for e in ents) / max(len(adv_lp_ent), 1)
        loss = pol_loss + ent_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), GRAD_CLIP)
        opt.step()

        # Best checkpoint: max(min normalised advantage across non-empty buckets)
        bucket_nadv = []
        for bid, revs in enumerate(bucket_revs):
            if revs and welfords[bid].n >= 2:
                vals = [(r - welfords[bid].mean) / welfords[bid].std for r in revs]
                bucket_nadv.append(float(np.mean(vals)))
        if bucket_nadv:
            min_adv = min(bucket_nadv)
            if min_adv > best_min:
                best_min = min_adv
                best_sd  = {k: v.clone() for k, v in pol.state_dict().items()}
                best_ep  = ep

        if ep % 20 == 0:
            bra = [f"{np.mean(r):.1f}" if r else "—" for r in bucket_revs]
            print(f"[P2-s{seed}] ep={ep}/{PH2_EPOCHS}  "
                  f"min_adv={bucket_nadv and min(bucket_nadv) or 0:.3f}  "
                  f"bucket_rev={bra}", flush=True)
            sp = save_prefix.replace("_ep.pt", f"_ep{ep}.pt")
            torch.save(pol.state_dict(), sp); _append_readme(sp)

    if best_sd:
        pol.load_state_dict(best_sd)
        print(f"[P2-s{seed}] Restored best ep={best_ep}  min_adv={best_min:.3f}", flush=True)
    return pol, best_ep


# ── README helpers ────────────────────────────────────────────────────────────

def _sha8(path: str) -> str:
    try:
        return hashlib.sha256(open(path, "rb").read()).hexdigest()[:8]
    except Exception:
        return "????????"

def _append_readme(path: str):
    sha  = _sha8(path)
    line = f"| `{os.path.basename(path)}` | budget-transformer | sha={sha} |\n"
    try:
        with open(README, "a") as f:
            f.write(line)
    except Exception:
        pass


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--device",  default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_p1", action="store_true",
                    help="Skip Phase 1, load an existing P1 checkpoint via --p1_ckpt")
    ap.add_argument("--p1_ckpt", default="",
                    help="Path to a Phase-1 checkpoint (used with --skip_p1)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"=== Budget Transformer Training  seed={args.seed}  device={device} ===")
    print(f"Base: {BASE_CKPT}  sha={BASE_SHA}")
    print(f"B_MAX={B_MAX}  C={C}  STD_FLOOR={STD_FLOOR}  N_MC={N_MC}")
    print(f"P1: {PH1_EPOCHS} ep  k∈{K_P1}  {N_SEEDS_P1} seeds/k")
    print(f"P2: {PH2_EPOCHS} ep  k log-uniform [1,40]  entropy={ENTROPY}", flush=True)

    graphs = [generate_forest_fire(n, FF_P, FF_PB, seed=args.seed * 100 + i)
              for i, n in enumerate(TRAIN_SIZES)]
    print(f"Training graphs: {[g.number_of_nodes() for g in graphs]}", flush=True)

    save_prefix = os.path.join(CKPT_DIR, f"transf_budget_s{args.seed}_ep.pt")

    if args.skip_p1 and args.p1_ckpt and os.path.exists(args.p1_ckpt):
        print(f"[main] Skipping Phase 1 — loading {args.p1_ckpt}", flush=True)
        cfg_t  = load_config_with_base(CFG_TFM)
        in_dim = int(cfg_t.features.dim) + 1
        H, NL, DO = (int(cfg_t.encoder.hidden_dim),
                     int(cfg_t.encoder.n_layers),
                     float(cfg_t.encoder.dropout))
        enc = GraphSAGEEncoder(in_dim, H, NL, DO)
        tfm = EpisodeTransformerSliding.from_config(cfg_t.transformer)
        pol = TransformerJointPolicy(enc, tfm,
                                     gnn_dim=H, context_dim=tfm.context_dim).to(device)
        pol.load_state_dict(
            torch.load(args.p1_ckpt, map_location=device, weights_only=True))
        print(f"[main] Loaded P1 checkpoint — going to Phase 2", flush=True)
    else:
        pol = _load_and_extend(BASE_CKPT, device)
        traj_cache = _build_traj_cache(graphs, args.seed)
        pol = phase1(pol, graphs, traj_cache, device, args.seed, save_prefix)
        p1_path = save_prefix.replace("_ep.pt", f"_p1_ep{PH1_EPOCHS}.pt")
        torch.save(pol.state_dict(), p1_path)
        _append_readme(p1_path)
        print(f"[main] Phase 1 done → {p1_path}  sha={_sha8(p1_path)}", flush=True)

    pol, best_ep = phase2(pol, graphs, device, args.seed, save_prefix)
    best_path = save_prefix.replace("_ep.pt", "_best.pt")
    torch.save(pol.state_dict(), best_path)
    _append_readme(best_path)
    sha = _sha8(best_path)
    print(f"\n[main] Final → {best_path}  sha={sha}", flush=True)
    print(sha, flush=True)

    import subprocess
    subprocess.run(["git", "add", "-f", best_path], cwd=_ROOT)
    subprocess.run(["git", "add", "-f", README],    cwd=_ROOT)
    r = subprocess.run(
        ["git", "commit", "-m", f"transf_budget_s{args.seed}_best sha={sha}"],
        cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                       capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h, flush=True)


if __name__ == "__main__":
    main()
