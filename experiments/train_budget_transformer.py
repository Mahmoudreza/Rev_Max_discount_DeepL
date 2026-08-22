#!/usr/bin/env python3
"""train_budget_transformer.py — Budget-aware Transformer training.

EXACT same recipe as run_budget_unified_training.py; ONLY the sequence encoder
changes (EpisodeTransformerSliding in place of EpisodeLSTM).

Architecture:
  GraphSAGEEncoder(in_dim=CFG.features.dim+1, hidden_dim=H, n_layers=2)
  EpisodeTransformerSliding (from rev_gnn_transformer_300ep.yaml)
  TransformerJointPolicy

Feature 21: B_t / B_MAX  where B_MAX = 40*c = 12.0  (same as unified LSTM)
Init: rev_gnn_transformer.pt (sha c24215b8), input_proj extended
      in_dim → in_dim+1 with ZERO-INITIALIZED extra column.

Phase 1 (300 epochs): Mixed-expert imitation, CE + 0.3*MSE
Phase 2 (200 epochs): REINFORCE, k log-uniform, per-bucket Welford,
                       entropy=0.01, clip=1.0, STD_FLOOR=1.0
                       checkpoint on max(min_bucket_normalised_advantage)

Output: results/checkpoints/transf_budget_s{SEED}_ep{EP}.pt

Usage (3 seeds in parallel on 3 GPUs):
  for S in 0 1 2; do
    nohup venv/bin/python3 -u experiments/train_budget_transformer.py \\
      --seed $S --device cuda:$S \\
      > /tmp/transf_budget_s${S}.log 2>&1 &
    echo "Seed $S PID=$!"
  done
"""
from __future__ import annotations

import argparse, hashlib, json, math, os, sys, time
from pathlib import Path
from typing import Dict, List, Tuple

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
from src.training.mixed_expert_trajectories import (
    build_trajectory_cache, generate_budget_expert_trajectory,
)

_ROOT       = str(Path(__file__).parent.parent)
CFG_TFM     = os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml")
BASE_CKPT   = os.path.join(_ROOT, "results/checkpoints/rev_gnn_transformer.pt")
BASE_SHA    = "c24215b8"
CKPT_DIR    = os.path.join(_ROOT, "results/checkpoints")
README      = os.path.join(CKPT_DIR, "README.md")

C           = 0.3
B_MAX       = 40 * C        # 12.0  — normalisation denominator (same as LSTM)
FF_P, FF_PB = 0.37, 0.32
TRAIN_SIZES = [200, 260, 320, 380, 440]
W_HIGH      = 2.0
N_MC        = 200

PH1_EPOCHS   = 300
PH1_LR       = 1e-4
PRICE_ALPHA  = 0.3
K_SAMPLES_P1 = [1, 3, 5, 10, 15, 25, 40]
N_SEEDS_P1   = 10

PH2_EPOCHS   = 200
PH2_LR       = 5e-5
ENTROPY_COEF = 0.01
GRAD_CLIP    = 1.0
STD_FLOOR    = 1.0    # SENTINEL: must be 1.0

BUCKETS = [(1,2),(3,5),(6,10),(11,20),(21,40)]

def _bucket_of(k):
    for i,(lo,hi) in enumerate(BUCKETS):
        if lo<=k<=hi: return i
    return len(BUCKETS)-1


# ── Welford ───────────────────────────────────────────────────────────────────

class WelfordBucket:
    def __init__(self):
        self.n=0; self.m1=0.0; self.m2=0.0
    def update(self, x):
        self.n+=1; d=x-self.m1; self.m1+=d/self.n
        self.m2+=d*(x-self.m1)
    @property
    def mean(self): return self.m1
    @property
    def std(self): return max(math.sqrt(self.m2/max(self.n-1,1)), STD_FLOOR)
    def normalise(self, x): return (x-self.mean)/self.std


# ── Features (21-dim) ─────────────────────────────────────────────────────────

def _features(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    n    = cache["n"]
    bcol = np.full((n,1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)

def _ei(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    m = {v:i for i,v in enumerate(G.nodes())}
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)

def _avail(env, n, device):
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for idx in env.available_nodes: mask[idx]=True
    return mask


# ── Load + extend checkpoint ──────────────────────────────────────────────────

def _load_and_extend(ckpt_path, device):
    """Load transformer checkpoint; extend input_proj by 1 dim (zero-init)."""
    sha = hashlib.sha256(open(ckpt_path,"rb").read()).hexdigest()
    assert sha.startswith(BASE_SHA), f"ABORT: sha={sha[:8]} expected {BASE_SHA}"
    print(f"[init] base sha verified: {sha[:16]}", flush=True)

    cfg = load_config_with_base(CFG_TFM)
    in_dim = int(cfg.features.dim)
    H      = int(cfg.encoder.hidden_dim)
    NL     = int(cfg.encoder.n_layers)
    DO     = float(cfg.encoder.dropout)

    # Build (in_dim+1)-dim policy fresh
    enc_new = GraphSAGEEncoder(in_dim+1, H, NL, DO)
    tfm     = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol_new = TransformerJointPolicy(enc_new, tfm,
                                      gnn_dim=H, context_dim=tfm.context_dim).to(device)

    # Load original in_dim weights
    sd_old = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd_old,dict) and 'state_dict' in sd_old: sd_old=sd_old['state_dict']

    sd_new = pol_new.state_dict()
    copied = 0
    for k,v in sd_old.items():
        if k in sd_new and k != "encoder.input_proj.weight":
            sd_new[k] = v.clone(); copied+=1

    # Extend input_proj.weight: (H, in_dim) → (H, in_dim+1) with zero column
    old_w = sd_old["encoder.input_proj.weight"]   # (H, in_dim)
    new_w = sd_new["encoder.input_proj.weight"]   # (H, in_dim+1)
    new_w[:, :in_dim] = old_w
    new_w[:, in_dim]  = 0.0
    sd_new["encoder.input_proj.weight"] = new_w
    pol_new.load_state_dict(sd_new, strict=True)
    print(f"[init] {in_dim}→{in_dim+1}-dim transformer built "
          f"({sum(p.numel() for p in pol_new.parameters())} params, "
          f"copied {copied+1} tensors)", flush=True)
    return pol_new, in_dim+1


# ── One greedy rollout (for eval/phase2) ─────────────────────────────────────

def _rollout(pol, G, cache, ei_t, B0, seed, device, k_feat=None):
    """Returns (revenue, log_probs, entropies, selected_nodes)."""
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes=list(G.nodes()); n=len(nodes)
    kf = k_feat or max(1, round(B0/C))
    pol.reset_episode(device)
    revenue=0.0; log_probs=[]; entropies=[]

    while env.available_nodes and not env._check_bankrupt():
        feats = _features(cache, env, kf)
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        av = _avail(env, n, device)
        if not av.any(): break
        with torch.enable_grad():
            scores, h, ctx, _ = pol.forward(x, ei_t, av)
            safe = scores.clone(); safe[~av]=-1e9
            probs = torch.softmax(safe, dim=-1)
            dist  = torch.distributions.Categorical(probs=probs[av])
            sel_in_av = dist.sample()
            sel_glob  = av.nonzero(as_tuple=True)[0][sel_in_av]
            sel_idx   = int(sel_glob)
            lp = dist.log_prob(sel_in_av)
            ent= dist.entropy()
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel_idx], ctx])).mean.clamp(1e-4,1-1e-4).detach())
        log_probs.append(lp); entropies.append(ent)
        v = nodes[sel_idx]
        _, r, done, _ = env.step(env.node_to_idx[v], disc)
        revenue += r
        pol.update_sequence_state(disc, r>0, r)
        if done: break

    return revenue, log_probs, entropies


# ── Phase 1: imitation ────────────────────────────────────────────────────────

def phase1(pol, graphs, traj_cache, device, seed, save_prefix):
    """300 epochs CE + PRICE_ALPHA*MSE from cached expert trajectories."""
    opt = torch.optim.Adam(pol.parameters(), lr=PH1_LR, weight_decay=1e-5)
    N   = len(graphs)
    print(f"[P1-s{seed}] {PH1_EPOCHS} epochs, {N} graphs × {len(K_SAMPLES_P1)} k-levels × {N_SEEDS_P1} seeds", flush=True)

    for ep in range(1, PH1_EPOCHS+1):
        losses=[]
        for gi,G in enumerate(graphs):
            cache = build_graph_feature_cache(G, compute_static_features(G))
            ei_t  = _ei(G, device)
            nodes = list(G.nodes()); n=len(nodes)
            np.random.shuffle(K_SAMPLES_P1 := list(K_SAMPLES_P1))
            for k in K_SAMPLES_P1:
                B0 = k * C
                for s in range(N_SEEDS_P1):
                    traj = traj_cache.get((gi,k,s))
                    if traj is None: continue
                    pol.reset_episode(device)
                    set_seed(s*997+k*13+gi*7)
                    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C,
                                          seed=s, weight_high=W_HIGH, n_mc_samples=N_MC)
                    env = BudgetRevenueEnv(G, cfg); env.reset()
                    for (t_node, t_disc, t_acc, t_price) in traj:
                        feats = _features(cache, env, k)
                        x = torch.tensor(feats, dtype=torch.float32, device=device)
                        av= _avail(env, n, device)
                        if not av.any(): break
                        scores, h, ctx, _ = pol.forward(x, ei_t, av)
                        # CE node selection
                        nmap = {v:i for i,v in enumerate(nodes)}
                        t_idx = nmap.get(t_node, 0)
                        safe  = scores.clone(); safe[~av]=-1e9
                        ce    = -torch.log_softmax(safe,dim=-1)[t_idx]
                        # MSE pricing
                        price_dist = pol.get_discount_distribution(
                            torch.cat([h[t_idx], ctx]))
                        mse = ((price_dist.mean - torch.tensor(t_disc,device=device))**2)
                        loss = ce + PRICE_ALPHA * mse
                        losses.append(loss)
                        pol.update_sequence_state(t_disc, t_acc, t_price)
                        _, _, done, _ = env.step(env.node_to_idx[t_node], t_disc)
                        if done: break

        if losses:
            total = torch.stack(losses).mean()
            opt.zero_grad(); total.backward(); nn.utils.clip_grad_norm_(pol.parameters(),1.0)
            opt.step()

        if ep % 50 == 0:
            print(f"[P1-s{seed}] ep={ep}/{PH1_EPOCHS}  loss={total.item():.4f}", flush=True)
        if ep % 100 == 0:
            sp = save_prefix.replace("_ep", f"_p1_ep{ep}")
            torch.save(pol.state_dict(), sp)
            _append_readme(sp)
            print(f"[P1-s{seed}] saved {sp}", flush=True)

    return pol


# ── Phase 2: REINFORCE ────────────────────────────────────────────────────────

def phase2(pol, graphs, device, seed, save_prefix):
    """200 epochs REINFORCE, per-bucket Welford, kappa log-uniform."""
    opt = torch.optim.Adam(pol.parameters(), lr=PH2_LR, weight_decay=1e-5)
    assert STD_FLOOR == 1.0, f"SENTINEL VIOLATED: STD_FLOOR={STD_FLOOR}"
    welfords = [WelfordBucket() for _ in BUCKETS]
    rng = np.random.default_rng(seed=seed+123)
    best_min_adv = -1e9; best_sd = None; best_ep = 0

    print(f"\n[P2-s{seed}] {PH2_EPOCHS} epochs REINFORCE, lr={PH2_LR}, "
          f"entropy={ENTROPY_COEF}, STD_FLOOR={STD_FLOOR}", flush=True)
    caches = {}
    eis    = {}
    for gi,G in enumerate(graphs):
        caches[gi] = build_graph_feature_cache(G, compute_static_features(G))
        eis[gi]    = _ei(G, device)

    for ep in range(1, PH2_EPOCHS+1):
        log_probs_all=[]; entropies_all=[]; adv_list=[]; bucket_revs=[[] for _ in BUCKETS]

        for gi,G in enumerate(graphs):
            # Sample k log-uniformly from [1,40]
            k = int(np.exp(rng.uniform(math.log(1), math.log(40))))
            k = max(1, min(40, k))
            B0= k * C
            s = int(rng.integers(0, 1000))
            rev, lps, ents = _rollout(pol, G, caches[gi], eis[gi], B0, s, device, k_feat=k)
            bid = _bucket_of(k)
            adv = welfords[bid].normalise(rev)
            log_probs_all += lps; entropies_all += ents
            adv_list.append((adv, lps, ents))
            bucket_revs[bid].append(rev)

        # Update Welford AFTER computing all advantages this epoch
        for gi,G in enumerate(graphs):
            for (adv,lps,ents) in adv_list:
                pass  # already used for computing adv
        # Actually update Welford using this epoch's revenues
        for bid, revs in enumerate(bucket_revs):
            for r in revs: welfords[bid].update(r)

        if not log_probs_all: continue
        pol_loss = -sum(adv * sum(lp for lp in lps)
                        for (adv,lps,ents) in adv_list) / max(len(adv_list),1)
        ent_loss = -ENTROPY_COEF * sum(e for (_,_,ents) in adv_list
                                       for e in ents) / max(len(adv_list),1)
        loss = pol_loss + ent_loss
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(pol.parameters(), GRAD_CLIP)
        opt.step()

        # Checkpoint selection: max(min normalised adv across buckets)
        bucket_mean_advs = []
        for bid,revs in enumerate(bucket_revs):
            if revs:
                adv_vals=[(r-welfords[bid].mean)/welfords[bid].std for r in revs]
                bucket_mean_advs.append(float(np.mean(adv_vals)))
        if bucket_mean_advs:
            min_adv = min(bucket_mean_advs)
            if min_adv > best_min_adv:
                best_min_adv = min_adv; best_sd = {k:v.clone() for k,v in pol.state_dict().items()}; best_ep=ep

        if ep % 20 == 0:
            bra = [f"{np.mean(r):.1f}" if r else "—" for r in bucket_revs]
            print(f"[P2-s{seed}] ep={ep}/{PH2_EPOCHS}  min_adv={min(bucket_mean_advs) if bucket_mean_advs else 0:.3f}"
                  f"  bucket_rev={bra}", flush=True)
            sp = save_prefix.replace("_ep", f"_ep{ep}")
            torch.save(pol.state_dict(), sp)
            _append_readme(sp)

    if best_sd:
        pol.load_state_dict(best_sd)
        print(f"[P2-s{seed}] Best ep={best_ep} min_adv={best_min_adv:.3f}", flush=True)
    return pol, best_ep


# ── README append ──────────────────────────────────────────────────────────────

def _sha8(path):
    try: return hashlib.sha256(open(path,"rb").read()).hexdigest()[:8]
    except: return "????????"

def _append_readme(path):
    sha = _sha8(path)
    line = f"| `{os.path.basename(path)}` | budget-transformer training | sha={sha} |\n"
    try:
        with open(README,"a") as f: f.write(line)
    except Exception: pass


# ── Trajectory cache ──────────────────────────────────────────────────────────

def _build_traj_cache(graphs, seed):
    """Build Phase-1 trajectory cache. Reuses if cache key matches."""
    import pickle
    cache_path = os.path.join(_ROOT, f"results/logs/traj_cache_budget_s{seed}.pkl")
    key = ("budget_expert", "monotone", "flat", W_HIGH, str(K_SAMPLES_P1), str(N_SEEDS_P1),
           str([g.number_of_nodes() for g in graphs]))
    if os.path.exists(cache_path):
        try:
            with open(cache_path,"rb") as f: saved_key, traj_cache = pickle.load(f)
            if saved_key == str(key):
                print(f"[traj] Cache HIT: {cache_path}", flush=True)
                return traj_cache
        except Exception: pass
    print(f"[traj] Cache MISS — building trajectories...", flush=True)
    traj_cache = {}
    for gi, G in enumerate(graphs):
        for k in K_SAMPLES_P1:
            B0 = k * C
            for s in range(N_SEEDS_P1):
                try:
                    traj = generate_budget_expert_trajectory(G, B0, C, seed=s,
                                                              weight_high=W_HIGH,
                                                              n_mc=N_MC)
                    traj_cache[(gi,k,s)] = traj
                except Exception as e:
                    pass
    with open(cache_path,"wb") as f: pickle.dump((str(key), traj_cache), f)
    print(f"[traj] Saved {len(traj_cache)} trajectories → {cache_path}", flush=True)
    return traj_cache


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--skip_p1", action="store_true", help="Skip Phase 1 (load p1_end ckpt)")
    ap.add_argument("--p1_ckpt", default="", help="Path to Phase-1 checkpoint for --skip_p1")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)
    print(f"=== Budget Transformer Training — seed={args.seed} device={device} ===")
    print(f"Base: {BASE_CKPT} (sha={BASE_SHA})")
    print(f"B_MAX={B_MAX}, C={C}, STD_FLOOR={STD_FLOOR}")
    print(f"P1: {PH1_EPOCHS} epochs, k∈{K_SAMPLES_P1}, {N_SEEDS_P1} seeds/k")
    print(f"P2: {PH2_EPOCHS} epochs, k log-uniform [1,40], entropy={ENTROPY_COEF}", flush=True)

    # Build training graphs
    graphs = [generate_forest_fire(n, FF_P, FF_PB, seed=args.seed*100+i)
              for i,n in enumerate(TRAIN_SIZES)]
    print(f"Training graphs: {[g.number_of_nodes() for g in graphs]}", flush=True)

    save_prefix = os.path.join(CKPT_DIR, f"transf_budget_s{args.seed}_ep.pt")

    if args.skip_p1 and args.p1_ckpt and os.path.exists(args.p1_ckpt):
        print(f"[main] Skipping Phase 1 — loading {args.p1_ckpt}", flush=True)
        cfg = load_config_with_base(CFG_TFM)
        in_dim = int(cfg.features.dim)+1
        H=int(cfg.encoder.hidden_dim); NL=int(cfg.encoder.n_layers); DO=float(cfg.encoder.dropout)
        enc = GraphSAGEEncoder(in_dim,H,NL,DO)
        tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
        pol = TransformerJointPolicy(enc,tfm,gnn_dim=H,context_dim=tfm.context_dim).to(device)
        pol.load_state_dict(torch.load(args.p1_ckpt,map_location=device,weights_only=True))
    else:
        pol, in_dim = _load_and_extend(BASE_CKPT, device)
        traj_cache  = _build_traj_cache(graphs, args.seed)
        pol         = phase1(pol, graphs, traj_cache, device, args.seed, save_prefix)
        p1_path = save_prefix.replace("_ep.pt", f"_p1_ep{PH1_EPOCHS}.pt")
        torch.save(pol.state_dict(), p1_path)
        _append_readme(p1_path)
        print(f"[main] Phase 1 done → {p1_path}  sha={_sha8(p1_path)}", flush=True)

    pol, best_ep = phase2(pol, graphs, device, args.seed, save_prefix)
    best_path = save_prefix.replace("_ep.pt", f"_best.pt")
    torch.save(pol.state_dict(), best_path)
    _append_readme(best_path)
    sha = _sha8(best_path)
    print(f"\n[main] Final checkpoint → {best_path}  sha={sha}", flush=True)
    print(sha)  # bare hash for quick capture

    # Git commit
    import subprocess
    subprocess.run(["git","add","-f",best_path], cwd=_ROOT)
    subprocess.run(["git","add","-f",README], cwd=_ROOT)
    subprocess.run(["git","commit","-m",f"transf_budget_s{args.seed}_best sha={sha}"], cwd=_ROOT)
    h=subprocess.run(["git","rev-parse","--short","HEAD"],capture_output=True,text=True,cwd=_ROOT).stdout.strip()
    print(h)

if __name__ == "__main__":
    main()
