#!/usr/bin/env python3
"""train_arms.py — Arms A / B / C for Phase-2 credit-assignment ablation.

  ARM A: RL from scratch (no Phase-1). REINFORCE, episode-total advantage.
         N_RL_A=50 epochs × 5 graphs × 20 eps = 5000 RL episodes.
  ARM B: Phase-1 imitation (200 ep) + reward-to-go REINFORCE.
         N_RL_BC=30 epochs × 5 graphs × 20 eps = 3000 RL episodes.
  ARM C: Phase-1 imitation (200 ep) + REINFORCE on profit Pi=R-c*|S_T|.
         same 3000 RL episodes as Arm B.

Episode counts stated: Arm A = 5000, Arms B/C = 3000.

Usage (three parallel jobs):
  python train_arms.py --arm a --seed 0 --device cuda:0 &
  python train_arms.py --arm a --seed 1 --device cuda:1 &
  python train_arms.py --arm a --seed 2 --device cuda:2 &
  (and similarly for b, c)

Checkpoints: c1_{arm}_s{seed}_ep{epoch}.pt (never overwrites existing).
SHA appended to results/checkpoints/README.md after each save.
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                  compute_node_features_fast)
from src.utils.helpers import set_seed
from src.evaluation.baselines import greedy_discount_trajectory

_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

CKPT_DIR    = "results/checkpoints"
TRAJ_CACHE  = "results/traj_cache"  # cached greedy_discount trajectories
README      = os.path.join(CKPT_DIR, "README.md")
FF_SIZES    = [200, 260, 320, 380, 440]
HID         = 64
NIM         = 200          # Phase-1 imitation epochs (Arms B, C)
SUBSAMPLE   = 30
PW          = 0.3          # mse weight in imitation loss
LR_P1       = 1e-3
LR_P2       = 3e-4
MC          = 5
GRAD_CLIP   = 1.0
ENT_COEF    = 0.01
STD_FLOOR   = 1.0
C_PROD      = 0.30         # production cost (Arm C)
EPS_PER_EPOCH = 20         # episodes per graph per RL epoch
N_RL_A      = 50           # RL epochs for Arm A  → 5000 eps
N_RL_BC     = 30           # RL epochs for Arms B/C → 3000 eps
SAVE_EVERY  = 10           # save checkpoint every N RL epochs


def _sha8(p): return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

def _ei(G):
    nm = {v: i for i, v in enumerate(G.nodes())}
    E = list(G.edges())
    s = [nm[u] for u,_ in E]+[nm[v] for _,v in E]
    d = [nm[v] for _,v in E]+[nm[u] for u,_ in E]
    return torch.tensor([s, d], dtype=torch.long)

def _make_pol():
    enc  = GraphSAGEEncoder(in_dim=20, hidden_dim=HID, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)

def _make_env(G, seed):
    cfg = RevenueEnvConfig(influence_model="monotone", b=1.0, weight_low=0.0,
                           weight_high=2.0, n_mc_samples=MC, reward_type="flat",
                           gamma=1.0, seed=seed)
    return RevenueEnv(G, cfg)


# ── Phase-1 imitation (shared by B, C) ──────────────────────────────────────

def collect_trajectories(G, n_traj, seed_offset, graph_seed=None):
    """Collect greedy_discount trajectories; cache to disk by (graph_seed, seed_offset).
    Trajectories depend only on (graph_seed, seed_offset, n_traj, MC) — identical
    across arms b/c/p1 for the same arm-seed, so caching is safe.
    """
    import pickle
    cache_hit = False
    cache_path = None
    if graph_seed is not None:
        os.makedirs(TRAJ_CACHE, exist_ok=True)
        cache_path = os.path.join(TRAJ_CACHE,
            f"trajs_expert-greedydisc_gseed{graph_seed}_off{seed_offset}_n{n_traj}_mc{MC}.pkl")
        if os.path.exists(cache_path):
            trajs = pickle.load(open(cache_path, 'rb'))
            print(f"    CACHE HIT {cache_path} ({len(trajs)} trajs)", flush=True)
            return trajs

    from omegaconf import OmegaConf
    cfg = OmegaConf.create({"influence":{"model":"monotone","b":1.0,"weight_low":0.0,
                               "weight_high":2.0,"n_mc_samples":MC},
                               "reward":{"type":"flat","gamma":1.0},"budget":{"k":50},
                               "project":{"seed":0}})
    trajs = []
    for i in range(n_traj):
        cfg.project.seed = seed_offset + i
        try:
            traj = greedy_discount_trajectory(G, cfg)
            if traj: trajs.append(traj)
        except Exception:
            pass

    if cache_path:
        pickle.dump(trajs, open(cache_path, 'wb'))
        print(f"    CACHE SAVED {cache_path} ({len(trajs)} trajs)", flush=True)
    return trajs

def imitation_loss(pol, G, ei, cache, trajs, device, subsample=SUBSAMPLE):
    pol.train()
    idxs = np.random.choice(len(trajs), min(subsample, len(trajs)), replace=False)
    total_loss = torch.tensor(0.0, device=device)
    nodes = list(G.nodes()); n = G.number_of_nodes()
    node_map = {v: i for i, v in enumerate(nodes)}
    ei_d = ei.to(device)
    for ti in idxs:
        traj = trajs[ti]
        env = _make_env(G, seed=ti); env.reset(); pol.reset_episode(device)
        ep_ce = torch.tensor(0.0, device=device)
        ep_mse = torch.tensor(0.0, device=device)
        for item in traj:
            # greedy_discount_trajectory returns List[Dict] with "node_idx"/"discount"
            if isinstance(item, dict):
                node_idx_t = item["node_idx"]
                disc_target = float(item["discount"])
                node = nodes[node_idx_t]
            else:
                node_idx_t, disc_target = item[0], float(item[1])
                node = nodes[node_idx_t] if isinstance(node_idx_t, int) and node_idx_t < len(nodes) else node_idx_t
            feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
            x  = torch.tensor(feats[:,:20], dtype=torch.float32, device=device)
            av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
            ms, h, ctx, _ = pol.forward(x, ei_d, av)
            target_idx = node_map.get(node, 0)
            ep_ce  = ep_ce  + F.cross_entropy(ms.unsqueeze(0), torch.tensor([target_idx], device=device))
            comb   = torch.cat([h[target_idx], ctx])
            ep_mse = ep_mse + F.mse_loss(pol.get_discount_distribution(comb).mean,
                                         torch.tensor(float(disc_target), device=device))
            price = env._estimate_valuation(node) * (1.0 - float(disc_target))
            tv = env._true_valuation(node)
            if tv >= price: env.S.add(node)
            env.offered.add(node); env.t += 1
            pol.update_sequence_state(float(disc_target), tv >= price, price if tv >= price else 0.0)
        n_steps = max(1, len(traj))
        total_loss = total_loss + (ep_ce + PW * ep_mse) / n_steps
    return total_loss / max(1, len(idxs))


# ── REINFORCE episode ─────────────────────────────────────────────────────────

def rl_episode(pol, G, ei, cache, device, rng_seed):
    """Run one episode, return (log_probs, rewards, entropies, n_sales)."""
    pol.train()
    set_seed(rng_seed)
    env = _make_env(G, seed=rng_seed % 10000); env.reset()
    nodes = list(G.nodes()); n = G.number_of_nodes()
    # Build idx map locally; works regardless of whether env exposes node_to_idx
    node_to_idx = {v: i for i, v in enumerate(nodes)}
    pol.reset_episode(device)
    ei_d = ei.to(device)
    log_probs, rewards, entropies = [], [], []

    while len(env.offered) < n:
        feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        x  = torch.tensor(feats[:,:20], dtype=torch.float32, device=device)
        av = torch.tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        if not av.any(): break
        ms, h, ctx, _ = pol.forward(x, ei_d, av)

        # Safe sampling: mask unavailable nodes, use stable softmax
        with torch.no_grad():
            safe_ms = ms.clone()
            safe_ms[~av] = -1e9
            safe_probs = F.softmax(safe_ms.float(), dim=0).clamp(min=0)
            if not torch.isfinite(safe_probs).all() or safe_probs.sum() < 1e-8:
                safe_probs = av.float() / av.float().sum()
            safe_probs = safe_probs / safe_probs.sum()
            sel_idx = int(torch.multinomial(safe_probs.cpu(), 1))
        # Log-prob for gradient (from original ms with gradients)
        log_p_sel = F.log_softmax(ms, dim=0)
        lp_node   = log_p_sel[sel_idx]
        # Entropy: mask out unavailable to avoid nan
        lp_for_ent = log_p_sel.clone()
        lp_for_ent[~av] = 0.0
        probs_for_ent = safe_probs.to(ms.device)
        ent_node = -(probs_for_ent * lp_for_ent).sum()

        comb = torch.cat([h[sel_idx], ctx])
        dist = pol.get_discount_distribution(comb)
        disc_t = dist.rsample().clamp(1e-4, 1-1e-4)
        lp_disc = dist.log_prob(disc_t)
        try: ent_disc = dist.entropy()
        except Exception: ent_disc = torch.tensor(0.0, device=device)

        v = nodes[sel_idx]; disc_val = float(disc_t.detach())
        # Use local node_to_idx; fall back to sel_idx if env provides it
        idx_for_step = getattr(env, 'node_to_idx', node_to_idx)[v]
        _, reward, done, _ = env.step(idx_for_step, disc_val)

        log_probs.append(lp_node + lp_disc)
        rewards.append(float(reward))
        entropies.append(ent_node + ent_disc)
        pol.update_sequence_state(disc_val, reward > 0, reward)
        if done: break

    n_sales = sum(1 for r in rewards if r > 0)
    return log_probs, rewards, entropies, n_sales


class WelfordBaseline:
    """Welford running mean (single value for Arms A/C; per-position for Arm B)."""
    def __init__(self, per_step=False, max_steps=2100):
        self.per_step = per_step
        if per_step:
            self.n = np.zeros(max_steps, dtype=np.float64)
            self.m = np.zeros(max_steps, dtype=np.float64)
        else:
            self.n = 0; self.m = 0.0

    def update(self, val, t=None):
        if self.per_step:
            self.n[t] += 1
            self.m[t] += (val - self.m[t]) / self.n[t]
        else:
            self.n += 1
            self.m += (val - self.m) / self.n

    def get(self, t=None):
        if self.per_step:
            if t is None:
                # Called for display only — return mean of populated positions
                mask = self.n > 0
                return float(np.mean(self.m[mask])) if mask.any() else 0.0
            return float(self.m[t]) if self.n[t] > 0 else 0.0
        return float(self.m)


def rl_update(pol, opt, G, ei, cache, device, arm, baseline, rng_seed):
    """Run EPS_PER_EPOCH episodes, compute policy gradient, step optimizer."""
    all_losses = []; total_rev = 0.0; total_n_sales = 0

    for ep in range(EPS_PER_EPOCH):
        log_probs, rewards, entropies, n_sales = rl_episode(
            pol, G, ei, cache, device, rng_seed + ep)
        total_rev += sum(rewards); total_n_sales += n_sales

        if arm == 'b':
            # Reward-to-go per step
            rtg = []
            running = 0.0
            for r in reversed(rewards):
                running += r
                rtg.insert(0, running)
            adv_arr = []
            for t, ret in enumerate(rtg):
                baseline.update(ret, t=t)
                adv_arr.append(ret - baseline.get(t=t))
            # Normalise advantages
            if len(adv_arr) > 1:
                a = np.array(adv_arr)
                std = max(a.std(), STD_FLOOR)
                adv_arr = ((a - a.mean()) / std).tolist()
        else:
            # Episode total (Arms A and C)
            if arm == 'c':
                total_r = sum(rewards) - C_PROD * n_sales
            else:
                total_r = sum(rewards)
            baseline.update(total_r)
            raw_adv = total_r - baseline.get()
            adv_std = max(abs(raw_adv), STD_FLOOR)
            adv = raw_adv / adv_std
            adv_arr = [adv] * len(log_probs)

        ep_loss = torch.tensor(0.0, device=device)
        for lp, adv_t, ent in zip(log_probs, adv_arr, entropies):
            ep_loss = ep_loss - adv_t * lp - ENT_COEF * ent
        ep_loss = ep_loss / max(1, len(log_probs))
        all_losses.append(ep_loss)

    loss = sum(all_losses) / max(1, len(all_losses))
    opt.zero_grad(); loss.backward()
    torch.nn.utils.clip_grad_norm_(pol.parameters(), GRAD_CLIP)
    opt.step()
    mean_rev = total_rev / max(1, EPS_PER_EPOCH)
    mean_ns  = total_n_sales / max(1, EPS_PER_EPOCH)
    return float(loss.detach()), mean_rev, mean_ns


# ── Save helper ──────────────────────────────────────────────────────────────

def _save(pol, path, meta):
    if os.path.exists(path):
        print(f"  SKIP (exists): {path}"); return
    os.makedirs(CKPT_DIR, exist_ok=True)
    torch.save({"policy_state_dict": pol.state_dict(), **meta}, path)
    sha = _sha8(path)
    print(f"  Saved {path}  sha={sha}", flush=True)
    with open(README, "a") as f:
        f.write(f"\n## {os.path.basename(path)}\n")
        f.write(f"sha8={sha}  arm={meta.get('arm')}  seed={meta.get('seed')}  "
                f"rl_epoch={meta.get('rl_epoch')}\n")
    import subprocess; _ROOT = str(Path(__file__).parent.parent)
    subprocess.run(["git","add","-f",path,README], cwd=_ROOT)
    h = subprocess.run(["git","commit","-m",f"ckpt {os.path.basename(path)} sha={sha}"],
                       cwd=_ROOT, capture_output=True, text=True).stdout.strip()
    print(f"  commit: {h or '(already up to date)'}", flush=True)


# ── Main training function ───────────────────────────────────────────────────

def train(arm, seed, device_str, load_p1=""):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    set_seed(seed)
    print(f"\n=== ARM {arm.upper()} seed={seed} device={device} ===", flush=True)

    pol = _make_pol().to(device)

    # Build graphs, caches, edge indices
    graphs=[]; eis=[]; caches=[]
    for i, sz in enumerate(FF_SIZES):
        G = generate_forest_fire(sz, 0.37, 0.32, seed=seed*100+i)
        graphs.append(G); eis.append(_ei(G))
        caches.append(build_graph_feature_cache(G, compute_static_features(G)))

    # ── PHASE 1: imitation (Arms B and C only) ───────────────────────────────
    if arm in ('b', 'c') and load_p1:
        # Load shared Phase-1 checkpoint, skip imitation training
        ckpt = torch.load(load_p1, map_location=device)
        pol.load_state_dict(ckpt["policy_state_dict"])
        print(f"  Loaded Phase-1 from {load_p1}  (nim={ckpt.get('nim','?')})", flush=True)
    elif arm in ('b', 'c'):
        print(f"Phase-1 imitation: {NIM} epochs", flush=True)
        opt_p1 = torch.optim.Adam(pol.parameters(), lr=LR_P1)
        trajs_all = []
        for i, (G, ei, cache) in enumerate(zip(graphs, eis, caches)):
            print(f"  Collecting trajs G{i}...", flush=True)
            tr = collect_trajectories(G, n_traj=200, seed_offset=seed*1000+i*200,
                                      graph_seed=seed*100+i)
            trajs_all.append(tr); print(f"    → {len(tr)} trajs", flush=True)
        for ep in range(NIM):
            ep_loss = 0.0
            for G, ei, cache, trajs in zip(graphs, eis, caches, trajs_all):
                if not trajs: continue
                loss = imitation_loss(pol, G, ei, cache, trajs, device)
                opt_p1.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), GRAD_CLIP)
                opt_p1.step(); ep_loss += float(loss)
            if ep % 50 == 0 or ep < 3:
                print(f"  P1 ep={ep:3d}  loss={ep_loss/len(graphs):.4f}", flush=True)
        print("Phase-1 done.", flush=True)

    # ── PHASE 2: REINFORCE ───────────────────────────────────────────────────
    n_rl_epochs = N_RL_A if arm == 'a' else N_RL_BC
    total_rl_eps = n_rl_epochs * len(FF_SIZES) * EPS_PER_EPOCH
    print(f"Phase-2 RL: {n_rl_epochs} epochs × {len(FF_SIZES)} graphs × "
          f"{EPS_PER_EPOCH} eps = {total_rl_eps} RL episodes total", flush=True)

    opt_p2 = torch.optim.Adam(pol.parameters(), lr=LR_P2)
    per_step_baseline = (arm == 'b')
    baseline = WelfordBaseline(per_step=per_step_baseline)
    rng = np.random.RandomState(seed + 42)

    for ep in range(n_rl_epochs):
        epoch_rev = 0.0
        for gi, (G, ei, cache) in enumerate(zip(graphs, eis, caches)):
            rs = int(rng.randint(0, 100000))
            loss_val, mr, mn = rl_update(pol, opt_p2, G, ei, cache, device,
                                          arm, baseline, rng_seed=rs)
            epoch_rev += mr
        mean_rev = epoch_rev / len(graphs)
        if ep % 5 == 0 or ep == n_rl_epochs - 1:
            print(f"  RL ep={ep:3d}/{n_rl_epochs}  mean_rev={mean_rev:.2f}  "
                  f"baseline={baseline.get():.2f}", flush=True)

        if (ep + 1) % SAVE_EVERY == 0 or ep == n_rl_epochs - 1:
            ckpt = os.path.join(CKPT_DIR,
                                f"c1_{arm}_s{seed}_ep{(ep+1):04d}.pt")
            _save(pol, ckpt, {"arm": arm, "seed": seed, "rl_epoch": ep+1,
                               "total_rl_eps": total_rl_eps,
                               "recipe": f"arm_{arm}_FF_only"})

    print(f"=== ARM {arm.upper()} seed={seed} DONE ===", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm",     required=True, choices=["a","b","c","p1"])
    ap.add_argument("--seed",    type=int, default=0)
    ap.add_argument("--device",  default="cuda:0")
    ap.add_argument("--save_p1", default="", help="Phase-1-only: save ckpt here then exit")
    ap.add_argument("--load_p1", default="", help="Skip Phase-1: load this ckpt before Phase-2")
    args = ap.parse_args()

    if args.arm == "p1" or args.save_p1:
        # Phase-1 only mode: train imitation, save, exit
        real_arm = "b"  # B and C share identical Phase-1
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        set_seed(args.seed)
        print(f"\n=== PHASE-1 ONLY seed={args.seed} device={device} ===", flush=True)
        pol = _make_pol().to(device)
        graphs=[]; eis=[]; caches=[]
        for i, sz in enumerate(FF_SIZES):
            G = generate_forest_fire(sz, 0.37, 0.32, seed=args.seed*100+i)
            graphs.append(G); eis.append(_ei(G))
            caches.append(build_graph_feature_cache(G, compute_static_features(G)))
        print(f"Phase-1 imitation: {NIM} epochs", flush=True)
        opt_p1 = torch.optim.Adam(pol.parameters(), lr=LR_P1)
        trajs_all = []
        for i, (G, ei, cache) in enumerate(zip(graphs, eis, caches)):
            print(f"  Collecting trajs G{i}...", flush=True)
            tr = collect_trajectories(G, n_traj=200, seed_offset=args.seed*1000+i*200,
                                      graph_seed=args.seed*100+i)
            trajs_all.append(tr); print(f"    → {len(tr)} trajs", flush=True)
        for ep in range(NIM):
            ep_loss = 0.0
            for G, ei, cache, trajs in zip(graphs, eis, caches, trajs_all):
                if not trajs: continue
                loss = imitation_loss(pol, G, ei, cache, trajs, device)
                opt_p1.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(pol.parameters(), GRAD_CLIP)
                opt_p1.step(); ep_loss += loss.item()
            if ep % 10 == 0 or ep < 3:
                print(f"  P1 ep={ep:3d}  loss={ep_loss/len(graphs):.4f}", flush=True)
        out = args.save_p1 or os.path.join(CKPT_DIR, f"c1_p1_s{args.seed}_ep{NIM:04d}.pt")
        os.makedirs(CKPT_DIR, exist_ok=True)
        torch.save({"policy_state_dict": pol.state_dict(), "seed": args.seed, "nim": NIM}, out)
        sha = _sha8(out)
        print(f"Phase-1 done. Saved {out}  sha={sha}", flush=True)
        import subprocess; _ROOT = str(Path(__file__).parent.parent)
        subprocess.run(["git","add","-f",out,README], cwd=_ROOT)
        subprocess.run(["git","commit","-m",f"p1 ckpt s{args.seed} sha={sha}"], cwd=_ROOT, capture_output=True)
    else:
        train(args.arm, args.seed, args.device, load_p1=args.load_p1)
