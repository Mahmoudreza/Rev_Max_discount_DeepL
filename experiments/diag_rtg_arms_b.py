#!/usr/bin/env python3
"""diag_rtg_arms_b.py — Print per-step advantage statistics for Arm B.
Reports RTG, Welford-baseline, raw advantage, normalised advantage
at steps 1, 100, 500, 900 for one full episode.

Usage:
  venv/bin/python3 -u experiments/diag_rtg_arms_b.py \
      --ckpt results/checkpoints/c1_b_s1_ep0030.pt --device cuda:3
"""
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import numpy as np
import torch
import torch.nn.functional as F

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.utils.helpers import set_seed

HID = 64; MC = 5; STD_FLOOR = 1.0

def _make_pol(in_dim=20):
    enc  = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=HID, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    return SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)

def load_pol(path, device):
    ckpt = torch.load(path, map_location=device)
    for key in ('policy_state_dict', 'model_state_dict', 'state_dict'):
        if isinstance(ckpt, dict) and key in ckpt:
            sd = ckpt[key]; break
    else:
        sd = ckpt
    w = sd.get('encoder.input_proj.weight', sd.get('encoder.layers.0.weight', None))
    in_dim = int(w.shape[1]) if w is not None else 20
    pol = _make_pol(in_dim=in_dim).to(device)
    pol.load_state_dict(sd); pol.eval()
    return pol

def run_episode(pol, G, seed, device):
    """Return full list of rewards (one per step)."""
    set_seed(seed)
    cfg = RevenueEnvConfig(influence_model="monotone", b=1.0,
                           weight_low=0.0, weight_high=2.0,
                           n_mc_samples=MC, reward_type="flat", gamma=1.0, seed=seed)
    env = RevenueEnv(G, cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    from torch import tensor
    m = {v: i for i,v in enumerate(nodes)}; E = list(G.edges())
    s = [m[u] for u,_ in E]+[m[v] for _,v in E]
    d = [m[v] for _,v in E]+[m[u] for u,_ in E]
    ei = tensor([s,d],dtype=torch.long).to(device)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    pol.reset_episode(device)
    rewards = []
    for _ in range(n):
        if not any(v not in env.offered for v in nodes): break
        feats = compute_node_features_fast(cache, env.S, set(env.offered), env.t, n, env)
        x  = tensor(feats[:, :getattr(pol,'_in_dim',20)], dtype=torch.float32, device=device)
        av = tensor([v not in env.offered for v in nodes], dtype=torch.bool, device=device)
        with torch.no_grad():
            ms, h, ctx, _ = pol.forward(x, ei, av)
            safe = ms.clone(); safe[~av] = -1e9
            sel = int(safe.argmax())
            disc = float(pol.get_discount_distribution(
                torch.cat([h[sel],ctx])).mean.clamp(1e-4,1-1e-4).detach())
        v = nodes[sel]
        _, r, done, _ = env.step(env.node_to_idx[v], disc)
        rewards.append(float(r))
        pol.update_sequence_state(disc, r>0, r)
        if done: break
    return rewards

class WelfordBaseline:
    def __init__(self, max_steps=2100):
        self.n = np.zeros(max_steps); self.m = np.zeros(max_steps)
    def update(self, val, t):
        self.n[t] += 1; self.m[t] += (val - self.m[t]) / self.n[t]
    def get(self, t):
        return float(self.m[t]) if self.n[t] > 0 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed",   type=int, default=0)
    ap.add_argument("--graph_size", type=int, default=1000)
    ap.add_argument("--n_episodes", type=int, default=3,
                    help="Accumulate this many episodes in Welford before reporting")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pol = load_pol(args.ckpt, device)
    pol._in_dim = getattr(pol, '_in_dim', 20)

    G = generate_forest_fire(args.graph_size, 0.37, 0.32, seed=args.seed)
    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges", flush=True)

    baseline = WelfordBaseline(max_steps=G.number_of_nodes() + 100)

    # Warm up baseline with n_episodes-1 warm-up episodes
    for ep in range(args.n_episodes - 1):
        rewards = run_episode(pol, G, seed=ep, device=device)
        rtg = []; running = 0.0
        for r in reversed(rewards):
            running += r; rtg.insert(0, running)
        for t, ret in enumerate(rtg):
            baseline.update(ret, t)
    print(f"Warmed up baseline with {args.n_episodes-1} episodes.")

    # Diagnostic episode
    rewards = run_episode(pol, G, seed=args.n_episodes-1, device=device)
    T = len(rewards)
    print(f"Episode length: {T} steps, total_reward={sum(rewards):.2f}", flush=True)

    # Compute RTG
    rtg = []; running = 0.0
    for r in reversed(rewards):
        running += r; rtg.insert(0, running)

    # Per-step: raw RTG, baseline, raw_adv, then normalised
    raw_adv = []
    for t, ret in enumerate(rtg):
        b_val = baseline.get(t)
        raw_adv.append(ret - b_val)
        baseline.update(ret, t)  # update with this episode too

    a = np.array(raw_adv)
    std_norm = max(a.std(), STD_FLOOR)
    norm_adv = (a - a.mean()) / std_norm

    print(f"\n{'step':>6}  {'RTG(raw)':>10}  {'baseline':>10}  {'raw_adv':>10}  {'norm_adv':>10}")
    for t in [0, min(99,T-1), min(499,T-1), min(899,T-1), T-1]:
        print(f"  t={t:<4d}  {rtg[t]:>10.3f}  {baseline.get(t):>10.3f}  "
              f"{raw_adv[t]:>10.3f}  {norm_adv[t]:>10.3f}", flush=True)

    print(f"\nRaw advantage  stats: mean={a.mean():+.3f}  std={a.std():.3f}  "
          f"min={a.min():+.3f}  max={a.max():+.3f}", flush=True)
    print(f"Normd advantage stats: mean={norm_adv.mean():+.3f}  std={norm_adv.std():.3f}  "
          f"min={norm_adv.min():+.3f}  max={norm_adv.max():+.3f}", flush=True)
    print(f"std_norm used: {std_norm:.3f}  (STD_FLOOR={STD_FLOOR})", flush=True)

    # Scale ratio: step 1 vs step 900
    t1   = min(0,   T-1)
    t100 = min(99,  T-1)
    t500 = min(499, T-1)
    t900 = min(899, T-1)
    print(f"\nRatio RTG[t=1]/RTG[t=900]   = {rtg[t1]:.2f}/{rtg[t900]:.3f} = "
          f"{rtg[t1]/max(abs(rtg[t900]),1e-6):.1f}x", flush=True)
    print(f"Ratio raw_adv[1]/raw_adv[900]= {raw_adv[t1]:.2f}/{raw_adv[t900]:.3f} = "
          f"{raw_adv[t1]/max(abs(raw_adv[t900]),1e-6):.1f}x", flush=True)
    print(f"Ratio norm_adv[1]/norm_adv[900]= {norm_adv[t1]:.3f}/{norm_adv[t900]:.3f} = "
          f"{norm_adv[t1]/max(abs(norm_adv[t900]),1e-6):.1f}x", flush=True)

if __name__ == "__main__":
    main()
