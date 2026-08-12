#!/usr/bin/env python3
"""quick_modff_eval.py — 5-seed Modular_FF eval for arm_b checkpoint progress check."""
from __future__ import annotations
import hashlib, os, sys, time
import numpy as np
import torch
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_modular_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

CKPT = "results/checkpoints/rev_gnn_lstm_densemix.pt"
C = 0.3; K_EVAL = 50; WEIGHT_HIGH = 2.0; N_MC = 5
SEEDS = list(range(5))

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _load_policy(device):
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(CKPT, map_location=device, weights_only=True)
    if "policy_state_dict" in sd:   sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd:  sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _avail(env, n, device):
    m = torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: m[i]=True
    return m

def _feat(cache, env):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, K_EVAL, env)
    return np.concatenate([base, np.ones((cache["n"],1),dtype=np.float32)], axis=1)

@torch.no_grad()
def eval_episode(policy, G, cache, ei, seed, device):
    cfg = BudgetEnvConfig(budget_B=K_EVAL*C, production_cost=C, seed=seed,
                         weight_high=WEIGHT_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device); rev = 0.0
    for _ in range(n):
        if not env.available_nodes or env._check_bankrupt(): break
        x  = torch.FloatTensor(_feat(cache, env)).to(device)
        av = _avail(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = policy.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        _, _, done, info = env.step(ni, d)
        if info["accepted"]: rev += info["offered_price"]
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.0))
        if done: break
    return rev

def main():
    sha = _sha8(CKPT)
    print(f"checkpoint sha8={sha}", flush=True)
    device = torch.device("cpu")
    policy = _load_policy(device)

    # Modular_FF: 2×250 nodes (matches run_topology_arms_eval.py)
    G = generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    ei    = _edge_index(G, device)
    cache = build_graph_feature_cache(G, compute_static_features(G))
    print(f"Modular_FF n={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)

    revs = []
    for s in SEEDS:
        t0 = time.time()
        r = eval_episode(policy, G, cache, ei, s, device)
        revs.append(r)
        print(f"  seed={s} rev={r:.1f} ({time.time()-t0:.0f}s)", flush=True)

    m = float(np.mean(revs))
    print(f"\nModular_FF MEAN={m:.1f}  ref=414.4  diff={m-414.4:+.1f}", flush=True)

if __name__ == "__main__":
    main()
