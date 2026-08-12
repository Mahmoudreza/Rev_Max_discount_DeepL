#!/usr/bin/env python3
"""env_validate_frozen.py — Reproduce FF_1000=448.6 and Rice=214.1 from frozen rev_gnn_lstm.pt.
Unconstrained protocol: budget_col=1.0, k=50, N_MC=5, 5 seeds (0..4).
"""
from __future__ import annotations
import hashlib, os, sys
import numpy as np
import torch
import networkx as nx
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

CKPT = "results/checkpoints/rev_gnn_lstm.pt"
EXPECTED_SHA = "8fbc4648"
C = 0.3; K_EVAL = 50; WEIGHT_HIGH = 2.0; N_MC = 5
SEEDS = list(range(5))

def _sha8(p):
    return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _load_policy(device):
    # rev_gnn_lstm.pt (8fbc4648) = Idea-1 checkpoint: 20-dim features (no budget col)
    enc  = GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(CKPT, map_location=device, weights_only=True)
    if "policy_state_dict" in sd:  sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges:
        return torch.zeros((2,0), dtype=torch.long, device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges] + [nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges] + [nmap[u] for u,_ in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)

def _avail(env, n, device):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for i in env.available_nodes: m[i] = True
    return m

@torch.no_grad()
def eval_episode(policy, G, cache, ei, seed, device):
    cfg = BudgetEnvConfig(budget_B=K_EVAL*C, production_cost=C, seed=seed,
                         weight_high=WEIGHT_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device)
    rev = 0.0
    for _ in range(n):
        if not env.available_nodes or env._check_bankrupt(): break
        # Idea-1 frozen checkpoint: 20-dim features (no budget column)
        feat = compute_node_features_fast(cache, env.S, env.offered, env.t, K_EVAL, env)
        x    = torch.FloatTensor(feat).to(device)
        av   = _avail(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = policy.forward(x, ei, av)
        ni  = int(sc.argmax().item())
        d   = float(policy.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        obs, rw, done, info = env.step(ni, d)
        if info["accepted"]: rev += info["offered_price"]
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step", 0.0))
        if done: break
    return rev

def run_net(policy, G, device, label):
    ei    = _edge_index(G, device)
    sf    = compute_static_features(G)
    cache = build_graph_feature_cache(G, sf)
    revs  = []
    for s in SEEDS:
        r = eval_episode(policy, G, cache, ei, s, device)
        revs.append(r)
        print(f"  {label} seed={s} rev={r:.1f}", flush=True)
    m = float(np.mean(revs))
    print(f"  {label} MEAN={m:.1f}  seeds={[round(v,1) for v in revs]}", flush=True)
    return m

def main():
    sha = _sha8(CKPT)
    print(f"checkpoint sha8={sha}", flush=True)
    assert sha == EXPECTED_SHA, f"SHA MISMATCH: got {sha} expected {EXPECTED_SHA}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    policy = _load_policy(device)

    print("=== FF_1000 ===", flush=True)
    ff = generate_forest_fire(n=1000, p=0.37, pb=0.32, seed=0)
    m_ff = run_net(policy, ff, device, "FF_1000")

    print("=== Rice-FB ===", flush=True)
    rice = load_rice_facebook()
    m_rice = run_net(policy, rice, device, "Rice_FB")

    ref_ff   = 448.6
    ref_rice = 214.1
    tol = 15.0  # noise tolerance
    ok_ff   = abs(m_ff   - ref_ff)   <= tol
    ok_rice = abs(m_rice - ref_rice) <= tol

    print(f"\nFF_1000:  got={m_ff:.1f}  ref={ref_ff}  diff={m_ff-ref_ff:+.1f}  {'OK' if ok_ff else 'DRIFT'}")
    print(f"Rice_FB:  got={m_rice:.1f}  ref={ref_rice}  diff={m_rice-ref_rice:+.1f}  {'OK' if ok_rice else 'DRIFT'}")
    if ok_ff and ok_rice:
        print(f"\nENV VALID (FF={m_ff:.1f} Rice={m_rice:.1f})")
    else:
        print(f"\nENV DRIFT (FF={m_ff:.1f} Rice={m_rice:.1f})")

if __name__ == "__main__":
    main()
