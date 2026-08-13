#!/usr/bin/env python3
"""sanity_unified_ff1000_k10.py — anchor OURS_SMALL on FF_1000 k=10, seeds=[42,123,7].
Expected: MEAN near 352.7 (from unified_sweep.json).
Run: python -u experiments/sanity_unified_ff1000_k10.py 2>&1
"""
from __future__ import annotations
import hashlib, os, sys, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.utils.budget_features import compute_budget_node_features_fast

CKPT   = "results/checkpoints/rev_gnn_lstm_unified.pt"
K_BUD  = 10
C      = 0.3
B      = K_BUD * C   # = 3.0
W_HIGH = 2.0
N_MC   = 5
SEEDS  = [42, 123, 7]
# Original gate used sha=00071438, both Mac+server now have sha=57c23076.
# Test BOTH round_ratio conventions to identify which protocol matches ~352.7:
#   Convention A: k_feat=n_nodes (largek harness)
#   Convention B: k_feat=budget_k (possible unified harness)
EXPECT = 352.7
TOL    = 30.0

def _edge_index(G, device):
    edges = list(G.edges())
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _load(device):
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

@torch.no_grad()
def _run_episode(policy, G, cache, ei, seed, device, k_feat):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device); rev = 0.0
    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(cache, env.S, env.offered, env.t,
                                                   k=k_feat, env=env)
        x = torch.FloatTensor(feats).to(device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for i in env.available_nodes: mask[i] = True
        if not mask.any(): break
        sc, h, ctx, _ = policy(x, ei, mask)
        ni = int(sc.argmax().item())
        d  = float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        ev = env._estimate_valuation(env.nodes[ni])
        p  = ev * (1.0 - d)
        if env.B - C + p < -1e-9:
            env.offered.add(env.nodes[ni]); env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(d, False, 0.0)
            continue
        _, _, done, info = env.step(ni, d)
        if info["accepted"]: rev += info["offered_price"]
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.0))
        if done: break
    return rev

def main():
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sha8 = hashlib.sha256(open(CKPT,"rb").read()).hexdigest()[:8]
    print(f"[anchor-unified] unified.pt sha8={sha8}  k={K_BUD}  B={B:.1f}  device={device}")
    print(f"  NOTE: original gate sha=00071438; current={sha8}")

    G = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    n_val = G.number_of_nodes()
    print(f"  FF_1000: n={n_val} edges={G.number_of_edges()}")
    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei    = _edge_index(G, device)
    pol   = _load(device)

    # Convention A: k_feat=n_nodes (largek harness convention)
    print(f"\n--- Convention A: k_feat=n_nodes={n_val} (round_ratio=t/{n_val})")
    revs_a = []
    for s in SEEDS:
        r = _run_episode(pol, G, cache, ei, s, device, k_feat=n_val)
        revs_a.append(r)
        print(f"  seed={s}  rev={r:.1f}")
    mean_a = float(np.mean(revs_a))
    ok_a = abs(mean_a - EXPECT) < TOL
    print(f"  MEAN_A={mean_a:.1f}  expected={EXPECT}  {'PASS ✓' if ok_a else 'FAIL'}")

    # Convention B: k_feat=budget_k (possible unified training convention)
    print(f"\n--- Convention B: k_feat=budget_k={K_BUD} (round_ratio=t/{K_BUD})")
    revs_b = []
    for s in SEEDS:
        r = _run_episode(pol, G, cache, ei, s, device, k_feat=K_BUD)
        revs_b.append(r)
        print(f"  seed={s}  rev={r:.1f}")
    mean_b = float(np.mean(revs_b))
    ok_b = abs(mean_b - EXPECT) < TOL
    print(f"  MEAN_B={mean_b:.1f}  expected={EXPECT}  {'PASS ✓' if ok_b else 'FAIL'}")

    # Verdict
    print(f"\n{'='*60}")
    if ok_a:
        print(f"ANCHOR: PASS  Convention A  MEAN={mean_a:.1f}  sha={sha8}")
        print(f"  → ksweep should use k_feat=n_nodes (Convention A)")
    elif ok_b:
        print(f"ANCHOR: PASS  Convention B  MEAN={mean_b:.1f}  sha={sha8}")
        print(f"  → ksweep must use k_feat=budget_k (Convention B) — update _feat_budget!")
    else:
        print(f"ANCHOR: FAIL  A={mean_a:.1f}  B={mean_b:.1f}  both differ from {EXPECT} by >{TOL}")
        print(f"  sha={sha8} may not be the same model as gate sha=00071438")
    print(f"  wall={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
