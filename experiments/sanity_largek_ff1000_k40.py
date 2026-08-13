#!/usr/bin/env python3
"""sanity_largek_ff1000_k40.py — verify OURS_LARGE on FF_1000 k=40 with two protocols.

Task 2 audit: run rev_gnn_lstm_largek.pt on FF_1000 k=40 seeds=[42,123,7] via:
  A) run_largek_eval.py protocol  (k=n_nodes passed to feature fn)
  B) eval_all_methods_ksweep.py protocol AFTER fix (same: k=n_nodes)
  C) eval_all_methods_ksweep.py protocol BEFORE fix (k=budget_k=40 — should give wrong ~49.0)

Expected: A ≈ B ≈ 473.1,  C ≈ 49.0 (broken)

Run: python -u experiments/sanity_largek_ff1000_k40.py 2>&1
"""
from __future__ import annotations
import os, sys, time
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

CKPT   = "results/checkpoints/rev_gnn_lstm_largek.pt"
K_BUD  = 40        # budget k-value
C      = 0.3
B      = K_BUD * C  # = 12.0
W_HIGH = 2.0
N_MC   = 5
SEEDS  = [42, 123, 7]


def _edge_index(G, device):
    edges = list(G.edges())
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _load_policy(device):
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(CKPT, map_location=device, weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

@torch.no_grad()
def _run(policy, G, cache, ei, seed, device, k_feat_arg, label):
    """k_feat_arg: what to pass as k= to compute_budget_node_features_fast"""
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device); rev = 0.0
    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(cache, env.S, env.offered, env.t,
                                                   k=k_feat_arg, env=env)
        x  = torch.FloatTensor(feats).to(device)
        mask = torch.zeros(n, dtype=torch.bool, device=device)
        for i in env.available_nodes: mask[i] = True
        if not mask.any(): break
        sc, h, ctx, _ = policy(x, ei, mask)
        ni = int(sc.argmax().item())
        d  = float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        # SKIP enforcement (as in run_largek_eval.py)
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[sanity] device={device}  k_bud={K_BUD}  B={B:.1f}  seeds={SEEDS}")
    import hashlib
    sha8 = hashlib.sha256(open(CKPT,"rb").read()).hexdigest()[:8]
    print(f"[sanity] largek.pt sha8={sha8} (expect 3033620a)")

    G = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    n_val = G.number_of_nodes()
    print(f"[sanity] FF_1000: n={n_val} edges={G.number_of_edges()}")

    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei    = _edge_index(G, device)
    pol   = _load_policy(device)

    print(f"\n{'─'*60}")
    print(f"Protocol A/B (CORRECT): k=n_nodes={n_val} → expect ~473.1")
    revs_ab = []
    for s in SEEDS:
        r = _run(pol, G, cache, ei, s, device, k_feat_arg=n_val, label="AB")
        revs_ab.append(r)
        print(f"  seed={s}  rev={r:.1f}")
    print(f"  MEAN={np.mean(revs_ab):.1f}")

    print(f"\nProtocol C (BROKEN BUG): k=budget_k={K_BUD} → expect ~49.0")
    revs_c = []
    for s in SEEDS:
        r = _run(pol, G, cache, ei, s, device, k_feat_arg=K_BUD, label="C")
        revs_c.append(r)
        print(f"  seed={s}  rev={r:.1f}")
    print(f"  MEAN={np.mean(revs_c):.1f}")

    print(f"\n{'─'*60}")
    print(f"VERDICT: A/B={np.mean(revs_ab):.1f}  C={np.mean(revs_c):.1f}")
    ok = abs(np.mean(revs_ab) - 473.1) < 30.0 and np.mean(revs_c) < 100.0
    print(f"FIX CONFIRMED: {'YES' if ok else 'NEEDS INVESTIGATION'}")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"\nwall={time.time()-t0:.0f}s")
