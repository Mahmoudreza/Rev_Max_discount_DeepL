#!/usr/bin/env python3
"""spotcheck_gatefail.py — spot-check sha=00071438 unified on Rice_FB/Modular_FF k=5,15.
Step1: provenance. Step2: anchor FF_1000 k=10 (expect 352.7±5). Step3: 4 cells. Step4: diag.
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
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.utils.budget_features import compute_budget_node_features_fast

CKPT  = "results/checkpoints/rev_gnn_lstm_unified_gatefail.pt"
C     = 0.3
W_HIGH= 2.0
N_MC  = 5
SEEDS = [42, 123, 7]

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
def _run_episode(policy, G, cache, ei, seed, device, k, collect_diag=False):
    n = G.number_of_nodes()
    B = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    policy.reset_episode(device)
    rev = 0.0
    diag = dict(discounts=[], n_skip=0, n_accept=0, n_offer=0)
    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(cache, env.S, env.offered, env.t,
                                                   k=n, env=env)
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
            if collect_diag: diag["n_skip"] += 1
            continue
        if collect_diag:
            diag["discounts"].append(d); diag["n_offer"] += 1
        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            rev += info["offered_price"]
            if collect_diag: diag["n_accept"] += 1
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.0))
        if done: break
    if collect_diag:
        diag["wallet"] = float(env.B)
    return rev, diag

def eval_net(pol, G, k, device, collect_diag=False):
    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei    = _edge_index(G, device)
    revs  = []
    last_diag = {}
    for s in SEEDS:
        r, d = _run_episode(pol, G, cache, ei, s, device, k, collect_diag and (s==SEEDS[-1]))
        revs.append(r)
        if d: last_diag = d
    return float(np.mean(revs)), revs, last_diag

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: provenance
    sha8 = hashlib.sha256(open(CKPT,"rb").read()).hexdigest()[:8]
    print(f"[1] OURS_SMALL  {CKPT}  sha8={sha8}  match={'YES' if sha8=='00071438' else 'NO-STOP'}")
    if sha8 != "00071438":
        sys.exit(1)

    pol = _load(device)

    # Step 2: anchor FF_1000 k=10
    G_ff = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    mean_a, revs_a, _ = eval_net(pol, G_ff, 10, device)
    ok = abs(mean_a - 352.7) <= 5.0
    print(f"[2] anchor FF_1000 k=10  MEAN={mean_a:.1f}  seeds={[round(r,1) for r in revs_a]}  {'PASS' if ok else 'FAIL-STOP'}")
    if not ok:
        sys.exit(1)

    # Step 3: 4 cells
    SERVER = {"Rice_FB":{"5":0.4,"15":1.8},"Modular_FF":{"5":0.5,"15":2.4}}
    G_rice = load_rice_facebook()
    G_mod  = generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    print("[3] net         k   mean   seeds                          server  delta")
    for net, G in [("Rice_FB",G_rice),("Modular_FF",G_mod)]:
        for k in [5,15]:
            do_diag = (net=="Modular_FF" and k==15)
            mean, revs, diag_data = eval_net(pol, G, k, device, collect_diag=do_diag)
            srv = SERVER[net][str(k)]
            print(f"  {net:<12} k={k}  {mean:6.2f}  {[round(r,1) for r in revs]}  srv={srv}  Δ={mean-srv:+.2f}")
            if do_diag:
                ds = diag_data.get("discounts",[])
                n_off = diag_data.get("n_offer",0)
                n_acc = diag_data.get("n_accept",0)
                print(f"[4] Modular_FF k=15 diag (seed={SEEDS[-1]}):")
                print(f"    mean_discount={np.mean(ds):.3f}" if ds else "    mean_discount=N/A")
                print(f"    frac_d>0.9={sum(1 for d in ds if d>0.9)/len(ds):.3f}" if ds else "    frac_d>0.9=N/A")
                print(f"    accept_rate={n_acc/n_off:.3f}" if n_off else "    accept_rate=N/A")
                print(f"    skipped_infeasible={diag_data.get('n_skip',0)}")
                print(f"    final_wallet={diag_data.get('wallet',0):.4f}")

if __name__ == "__main__":
    main()
