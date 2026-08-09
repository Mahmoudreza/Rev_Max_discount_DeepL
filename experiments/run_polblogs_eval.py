#!/usr/bin/env python3
"""run_polblogs_eval.py — Idea 1 (unconstrained) zero-shot eval on Polblogs.

Methods:
  Greedy-Discount, Rev-GNN-IM-RL, Rev-GNN-LSTM
  (IE-Strategy skipped: O(n*MC*cascade) too slow for n=1222 dense graph)

Protocols:
  (a) single-seed (seed=42, Table 1 protocol)
  (b) 5-seed (seeds 0..4, abstract-margin protocol)

Betweenness: approximate (k=200 pivots, polblogs n=1222 dense graph)
Output: results/logs/polblogs_eval.json
"""
import sys, os, json, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import networkx as nx
from types import SimpleNamespace

# Approximate betweenness (k=200 pivots) — exact O(nm) too slow for n=1222 dense graph
_orig_bc = nx.betweenness_centrality
def _approx_bc(G, normalized=True, **kwargs):
    kk = min(200, G.number_of_nodes())
    print(f"[approx_bc k={kk} n={G.number_of_nodes()}]", flush=True)
    return _orig_bc(G, k=kk, normalized=normalized, **kwargs)
nx.betweenness_centrality = _approx_bc

from src.env.polblogs_loader import load_polblogs, polblogs_stats
# greedy_discount skipped (O(n^2 x MC) too slow for n=1222; needs n_mc_samples=1 workaround)
from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.policies.joint_policy import JointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)

CKPT_DIR   = "results/checkpoints"
LSTM_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm.pt")
IMRL_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_im_rl.pt")
LSTM_SHA   = "8fbc4648"

FEAT_DIM   = 20
HID        = 64
SINGLE_SEED = 42
FIVE_SEEDS  = [0, 1, 2, 3, 4]


def _make_env_cfg(seed):
    # n_mc_samples=20: reduced from 200 for speed on polblogs n=1222 dense graph
    return RevenueEnvConfig(
        influence_model="monotone", b=1.0,
        weight_low=0.0, weight_high=2.0, n_mc_samples=20,
        reward_type="flat", gamma=1.0, seed=seed,
    )

def _make_cfg_ns(seed):
    from types import SimpleNamespace as NS
    return NS(influence=NS(model="monotone",b=1.0,weight_low=0.,weight_high=2.,n_mc_samples=200),
              reward=NS(type="flat",gamma=1.0), budget=NS(k=0), project=NS(seed=seed))

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _load_lstm(device):
    enc  = GraphSAGEEncoder(FEAT_DIM, HID, 2, 0.0)
    lstm = EpisodeLSTM(graph_dim=HID, lstm_hidden=HID, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=HID, context_dim=HID)
    pol.load_state_dict(torch.load(LSTM_CKPT, map_location=device, weights_only=True))
    return pol.to(device).eval()

def _load_imrl(device):
    enc  = GraphSAGEEncoder(FEAT_DIM, HID, 2, 0.0)
    pol  = JointPolicy(enc, hidden_dim=HID)
    pol.load_state_dict(torch.load(IMRL_CKPT, map_location=device, weights_only=True))
    return pol.to(device).eval()

def _edge_index(G, device):
    E = list(G.edges())
    if not E: return torch.zeros((2,0), dtype=torch.long, device=device)
    nm = {v:i for i,v in enumerate(G.nodes())}
    s = [nm[u] for u,_ in E]+[nm[v] for _,v in E]
    d = [nm[v] for _,v in E]+[nm[u] for u,_ in E]
    return torch.tensor([s,d], dtype=torch.long, device=device)


@torch.no_grad()
def eval_lstm(policy, graph, seed, device, cache, ei):
    env = RevenueEnv(graph, _make_env_cfg(seed)); env.reset()
    nodes = list(graph.nodes()); n = len(nodes)
    policy.reset_episode(device)
    rev, offered = 0.0, set()
    for _ in range(n):
        if len(offered)==n: break
        x_t   = torch.FloatTensor(compute_node_features_fast(cache, env.S, offered, env.t, 0, env)).to(device)
        avail = torch.tensor([v not in offered for v in nodes], dtype=torch.bool, device=device)
        scores, h, ctx, _ = policy.forward(x_t, ei, avail)
        idx   = int(scores.argmax()); node_v = nodes[idx]
        comb  = torch.cat([h[idx], ctx], dim=0)
        disc  = float(policy.get_discount_distribution(comb).mean.item())
        price = env._estimate_valuation(node_v) * (1.0 - disc)
        offered.add(node_v)
        if env._true_valuation(node_v) >= price:
            rev += price; env.S.add(node_v)
        env.t += 1
        policy.update_sequence_state(disc, env._true_valuation(node_v)>=price, price if env._true_valuation(node_v)>=price else 0.0)
    return rev


@torch.no_grad()
def eval_imrl(policy, graph, seed, device, cache, ei):
    env = RevenueEnv(graph, _make_env_cfg(seed)); env.reset()
    nodes = list(graph.nodes()); n = len(nodes)
    rev, offered = 0.0, set()
    for _ in range(n):
        if len(offered)==n: break
        x_t   = torch.FloatTensor(compute_node_features_fast(cache, env.S, offered, env.t, 0, env)).to(device)
        avail = torch.tensor([v not in offered for v in nodes], dtype=torch.bool, device=device)
        idx, disc, _ = policy.select_and_price(x_t, ei, avail, greedy=True)
        node_v = nodes[idx]
        price  = env._estimate_valuation(node_v) * (1.0 - disc)
        offered.add(node_v)
        if env._true_valuation(node_v) >= price:
            rev += price; env.S.add(node_v)
        env.t += 1
    return rev


def run_all(graph, seeds, device, lstm_pol, imrl_pol, cache, ei):
    imrl_r, lstm_r = [], []
    for seed in seeds:
        print(f"  seed={seed}", flush=True)
        imrl_r.append(eval_imrl(imrl_pol, graph, seed, device, cache, ei))
        lstm_r.append(eval_lstm(lstm_pol, graph, seed, device, cache, ei))
    return {
        "imrl": {"mean": float(np.mean(imrl_r)), "all": imrl_r},
        "lstm": {"mean": float(np.mean(lstm_r)), "all": lstm_r},
    }


def main():
    t0 = time.time()
    graph = load_polblogs()
    stats = polblogs_stats(graph)
    print(f"Polblogs LCC: n={stats['n']} m={stats['m']} mean_deg={stats['mean_deg']:.1f}", flush=True)
    print(f"Density: {stats['density']:.4f}", flush=True)

    print("Computing static features (approx_bc k=200)...", flush=True)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    print("Features done.", flush=True)

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    sha_lstm = _sha8(LSTM_CKPT); sha_imrl = _sha8(IMRL_CKPT)
    assert sha_lstm == LSTM_SHA, f"LSTM SHA fail: {sha_lstm}"
    print(f"SHAs: LSTM={sha_lstm} IM-RL={sha_imrl}", flush=True)

    lstm_pol = _load_lstm(device)
    imrl_pol = _load_imrl(device)
    ei       = _edge_index(graph, device)
    print("Policies loaded.", flush=True)

    print("\n--- Protocol (a): single-seed=42 ---", flush=True)
    res_a = run_all(graph, [SINGLE_SEED], device, lstm_pol, imrl_pol, cache, ei)
    for k,v in res_a.items(): print(f"  {k}: {v['mean']:.2f}", flush=True)

    print("\n--- Protocol (b): seeds 0..4 ---", flush=True)
    res_b = run_all(graph, FIVE_SEEDS, device, lstm_pol, imrl_pol, cache, ei)
    for k,v in res_b.items():
        print(f"  {k}: {v['mean']:.2f}  per_seed={[round(x,1) for x in v['all']]}", flush=True)

    l5 = res_b["lstm"]["mean"]
    print(f"\nLSTM 5-seed mean: {l5:.2f}", flush=True)

    out = {
        "graph": stats, "lstm_sha8": sha_lstm, "imrl_sha8": sha_imrl,
        "betweenness": "approx_k200",
        "ie_strategy": "SKIPPED (O(n*MC*cascade) too slow for n=1222 dense)",
        "protocol_a_seed": SINGLE_SEED,
        "protocol_b_seeds": FIVE_SEEDS,
        "protocol_a": res_a, "protocol_b": res_b,
        "wall_seconds": time.time()-t0,
    }
    os.makedirs("results/logs", exist_ok=True)
    json.dump(out, open("results/logs/polblogs_eval.json","w"), indent=2)
    print(f"Saved results/logs/polblogs_eval.json  ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    main()
