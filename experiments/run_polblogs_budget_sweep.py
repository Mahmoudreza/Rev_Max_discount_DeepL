#!/usr/bin/env python3
"""run_polblogs_budget_sweep.py — Idea 3 budget sweep on Polblogs (zero-shot).

Methods:
  1. Greedy+Budget (frozen executor)
  2. Cal-DP composite (v2 + v3, fresh calibration n_sim=30)
  3. Learned composite: unified-gatefail for k<16, largek for k>=16
  4. lstm_v1 (rev_gnn_lstm_budget_v1.pt)

Output: results/logs/polblogs_budget_sweep.json
"""
import sys, os, json, hashlib, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import networkx as nx
from types import SimpleNamespace

# Approximate betweenness (k=200 pivots) for dense polblogs graph
_orig_bc = nx.betweenness_centrality
def _approx_bc(G, normalized=True, **kwargs):
    k_pivots = min(200, G.number_of_nodes())
    return _orig_bc(G, k=k_pivots, normalized=normalized, **kwargs)
nx.betweenness_centrality = _approx_bc

from src.env.polblogs_loader import load_polblogs
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.revenue_env import RevenueEnv

# Faster get_current_influence: iterate neighbors (O(deg)) not S (O(|S|))
def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.0
    tw = sum(self._link_weights.get((node,n),0.0) for n in nb)
    if tw==0: return 0.0
    return sum(self._link_weights.get((node,n),0.0) for n in nb if n in self.S)/tw
RevenueEnv.get_current_influence = _fast_gci
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)

# ── Constants ────────────────────────────────────────────────────────────────
CKPT_DIR      = "results/checkpoints"
# SELECTED unified = gatefail (sha8=00071438); reproduces frozen 369.6 at k=20
UNIFIED_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified_gatefail.pt")
LARGEK_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_largek.pt")
LSTMV1_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_budget_v1.pt")
UNIFIED_SHA   = "00071438"
LARGEK_SHA    = "3033620a"

C         = 0.3
B_MAX     = 12.0   # 40 * C
K_LIST    = [1, 3, 5, 10, 15, 20, 30, 40]
SEEDS     = [42, 123, 7]
N_TRIALS  = 3
BOUNDARY  = 16    # k<16: unified; k>=16: largek
WEIGHT_HIGH = 2.0
N_SIM     = 30    # Cal-DP calibration simulations


def _sha8(path):
    return hashlib.sha256(open(path,"rb").read()).hexdigest()[:8]


def _load_budget_policy(ckpt, in_dim, device):
    enc  = GraphSAGEEncoder(in_dim, 64, 2, 0.0)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    pol.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    return pol.to(device).eval()


def _edge_index(G, device):
    edges = list(G.edges())
    if not edges:
        return torch.zeros((2,0), dtype=torch.long, device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges] + [nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges] + [nmap[u] for u,_ in edges]
    return torch.tensor([src,dst], dtype=torch.long, device=device)


def _make_env(graph, k, seed):
    B = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=WEIGHT_HIGH)
    return BudgetRevenueEnv(graph, cfg)


def _unified_feat(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    n    = cache["n"]
    col  = np.full((n, 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, col], axis=1)


@torch.no_grad()
def eval_budget_policy(policy, graph, cache, ei, k, seed, device):
    env = _make_env(graph, k, seed)
    env.reset()
    n   = graph.number_of_nodes()
    policy.reset_episode(device)
    revenue = 0.0
    for _ in range(n):
        if not env.available_nodes:
            break
        if env._check_bankrupt():
            break
        x_np  = _unified_feat(cache, env, k)
        x_t   = torch.FloatTensor(x_np).to(device)
        avail = torch.tensor([v not in env.offered and v not in env.S
                               for v in graph.nodes()], dtype=torch.bool, device=device)
        if not avail.any():
            break
        scores, h, ctx, _ = policy.forward(x_t, ei, avail)
        node_idx = int(scores.argmax().item())
        comb   = torch.cat([h[node_idx], ctx], dim=0)
        beta   = policy.get_discount_distribution(comb)
        disc   = float(beta.mean.item())
        est_v  = env._estimate_valuation(env.nodes[node_idx])
        offered_price = est_v * (1.0 - disc)
        if env.B - C + offered_price < -1e-9:
            env.offered.add(env.nodes[node_idx])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(disc, False, 0.0)
            continue
        obs, reward, done, info = env.step(node_idx, disc)
        if info["accepted"]:
            revenue += info["offered_price"]
        policy.update_sequence_state(disc, info["accepted"],
                                     info.get("revenue_step", 0.0))
        if done:
            break
    return revenue


def eval_greedy_budget(graph, k, seeds):
    """Greedy+Budget mean over seeds."""
    from src.env.budget_revenue_env import BudgetEnvConfig as BC
    cfg = BC(budget_B=k*C, production_cost=C, seed=seeds[0], weight_high=WEIGHT_HIGH)
    r = greedy_discount_budget(graph, k*C, C, n_trials=N_TRIALS, weight_high=WEIGHT_HIGH)
    return float(r.get("revenue", {}).get("mean", r.get("mean", 0.0)))


def eval_caldp(graph, k):
    """Cal-DP composite (max of v2, v3) for given k."""
    B   = k * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0, weight_high=WEIGHT_HIGH)
    try:
        r2 = dp_calibrated_v2_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIM)
        m2 = float(r2.get("revenue", {}).get("mean", r2.get("mean", 0.0)))
    except Exception as e:
        print(f"  Cal-DP v2 k={k} error: {e}")
        m2 = 0.0
    try:
        r3 = dp_calibrated_v3_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIM)
        m3 = float(r3.get("revenue", {}).get("mean", r3.get("mean", 0.0)))
    except Exception as e:
        print(f"  Cal-DP v3 k={k} error: {e}")
        m3 = 0.0
    return max(m2, m3), "v3" if m3 >= m2 else "v2"


def main():
    t0 = time.time()
    graph  = load_polblogs()
    n_nodes = graph.number_of_nodes()
    print(f"Polblogs n={n_nodes} m={graph.number_of_edges()}")

    device = torch.device("cpu")  # CPU faster for n=1222 (low MPS dispatch latency)
    print(f"Device: {device}")

    # Verify checkpoint SHAs
    us = _sha8(UNIFIED_CKPT); ls = _sha8(LARGEK_CKPT); v1s = _sha8(LSTMV1_CKPT)
    assert us == UNIFIED_SHA, f"Unified SHA fail: {us}"
    assert ls == LARGEK_SHA,  f"Largek SHA fail:  {ls}"
    print(f"SHAs OK: unified={us} largek={ls} lstmv1={v1s}")

    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    ei     = _edge_index(graph, device)

    print("Loading policies...")
    unified_pol = _load_budget_policy(UNIFIED_CKPT, 21, device)
    largek_pol  = _load_budget_policy(LARGEK_CKPT, 21, device)
    lstmv1_pol  = _load_budget_policy(LSTMV1_CKPT, 21, device)

    results = {}
    print(f"\n{'k':>3} | {'Greedy+B':>9} | {'CalDP':>7} | {'Learned':>8} | {'lstm_v1':>8} | bkr")
    print("-" * 60)
    for k in K_LIST:
        row = {}

        # 1. Greedy+Budget
        try:
            r_gr = greedy_discount_budget(graph, k*C, C, n_trials=N_TRIALS, weight_high=WEIGHT_HIGH)
            g_mean = float(r_gr.get("revenue", {}).get("mean", r_gr.get("mean", 0.0)))
        except Exception as e:
            print(f"  Greedy+B k={k} error: {e}")
            g_mean = float("nan")
        row["greedy_b"] = g_mean

        # 2. Cal-DP composite
        cdp, cdp_ver = eval_caldp(graph, k)
        row["caldp"] = cdp
        row["caldp_ver"] = cdp_ver

        # 3. Learned composite
        pol_k = largek_pol if k >= BOUNDARY else unified_pol
        learned_revs = []
        for seed in SEEDS:
            r = eval_budget_policy(pol_k, graph, cache, ei, k, seed, device)
            learned_revs.append(r)
        row["learned"] = float(np.mean(learned_revs))

        # 4. lstm_v1
        v1_revs = []
        for seed in SEEDS:
            r = eval_budget_policy(lstmv1_pol, graph, cache, ei, k, seed, device)
            v1_revs.append(r)
        row["lstm_v1"] = float(np.mean(v1_revs))

        results[k] = row
        print(f"{k:>3} | {g_mean:>9.1f} | {cdp:>7.1f} | {row['learned']:>8.1f} | {row['lstm_v1']:>8.1f} | ({cdp_ver})")

    # Bankruptcy summary
    print("\nBankruptcy: not separately tracked (SKIP protocol used everywhere)")

    out = {
        "seeds": SEEDS,
        "n_sim": N_SIM,
        "n_trials": N_TRIALS,
        "boundary": BOUNDARY,
        "unified_sha8": us,
        "largek_sha8": ls,
        "lstmv1_sha8": v1s,
        "results": {str(k): v for k, v in results.items()},
        "wall_seconds": time.time() - t0,
    }
    os.makedirs("results/logs", exist_ok=True)
    json.dump(out, open("results/logs/polblogs_budget_sweep.json", "w"), indent=2)
    print(f"\nSaved → results/logs/polblogs_budget_sweep.json  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
