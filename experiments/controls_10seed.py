#!/usr/bin/env python3
"""
experiments/controls_10seed.py — Block C (C1 + C2 + C3)
=========================================================
Budget protocol, 10 seeds, 5 networks, k=[5,10,15,20,30,40].

C1. Price-floor Greedy: Greedy+Budget but never posts price < c (skips instead).
    NEW implementation — does not modify existing greedy_discount_budget.

C2. Myopic Cal-DP: Cal-DP with lookahead removed. At each step picks tau
    that maximises A[d][ib][tau] * price_tau using the same calibrated
    tables. No J value function.
    Isolates CALIBRATION vs PLANNING contribution.

C3. PageRank ordering + policy pricing head:
    Sort buyers by PageRank desc; use policy pricing for each (no selection).
    Isolates learned selection vs structural heuristic.

Protocol: BudgetRevenueEnv, c=0.3, B=k*c, arm_b sha 0b549f93.
Writes per-network shards; merge with: cat results/logs/ctrl_*.json | python merge_controls.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)
from src.utils.graph_to_pyg import graph_to_pyg_data
try:
    from src.utils.helpers import get_available_mask, set_seed
except ImportError:
    from src.utils.features import get_available_mask, set_seed
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated import _deg_class, _infl_bucket

K_VALUES = [5, 10, 15, 20, 30, 40]
C = 0.3; N_TRIALS = 10; N_SIMS = 5; W_HIGH = 1.0
NETWORKS = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
CKPT_DIR = "results/checkpoints"
ARM_B_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
ARM_B_SHA  = "0b549f93"
TIERS = (1.0, 0.8, 0.5, 0.2, 0.0)


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]


def load_arm_b(device):
    assert _sha8(ARM_B_CKPT) == ARM_B_SHA
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(input_size=3, hidden_size=64)
    pol  = SequentialJointPolicy(encoder=enc, episode_rnn=lstm, n_nodes_max=2500, n_tiers=5)
    pol.load_state_dict(torch.load(ARM_B_CKPT, map_location="cpu"))
    pol.eval(); pol.to(device); return pol


def load_graph(net):
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000,0.37,0.32,seed=0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    if net == "FF_2000":    return generate_forest_fire(2000,0.37,0.32,seed=1)
    raise ValueError(net)


# ── C1: Price-floor Greedy ────────────────────────────────────────────────────
def _c1_greedy_floor_ep(graph, B, seed):
    """Greedy+Budget + price floor: skip if price < c."""
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    n = graph.number_of_nodes(); nodes = list(graph.nodes())
    ordering = sorted(nodes, key=lambda v: graph.degree(v), reverse=True)
    for node in ordering:
        if node in env.offered: continue
        nidx = env.node_to_idx[node]
        est_val = env._estimate_valuation(node)
        # greedy: use standard tier rule but skip if price < c
        # tier=0.5 gives price = 0.5*est_val; skip if 0.5*est_val < c
        for disc in [0.0, 0.2, 0.5]:  # start from most aggressive
            price = est_val * (1.0 - disc)
            if price >= C:  # floor satisfied
                break
        else:
            continue  # all prices below floor — skip
        if env._check_bankrupt(): break
        _, _, done, _ = env.step(nidx, disc)
        if done: break
    return float(env.total_revenue)


def c1_floor_k(graph, B, n_trials):
    return [_c1_greedy_floor_ep(graph, B, s) for s in range(n_trials)]


# ── C2: Myopic Cal-DP ────────────────────────────────────────────────────────
def c2_myopic_ep(graph, B, seed, V, A, P, cb, ib):
    """Cal-DP without lookahead: at each step max over tau of A*price."""
    n = graph.number_of_nodes(); nodes = list(graph.nodes())
    ordering = sorted(nodes, key=lambda v: graph.degree(v), reverse=True)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    env = BudgetRevenueEnv(graph, cfg); env.reset()
    for pos, node in enumerate(ordering):
        if node in env.offered: continue
        nidx = env.node_to_idx[node]
        if env._check_bankrupt(): break
        deg = graph.degree(node)
        cls = _deg_class(deg, cb)
        # influence bucket: use degree-proxy or observed influence
        # use position bucket heuristic: pos/n maps to infl bucket
        n_buckets = A.shape[1]
        ib_idx = min(int((pos / n) * n_buckets), n_buckets - 1)
        avg_val = float(V[cls, ib_idx]) if V[cls, ib_idx] > 0 else 1.0
        # pick tau maximising A[cls][ib][tau] * price(tau) — NO J
        best_tau, best_ev = -1, -1.0
        for ti, tau in enumerate(TIERS):
            price = avg_val * (1.0 - tau)
            p_acc = float(A[cls, ib_idx, ti])
            ev = p_acc * price
            if ev > best_ev:
                best_ev = ev; best_tau = tau
        if best_tau < 0: best_tau = TIERS[2]
        _, _, done, _ = env.step(nidx, best_tau)
        if done: break
    return float(env.total_revenue)


def c2_myopic_k(graph, B, n_trials, V, A, P, cb, ib):
    return [c2_myopic_ep(graph, B, s, V, A, P, cb, ib) for s in range(n_trials)]


# ── C3: PageRank ordering + policy pricing ────────────────────────────────────
def c3_pagerank_ep(pol, graph, B, seed, device):
    """PageRank-sorted order, policy pricing head."""
    set_seed(seed)
    pr = nx.pagerank(graph)
    nodes = list(graph.nodes())
    n = len(nodes)
    pr_order = sorted(range(n), key=lambda i: pr[nodes[i]], reverse=True)
    static = compute_static_features(graph); cache = build_graph_feature_cache(graph, static)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    env = BudgetRevenueEnv(graph, cfg); env.reset(); pol.reset_episode(device)
    with torch.no_grad():
        for pos in pr_order:
            if not env.available_nodes or env._check_bankrupt(): break
            if nodes[pos] in env.offered: continue
            mask = get_available_mask(n, frozenset(env.offered)|(frozenset(range(n))-{pos}), nodes, device)
            feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                offered=frozenset(env.offered), t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            _, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)
            _, rew, done, _ = env.step(pos, disc)
            pol.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    return float(env.total_revenue)


def c3_pagerank_k(pol, graph, B, n_trials, device):
    return [c3_pagerank_ep(pol, graph, B, s, device) for s in range(n_trials)]


def _stats(vals):
    a = np.array(vals, float)
    return {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2),
            "all": [round(v,2) for v in vals]}


def run_network(net, out_dir, device):
    graph = load_graph(net)
    cfg0  = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH)
    arm_b = load_arm_b(device)
    V, A, P, cb, ib = calibrate_v2_obs_table(graph, cfg0, n_sims=N_SIMS, seed=0)
    print(f"\n=== {net} ===", flush=True)
    results = {}
    for k in K_VALUES:
        B = k * C; t0 = time.time()
        r_c1 = c1_floor_k(graph, B, N_TRIALS)
        r_c2 = c2_myopic_k(graph, B, N_TRIALS, V, A, P, cb, ib)
        r_c3 = c3_pagerank_k(arm_b, graph, B, N_TRIALS, device)
        results[k] = {"C1_floor_greedy": _stats(r_c1),
                      "C2_myopic_caldp":  _stats(r_c2),
                      "C3_pagerank_pol":  _stats(r_c3)}
        print(f"  k={k:2d}  C1={results[k]['C1_floor_greedy']['mean']:.1f}"
              f"  C2={results[k]['C2_myopic_caldp']['mean']:.1f}"
              f"  C3={results[k]['C3_pagerank_pol']['mean']:.1f}"
              f"  ({time.time()-t0:.0f}s)", flush=True)
    shard = {"network": net, "n_trials": N_TRIALS, "shas": {"arm_b": ARM_B_SHA},
             "results": results}
    out = os.path.join(out_dir, f"ctrl_{net}.json")
    os.makedirs(out_dir, exist_ok=True)
    with open(out, "w") as f: json.dump(shard, f, indent=2)
    print(f"Saved → {out}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS)
    ap.add_argument("--out-dir", default="results/logs")
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    for net in args.networks:
        run_network(net, args.out_dir, device)


if __name__ == "__main__":
    main()
