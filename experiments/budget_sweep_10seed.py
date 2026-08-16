#!/usr/bin/env python3
"""
experiments/budget_sweep_10seed.py  — Block A (reviewer M2)
===========================================================
Budget sweep with 10 weight seeds (0..9) for 4 methods:
  IE+Budget, Greedy+Budget, Cal-DP (obs-v2/v3 composite), Rev-GNN-LSTM (arm_b FF+BA)

Protocol: BudgetRevenueEnv, c=0.3, B=k*c, k in [5,10,15,20,30,40], 5 networks.
Single learned policy: rev_gnn_lstm_densemix.pt (FF+BA), asserts sha=0b549f93.
One shard per network; safe for parallel execution.

Writes: results/logs/budget_10s_{NET}.json
Merge + paired tests: python experiments/merge_budget_10seed.py
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
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
from src.evaluation.budget_baselines import (
    greedy_discount_budget, _make_env,
)
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget

try:
    from src.evaluation.ie_budget import ie_strategy_budget
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget  # fallback

from src.utils.helpers import graph_to_pyg_data, get_available_mask, set_seed

# ── Protocol ──────────────────────────────────────────────────────────────────
K_VALUES    = [5, 10, 15, 20, 30, 40]
C           = 0.3
N_TRIALS    = 10      # 10 weight seeds (seeds 0..9)
SEEDS       = list(range(N_TRIALS))
N_SIMS      = 5       # 5×5=25k obs for Cal-DP (frozen budget)
W_HIGH      = 1.0

NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]

# ── Checkpoints ───────────────────────────────────────────────────────────────
CKPT_DIR   = "results/checkpoints"
ARM_B_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
ARM_B_SHA  = "0b549f93"   # standing convention: FF+BA final


def _sha8(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def load_arm_b(device):
    sha = _sha8(ARM_B_CKPT)
    assert sha == ARM_B_SHA, f"arm_b sha mismatch: got {sha}, want {ARM_B_SHA}"
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(input_size=3, hidden_size=64)
    pol  = SequentialJointPolicy(encoder=enc, episode_rnn=lstm,
                                 n_nodes_max=2500, n_tiers=5)
    pol.load_state_dict(torch.load(ARM_B_CKPT, map_location="cpu"))
    pol.eval()
    pol.to(device)
    return pol


def load_graph(net: str):
    if net == "polblogs":     return load_polblogs()
    if net == "FF_1000":      return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":      return load_rice_facebook()
    if net == "Modular_FF":   return generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
    if net == "FF_2000":      return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def _eval_arm_b_k(pol, graph, B, n_trials, device):
    """Run arm_b for n_trials seeds; returns list of revenues."""
    static  = compute_static_features(graph)
    cache   = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, weight_high=W_HIGH)
    revs = []
    for seed in range(n_trials):
        set_seed(seed)
        env = BudgetRevenueEnv(graph, cfg.__class__(budget_B=B, production_cost=C,
                                                     seed=seed, weight_high=W_HIGH))
        env.reset()
        pol.reset_episode(device)
        with torch.no_grad():
            for _ in range(n):
                avail = env.available_nodes
                if not avail:
                    break
                feats = compute_node_features_fast(
                    cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                    t=env.t, k=n, env=env)
                data = graph_to_pyg_data(graph, feats, device)
                mask = get_available_mask(n, frozenset(env.offered), nodes, device)
                nidx, disc, _ = pol.select_and_price(data.x, data.edge_index,
                                                      mask, greedy=True)
                if nidx not in avail:
                    nidx = avail[0]
                _, rew, done, _ = env.step(nidx, disc)
                pol.update_sequence_state(disc, rew > 0, float(rew))
                if done:
                    break
        revs.append(float(env.total_revenue))
    return revs


def run_network(net: str, out_path: str, device):
    graph = load_graph(net)
    cfg   = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH)
    print(f"\n=== {net} ===", flush=True)
    arm_b = load_arm_b(device)

    results = {}
    for k in K_VALUES:
        B  = k * C
        t0 = time.time()

        # IE+Budget
        r_ie = ie_strategy_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        ie_raw = r_ie.get("total_revenue", r_ie.get("revenue", {})).get("all", [
            r_ie.get("total_revenue", r_ie.get("revenue", {0: 0})).get("mean", 0.0)
        ] * N_TRIALS)

        # Greedy+Budget
        r_gd = greedy_discount_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS)
        gd_raw = r_gd.get("total_revenue", r_gd.get("revenue", {})).get("all", [
            r_gd.get("total_revenue", r_gd.get("revenue", {0: 0})).get("mean", 0.0)
        ] * N_TRIALS)

        # Cal-DP obs composite
        r2 = dp_calibrated_v2_obs_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)
        r3 = dp_calibrated_v3_obs_budget(graph, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)
        v2_raw = r2.get("revenue", {}).get("all", [r2.get("revenue",{}).get("mean",0.)]*N_TRIALS)
        v3_raw = r3.get("revenue", {}).get("all", [r3.get("revenue",{}).get("mean",0.)]*N_TRIALS)
        cdp_raw = [max(a, b) for a, b in zip(v2_raw, v3_raw)]

        # arm_b
        ab_raw = _eval_arm_b_k(arm_b, graph, B, N_TRIALS, device)

        def _stats(vals):
            a = np.array(vals, dtype=float)
            return {"mean": round(float(a.mean()),3), "std": round(float(a.std()),3),
                    "all": [round(v,3) for v in vals]}

        results[k] = {
            "IE+Budget":        _stats(ie_raw[:N_TRIALS]),
            "Greedy+Budget":    _stats(gd_raw[:N_TRIALS]),
            "Cal-DP":           _stats(cdp_raw),
            "Rev-GNN-LSTM":     _stats(ab_raw),
        }
        elapsed = time.time() - t0
        print(f"  k={k:2d}  IE={results[k]['IE+Budget']['mean']:.1f}"
              f"  GD={results[k]['Greedy+Budget']['mean']:.1f}"
              f"  CDP={results[k]['Cal-DP']['mean']:.1f}"
              f"  GNN={results[k]['Rev-GNN-LSTM']['mean']:.1f}"
              f"  ({elapsed:.0f}s)", flush=True)

    shard = {"network": net, "n_trials": N_TRIALS, "seeds": SEEDS,
             "shas": {"arm_b": ARM_B_SHA}, "results": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(shard, f, indent=2)
    print(f"Saved → {out_path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS_ALL)
    ap.add_argument("--out-dir",  default="results/logs")
    ap.add_argument("--gpu",      type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[budget_sweep_10seed] device={device}  n_trials={N_TRIALS}  sha={ARM_B_SHA}")

    for net in args.networks:
        out = os.path.join(args.out_dir, f"budget_10s_{net}.json")
        if os.path.exists(out):
            print(f"SKIP {net}: {out} already exists"); continue
        run_network(net, out, device)


if __name__ == "__main__":
    main()
