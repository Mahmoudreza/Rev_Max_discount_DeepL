#!/usr/bin/env python3
"""
experiments/ablation_unc_10seed.py — Block B (B1 + B2)
=======================================================
Unconstrained protocol, 10 seeds, all 5 networks.

B1. ORDERING ABLATION:
    (a) policy selects its own buyer order (standard eval path)
    (b) pricing head applied to fixed degree-descending order (selection head disabled)
    Delta (b)-(a) determines whether "who to target" claim is supported.

B2. IMITATION-ONLY ABLATION:
    Phase-1 checkpoint (before REINFORCE) vs final policy vs Greedy-Discount.
    Phase-1 candidate: rev_gnn_lstm_largek_p1.pt or rev_gnn_lstm_unified_ph1end.pt.
    If neither exists, B2 = "no Phase-1 checkpoint found".

B3 (reported inline): No Gaussian-head checkpoint found in repo; B3 claim dropped.

Protocol: RevenueEnv (unconstrained, no cost, no wallet), single graph per network,
          10 seeds [0..9], arm_b FF+BA (sha 0b549f93).

Writes: results/logs/ablation_unc_10seed.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.revenue_env import RevenueEnv
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
from src.utils.helpers import graph_to_pyg_data, get_available_mask, set_seed

# ── Protocol ──────────────────────────────────────────────────────────────────
SEEDS        = list(range(10))
NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
CKPT_DIR     = "results/checkpoints"
ARM_B_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
ARM_B_SHA    = "0b549f93"
P1_CKPTS     = [
    os.path.join(CKPT_DIR, "rev_gnn_lstm_largek_p1.pt"),
    os.path.join(CKPT_DIR, "rev_gnn_lstm_unified_ph1end.pt"),
]


def _sha8(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def _load_lstm_policy(ckpt, device, in_dim=21):
    enc  = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(input_size=3, hidden_size=64)
    pol  = SequentialJointPolicy(encoder=enc, episode_rnn=lstm,
                                 n_nodes_max=2500, n_tiers=5)
    pol.load_state_dict(torch.load(ckpt, map_location="cpu"))
    pol.eval(); pol.to(device)
    return pol


def load_graph(net):
    if net == "polblogs":   return load_polblogs()
    if net == "FF_1000":    return generate_forest_fire(1000, 0.37, 0.32, seed=0)
    if net == "Rice_FB":    return load_rice_facebook()
    if net == "Modular_FF": return generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    if net == "FF_2000":    return generate_forest_fire(2000, 0.37, 0.32, seed=1)
    raise ValueError(net)


def run_ep_free(pol, graph, seed, device):
    """Standard: policy selects own order."""
    set_seed(seed)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    env = RevenueEnv(graph, seed=seed)
    env.reset(); pol.reset_episode(device)
    with torch.no_grad():
        for _ in range(n):
            avail = env.available_nodes
            if not avail: break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            nidx, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in avail: nidx = avail[0]
            _, rew, done, _ = env.step(nidx, disc)
            pol.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    return float(env.total_revenue)


def run_ep_fixed(pol, graph, seed, device):
    """Fixed degree order: pricing only."""
    set_seed(seed)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    deg_ord = sorted(range(n), key=lambda i: graph.degree(nodes[i]), reverse=True)
    env = RevenueEnv(graph, seed=seed)
    env.reset(); pol.reset_episode(device)
    with torch.no_grad():
        for pos in deg_ord:
            if not env.available_nodes: break
            if nodes[pos] in env.offered: continue
            mask = get_available_mask(n, frozenset(env.offered) | (
                frozenset(range(n)) - {pos}), nodes, device)
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            _, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)
            _, rew, done, _ = env.step(pos, disc)
            pol.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    return float(env.total_revenue)


def greedy_disc_ep(graph, seed):
    """Greedy-Discount unconstrained baseline."""
    set_seed(seed)
    from src.evaluation.baselines import greedy_discount
    return float(greedy_discount(graph, seed=seed))


def _stats(vals):
    a = np.array(vals, float)
    return {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2),
            "all": [round(v,2) for v in vals]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS_ALL)
    ap.add_argument("--gpu", type=int, default=0)
    args = ap.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    sha = _sha8(ARM_B_CKPT)
    assert sha == ARM_B_SHA, f"sha mismatch {sha}"
    pol = _load_lstm_policy(ARM_B_CKPT, device)
    print(f"arm_b sha={sha}  device={device}  seeds={SEEDS}")

    p1_path = next((p for p in P1_CKPTS if os.path.exists(p)), None)
    if p1_path:
        p1_sha = _sha8(p1_path); p1_pol = _load_lstm_policy(p1_path, device)
        print(f"Phase-1 ckpt={p1_path}  sha8={p1_sha}")
    else:
        p1_pol = None; p1_sha = "not_found"
        print("B2: no Phase-1 checkpoint found")

    results = {}
    print(f"\n{'net':12s}  {'GD':>7}  {'(a)free':>9}  {'(b)fixed':>9}  delta  {'p1':>9}")
    print("-" * 65)
    for net in args.networks:
        t0 = time.time()
        graph = load_graph(net)
        gd   = [greedy_disc_ep(graph, s) for s in SEEDS]
        free = [run_ep_free(pol, graph, s, device) for s in SEEDS]
        fix  = [run_ep_fixed(pol, graph, s, device) for s in SEEDS]
        p1   = [run_ep_free(p1_pol, graph, s, device) for s in SEEDS] if p1_pol else []
        d_ab = round(np.mean(fix) - np.mean(free), 1)
        print(f"  {net:10s}  {np.mean(gd):>7.1f}  {np.mean(free):>9.1f}  "
              f"{np.mean(fix):>9.1f}  {d_ab:+5.1f}  "
              f"{'%7.1f' % np.mean(p1) if p1 else '    n/a':>9}   [{time.time()-t0:.0f}s]")
        results[net] = {
            "greedy_disc":    _stats(gd),
            "(a)free_order":  _stats(free),
            "(b)fixed_order": _stats(fix),
            "phase1_imit":    _stats(p1) if p1 else None,
        }

    os.makedirs("results/logs", exist_ok=True)
    out = {"shas": {"arm_b": sha, "p1": p1_sha}, "p1_path": p1_path,
           "seeds": SEEDS, "results": results}
    with open("results/logs/ablation_unc_10seed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved → results/logs/ablation_unc_10seed.json")
    print("\nB3: No Gaussian-head checkpoint found (rev_gnn_lstm_tc.pt is temperature-calibration"
          " variant, not Gaussian-head). B3 claim dropped.")
    print("\nOrdering delta (b)-(a)  [negative = own order is better]:")
    for net, r in results.items():
        d = r["(b)fixed_order"]["mean"] - r["(a)free_order"]["mean"]
        print(f"  {net}: {d:+.1f}")
    if any(r["phase1_imit"] for r in results.values()):
        print("\nPhase-1 vs Final:")
        for net, r in results.items():
            if r["phase1_imit"]:
                d = r["phase1_imit"]["mean"] - r["(a)free_order"]["mean"]
                print(f"  {net}: {d:+.1f}")


if __name__ == "__main__":
    main()
