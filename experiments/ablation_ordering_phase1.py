"""
experiments/ablation_ordering_phase1.py
=======================================
Two ablations on Rev-GNN-LSTM (unconstrained, 5 seeds [0..4]):

  1. ORDERING ABLATION (4 networks):
     (a) policy selects its own buyer order (standard eval)
     (b) pricing-only: fixed degree-descending order, policy restricted
         to one node at a time so only pricing head fires

  2. PHASE-1 ABLATION (imitation-only checkpoint, 4 networks):
     Checkpoint: results/checkpoints/rev_gnn_lstm_largek_p1.pt
     If that file doesn't exist, falls back to rev_gnn_lstm_unified_ph1end.pt

Saves: results/logs/ablation_ordering_phase1.json
"""
import argparse, hashlib, json, os, sys, time
import numpy as np
import torch
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
os.chdir(_REPO)

from src.utils.helpers import (
    load_config_with_base, set_seed, graph_to_pyg_data, get_available_mask
)
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast
)
from src.evaluation.idea1_eval import load_lstm_policy, _eval_greedy_discount
from src.evaluation.baselines import _make_env
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)

SEEDS     = list(range(5))
NETS_EVAL = ["FF_1000", "FF_2000", "Modular_FF", "Rice_FB"]

LSTM_CKPT  = "results/checkpoints/rev_gnn_lstm.pt"
P1_CKPTS   = [
    "results/checkpoints/rev_gnn_lstm_largek_p1.pt",
    "results/checkpoints/rev_gnn_lstm_unified_ph1end.pt",
]

GRAPH_LOADERS = {
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
    "Modular_FF": lambda: generate_modular_forest_fire([200,300,500], 0.37, 0.32, 0.01, seed=0),
    "Rice_FB":    load_rice_facebook,
}


def sha8(path):
    h = hashlib.sha256()
    with open(path, "rb") as f: h.update(f.read())
    return h.hexdigest()[:8]


def run_ep_free_order(policy, graph, cfg, device, seed: int) -> float:
    """Standard eval: policy selects own buyer order."""
    set_seed(seed)
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        for _ in range(n):
            available = env.available_nodes
            if not available: break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available: nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            policy.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    policy.to(device)
    return float(env.total_revenue)


def run_ep_fixed_order(policy, graph, cfg, device, seed: int) -> float:
    """Pricing-only: degree-descending order; one-node mask forces pricing head only."""
    set_seed(seed)
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static  = compute_static_features(graph)
    cache   = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    # degree-descending order (same as baselines + Cal-DP)
    degree_order = sorted(range(n), key=lambda i: graph.degree(nodes[i]), reverse=True)

    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        policy.reset_episode(eval_dev)
        for pos_idx in degree_order:
            available = env.available_nodes
            if not available: break
            # expose only this node so selection head is forced to it
            offered_plus_rest = frozenset(env.offered) | (
                frozenset(range(n)) - {pos_idx})
            mask = get_available_mask(n, offered_plus_rest, nodes, eval_dev)
            # if this node was already offered, skip
            if nodes[pos_idx] in env.offered: continue
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            _, rew, done, _ = env.step(pos_idx, disc)  # force degree-order node
            policy.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    policy.to(device)
    return float(env.total_revenue)


def run_ep_policy(policy, graph, cfg, device, seed: int, is_lstm=True, fixed=False) -> float:
    if is_lstm:
        if fixed:
            return run_ep_fixed_order(policy, graph, cfg, device, seed)
        return run_ep_free_order(policy, graph, cfg, device, seed)
    # non-lstm (im_rl): same logic but no reset_episode / update_sequence_state
    set_seed(seed)
    eval_dev = torch.device("cpu")
    policy.to(eval_dev)
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    with torch.no_grad():
        env = _make_env(graph, cfg)
        env.reset()
        for _ in range(n):
            available = env.available_nodes
            if not available: break
            feats = compute_node_features_fast(
                cache=cache, S=frozenset(env.S), offered=frozenset(env.offered),
                t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, eval_dev)
            mask = get_available_mask(n, frozenset(env.offered), nodes, eval_dev)
            nidx, disc, _ = policy.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in available: nidx = available[0]
            _, rew, done, _ = env.step(nidx, disc)
            if done: break
    policy.to(eval_dev)
    return float(env.total_revenue)


def eval_net_5seed(policy, graph, cfg, device, is_lstm=True, fixed=False):
    revs = [run_ep_policy(policy, graph, cfg, device, s, is_lstm, fixed) for s in SEEDS]
    return float(np.mean(revs)), float(np.std(revs)), revs


def gd_5seed(graph, cfg):
    revs = []
    for s in SEEDS:
        set_seed(s)
        revs.append(float(_eval_greedy_discount(graph, cfg)))
    return float(np.mean(revs)), float(np.std(revs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/rev_gnn_lstm.yaml")
    ap.add_argument("--networks", nargs="+", default=NETS_EVAL)
    args = ap.parse_args()

    cfg    = load_config_with_base(args.config)
    device = torch.device("cpu")

    # Load main LSTM checkpoint
    lstm_sha = sha8(LSTM_CKPT)
    lstm_pol = load_lstm_policy(LSTM_CKPT, cfg, device)

    # Find Phase-1 checkpoint
    p1_path = next((p for p in P1_CKPTS if os.path.exists(p)), None)
    if p1_path:
        p1_sha  = sha8(p1_path)
        p1_pol  = load_lstm_policy(p1_path, cfg, device)
        print(f"Phase-1 ckpt: {p1_path}  sha8={p1_sha}")
    else:
        p1_pol, p1_sha = None, "not found"
        print("WARNING: no Phase-1 checkpoint found")

    results = {}
    print(f"\nMain ckpt: {LSTM_CKPT}  sha8={lstm_sha}")
    print(f"\n{'net':12s}  {'GD_mean':>8}  {'(a)free_mean±std':>18}  {'(b)fixed_mean±std':>18}  {'p1_mean±std':>18}")
    print("-" * 82)

    for net in args.networks:
        t0    = time.time()
        graph = GRAPH_LOADERS[net]()
        gd_m, gd_s = gd_5seed(graph, cfg)
        a_m, a_s, a_r = eval_net_5seed(lstm_pol, graph, cfg, device, True, False)
        b_m, b_s, b_r = eval_net_5seed(lstm_pol, graph, cfg, device, True, True)
        if p1_pol:
            p1_m, p1_s, p1_r = eval_net_5seed(p1_pol, graph, cfg, device, True, False)
        else:
            p1_m, p1_s, p1_r = float("nan"), 0.0, []
        print(f"  {net:10s}  {gd_m:>8.1f}  {a_m:>8.1f}±{a_s:.1f}          "
              f"{b_m:>8.1f}±{b_s:.1f}          {p1_m:>8.1f}±{p1_s:.1f}   [{time.time()-t0:.0f}s]")
        results[net] = {
            "greedy_disc": {"mean": round(gd_m,2), "std": round(gd_s,2)},
            "(a)free_order": {"mean": round(a_m,2), "std": round(a_s,2), "per_seed": [round(r,2) for r in a_r]},
            "(b)fixed_order": {"mean": round(b_m,2), "std": round(b_s,2), "per_seed": [round(r,2) for r in b_r]},
            "phase1_imit": {"mean": round(p1_m,2) if p1_pol else None, "std": round(p1_s,2), "per_seed": [round(r,2) for r in p1_r]},
        }

    os.makedirs("results/logs", exist_ok=True)
    out = {"lstm_sha": lstm_sha, "p1_sha": p1_sha, "p1_path": p1_path, "results": results}
    with open("results/logs/ablation_ordering_phase1.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/logs/ablation_ordering_phase1.json")

    # ordering delta
    print("\nOrdering ablation delta (b)-(a):")
    for net, r in results.items():
        d = r["(b)fixed_order"]["mean"] - r["(a)free_order"]["mean"]
        print(f"  {net:12s}: {d:+.1f}")

    if p1_pol:
        print("\nPhase-1 vs Final LSTM delta:")
        for net, r in results.items():
            d = r["phase1_imit"]["mean"] - r["(a)free_order"]["mean"]
            print(f"  {net:12s}: {d:+.1f}")


if __name__ == "__main__":
    main()
