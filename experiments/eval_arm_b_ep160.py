#!/usr/bin/env python3
"""eval_arm_b_ep160.py — Standalone 5-seed eval of arm B checkpoint (ep160).

Run on server:
    python -u experiments/eval_arm_b_ep160.py 2>&1 | tee /tmp/ep160_eval.log

Evaluates: polblogs, FF_1000, Rice_FB, Modular_FF, FF_2000
Protocol:  5 seeds (0..4), BudgetRevenueEnv k=50 budget_B=15 C=0.3,
           budget_col=1.0 (unconstrained proxy), greedy action selection.
Applies Gate B rule: STRONG if polblogs>=530.4 AND FF_1000>=440.0 AND Rice>=190.0
Output:    results/logs/arm_b_ep160_eval.json
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Approximate betweenness (k=200 pivots) to avoid O(n^3) on dense graphs
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

CKPT     = "results/checkpoints/rev_gnn_lstm_densemix.pt"
LOG_OUT  = "results/logs/arm_b_ep160_eval.json"
C        = 0.3
K_EVAL   = 50          # budget_B = K_EVAL * C = 15
WEIGHT_HIGH = 2.0
N_MC     = 5
SEEDS    = list(range(5))

# Frozen reference column (arm_b ep80 Gate B values for comparison)
FROZEN_REF = {
    "polblogs":   374.2,
    "FF_1000":    448.6,
    "Rice_FB":    214.1,
    "Modular_FF": 414.4,
    "FF_2000":    915.0,
}
EP80_REF = {
    "polblogs":   662.9,
    "FF_1000":    446.8,
    "Rice_FB":    216.6,
    "Modular_FF": 221.2,
    "FF_2000":    872.5,
}


def _sha8(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:8]


def _load_policy(device):
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(CKPT, map_location=device, weights_only=True)
    if "policy_state_dict" in sd:   sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd:  sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()


def _edge_index(G, device):
    edges = list(G.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    nmap = {v: i for i, v in enumerate(G.nodes())}
    src = [nmap[u] for u, _ in edges] + [nmap[v] for _, v in edges]
    dst = [nmap[v] for _, v in edges] + [nmap[u] for u, _ in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _feat(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    # Budget col clamped to 1.0 (unconstrained proxy — same as Gate B eval)
    return np.concatenate([base, np.ones((cache["n"], 1), dtype=np.float32)], axis=1)


def _avail(env, n, device):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for i in env.available_nodes:
        m[i] = True
    return m


@torch.no_grad()
def eval_episode(policy, G, cache, ei, seed, device):
    cfg = BudgetEnvConfig(budget_B=K_EVAL * C, production_cost=C,
                         seed=seed, weight_high=WEIGHT_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg)
    env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device)
    rev = 0.0
    for _ in range(n):
        if not env.available_nodes or env._check_bankrupt():
            break
        x  = torch.FloatTensor(_feat(cache, env, K_EVAL)).to(device)
        av = _avail(env, n, device)
        if not av.any():
            break
        sc, h, ctx, _ = policy.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(policy.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            rev += info["offered_price"]
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step", 0.0))
        if done:
            break
    return rev


def eval_network(policy, G, device, label):
    t0 = time.time()
    cache = build_graph_feature_cache(G, compute_static_features(G))
    ei    = _edge_index(G, device)
    revs  = []
    for s in SEEDS:
        r = eval_episode(policy, G, cache, ei, s, device)
        revs.append(r)
        print(f"    seed={s} rev={r:.1f}", flush=True)
    mean = float(np.mean(revs))
    print(f"  {label} MEAN={mean:.1f}  ({time.time()-t0:.0f}s total)\n", flush=True)
    return mean, revs


def _gate_b(pb, ff, rice):
    if pb >= 530.4 and ff >= 440.0 and rice >= 190.0:
        return "STRONG"
    if pb >= 420.0 and ff >= 440.0 and rice >= 190.0:
        return "PARTIAL"
    return "FAIL"


def main():
    t_start = time.time()

    # Device: use GPU if available (server), else CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ep160-eval] device={device}", flush=True)

    if not os.path.exists(CKPT):
        print(f"ERROR: checkpoint not found: {CKPT}", file=sys.stderr)
        sys.exit(1)

    sha = _sha8(CKPT)
    print(f"[ep160-eval] checkpoint sha8={sha}  ({CKPT})", flush=True)

    # Load networks
    print("\n[ep160-eval] Loading networks...", flush=True)
    networks = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250, 250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }
    for name, G in networks.items():
        print(f"  {name}: n={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)

    # Load policy
    print(f"\n[ep160-eval] Loading policy...", flush=True)
    policy = _load_policy(device)

    # Evaluate each network
    results = {}
    for name, G in networks.items():
        print(f"\n[ep160-eval] === {name} ===", flush=True)
        mean, revs = eval_network(policy, G, device, name)
        results[name] = {"mean": mean, "seeds": revs}

    # Print comparison table
    wall = time.time() - t_start
    print(f"\n{'─'*72}")
    print(f"{'network':<14} | {'frozen_ref':>10} | {'ep80':>8} | {'ep160':>8} | {'vs ep80':>8}")
    print(f"{'─'*72}")
    for name in ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]:
        ep160 = results[name]["mean"]
        ref   = FROZEN_REF[name]
        ep80  = EP80_REF[name]
        diff  = ep160 - ep80
        print(f"{name:<14} | {ref:>10.1f} | {ep80:>8.1f} | {ep160:>8.1f} | {diff:>+8.1f}")
    print(f"{'─'*72}")

    # Gate B verdict
    pb  = results["polblogs"]["mean"]
    ff  = results["FF_1000"]["mean"]
    ri  = results["Rice_FB"]["mean"]
    verdict = _gate_b(pb, ff, ri)
    print(f"\nGATE B ep160: {verdict}")
    print(f"  polblogs={pb:.1f} (floor>=530.4)  FF_1000={ff:.1f} (floor>=440.0)  Rice={ri:.1f} (floor>=190.0)")
    print(f"\nWall time: {wall:.0f}s  ({wall/60:.1f} min)")

    # Save JSON
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "checkpoint": CKPT,
        "sha8": sha,
        "epoch": 160,
        "seeds": SEEDS,
        "k_eval": K_EVAL,
        "results": results,
        "gate_b": verdict,
        "ep80_ref": EP80_REF,
        "frozen_ref": FROZEN_REF,
        "wall_s": wall,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {LOG_OUT}")


if __name__ == "__main__":
    main()
