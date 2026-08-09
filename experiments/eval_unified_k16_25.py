#!/usr/bin/env python3
"""eval_unified_k16_25.py — Eval rev_gnn_lstm_unified_gatefail.pt at k=[16,20,25].

Harness check: k=20 must reproduce 369.6 ± 1.
Reports k=16, k=20, k=25 means with the same graph/seeds as unified_sweep.json.
"""
import os, sys, json, hashlib, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast

C      = 0.3
B_MAX  = 12.0
SEEDS  = [42, 123, 7]
K_EVAL = [16, 20, 25]
CKPT   = "results/checkpoints/rev_gnn_lstm_unified_gatefail.pt"
EXPECTED_SHA = "00071438c0ae8dbea97ae07b28cd4f6017e7ccc0eeb72cf2b742a8eabc4e2c61"


def _to_edge_index(graph, device):
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    nodes = list(graph.nodes())
    nmap = {v: i for i, v in enumerate(nodes)}
    src = [nmap[e[0]] for e in edges] + [nmap[e[1]] for e in edges]
    dst = [nmap[e[1]] for e in edges] + [nmap[e[0]] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _avail_mask(env, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes:
        mask[idx] = True
    return mask


def _unified_feat(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    n    = cache["n"]
    col  = np.full((n, 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, col], axis=1)


@torch.no_grad()
def eval_one(policy, graph, k, seed, device):
    n    = graph.number_of_nodes()
    B    = k * C
    cfg  = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed)
    env  = BudgetRevenueEnv(graph, cfg)
    env.reset()
    static = compute_static_features(graph)
    cache  = build_graph_feature_cache(graph, static)
    ei     = _to_edge_index(graph, device)
    policy.reset_episode(device)
    revenue = 0.0
    bankrupt = False
    for _t in range(n):
        if not env.available_nodes:
            break
        if env._check_bankrupt():
            bankrupt = True
            break
        x_np  = _unified_feat(cache, env, k)
        x_t   = torch.FloatTensor(x_np).to(device)
        avail = _avail_mask(env, n, device)
        scores, h, ctx, _ = policy.forward(x_t, ei, avail)
        node_idx = int(scores.argmax().item())
        comb     = torch.cat([h[node_idx], ctx], dim=0)
        beta     = policy.get_discount_distribution(comb)
        discount = float(beta.mean.item())
        est_val  = env._estimate_valuation(env.nodes[node_idx])
        offered_price = est_val * (1.0 - discount)
        if env.B - C + offered_price < -1e-9:
            env.offered.add(env.nodes[node_idx])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(discount, False, 0.0)
            continue
        obs, reward, done, info = env.step(node_idx, discount)
        if info["accepted"]:
            revenue += info["offered_price"]
        policy.update_sequence_state(discount, info["accepted"],
                                     info.get("revenue_step", 0.0))
        if done:
            break
    return revenue, bankrupt


def main():
    t0 = time.time()
    # Verify checkpoint sha
    sha = hashlib.sha256(open(CKPT, "rb").read()).hexdigest()
    assert sha == EXPECTED_SHA, f"SHA mismatch: {sha}"
    print(f"Checkpoint sha256: {sha[:16]}... OK")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(CKPT, map_location=device)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    pol  = pol.to(device)

    # Fixed FF graph (same graph as specialist eval: n=1000, p=0.37, pb=0.32, seed=0)
    graph = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    print(f"Graph: FF n={graph.number_of_nodes()} edges={graph.number_of_edges()}")

    results = {}
    for k in K_EVAL:
        revs = []
        bkr  = 0
        for seed in SEEDS:
            rev, bankrupt = eval_one(pol, graph, k, seed, device)
            revs.append(rev)
            if bankrupt:
                bkr += 1
        mean_ = float(np.mean(revs))
        std_  = float(np.std(revs))
        results[k] = {"mean": mean_, "std": std_, "all": revs, "bankrupt": bkr}
        print(f"k={k:2d}: mean={mean_:.1f}  std={std_:.1f}  per_seed={[round(r,1) for r in revs]}  bkr={bkr}")

    # Harness check
    k20 = results[20]["mean"]
    frozen_k20 = 369.6
    ok = abs(k20 - frozen_k20) <= 1.0
    print(f"\nHarness check: k=20 = {k20:.1f} (frozen={frozen_k20}, diff={k20-frozen_k20:+.1f}) {'PASS' if ok else 'FAIL'}")
    if not ok:
        print("WARNING: harness check failed — wrong checkpoint or graph seed")

    # Boundary
    spec_k16 = 351.1   # from largek_specialist_eval.json
    spec_k20 = 404.2
    u16 = results[16]["mean"]
    u20 = results[20]["mean"]
    boundary_k16 = (spec_k16 >= u16)
    boundary_k20 = (spec_k20 >= u20)
    print(f"\nBoundary check:")
    print(f"  k=16: specialist={spec_k16} vs unified={u16:.1f} -> specialist wins: {boundary_k16}")
    print(f"  k=20: specialist={spec_k20} vs unified={u20:.1f} -> specialist wins: {boundary_k20}")
    if boundary_k16:
        print("  BOUNDARY = k=16 (specialist wins from k=16)")
    else:
        print("  BOUNDARY = k=20 (unified still wins at k=16; specialist wins at k=20)")

    print(f"\nWall time: {time.time()-t0:.1f}s")

    out = {"ckpt_sha256": sha, "seeds": SEEDS, "results": {str(k): v for k,v in results.items()}}
    json.dump(out, open("results/logs/unified_k16_25_eval.json", "w"), indent=2)
    print("Saved → results/logs/unified_k16_25_eval.json")


if __name__ == "__main__":
    main()
