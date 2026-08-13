#!/usr/bin/env python3
"""eval_all_methods_ksweep.py — 6-method k-sweep [5,10,15,20,30,40], BudgetRevenueEnv, all 5 networks.

Methods (ALL in BudgetRevenueEnv, B=k*C, C=0.3):
  1. Greedy+Budget        — Babaei 2013 budget-aware
  2. Cal-DP composite     — max(v2, v3) per trial, fresh calibration per (network, k)
  3. OURS                 — deployment rule: unified(k<20) | largek(k>=20)
  4. lstm_v1              — rev_gnn_lstm_budget.pt (per-network released alternative)
  5. arm_a (unconstrained-trained) — rev_gnn_lstm_ba.pt, sha 32a9053a
  6. arm_b (unconstrained-trained) — rev_gnn_lstm_densemix.pt, sha 00368482, ep80

Summary lines cover OURS only. Rows 5-6 reported but excluded from summary.
Observation: networks/k where an unconstrained arm beats BOTH Greedy+Budget AND Cal-DP.

Run on server:
    python -u experiments/eval_all_methods_ksweep.py 2>&1 | tee /tmp/all_methods_ksweep.log

Output: results/logs/all_methods_ksweep.json
"""
from __future__ import annotations
import hashlib, json, os, sys, time
import numpy as np
import torch
import networkx as nx

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
from src.utils.budget_features import compute_budget_node_features_fast
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget

# ── Protocol ──────────────────────────────────────────────────────────────────
K_VALUES   = [5, 10, 15, 20, 30, 40]
C          = 0.3
b_RAY      = 1.0
W_HIGH     = 2.0
N_MC       = 5
N_TRIALS   = 3            # seeds 0,1,2 for all methods
SEEDS      = list(range(N_TRIALS))

# OURS deployment rule: k<20 → unified, k>=20 → largek
K_SWITCH   = 20

# ── Checkpoints ───────────────────────────────────────────────────────────────
CKPT_DIR     = "results/checkpoints"
OURS_SMALL   = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")    # k < 20
OURS_LARGE   = os.path.join(CKPT_DIR, "rev_gnn_lstm_largek.pt")     # k >= 20
LSTM_V1_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_budget.pt")     # a7828957
ARM_A_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_ba.pt")         # 32a9053a
ARM_B_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")   # 00368482

LOG_OUT = "results/logs/all_methods_ksweep.json"


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _load_policy(ckpt, device):
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

def _avail(env, n, device):
    m = torch.zeros(n, dtype=torch.bool, device=device)
    for i in env.available_nodes: m[i] = True
    return m

def _feat_budget(cache, env, k):
    # OURS/lstm_v1 trained with k=n_val for round_ratio (matches run_largek_eval.py)
    return compute_budget_node_features_fast(cache, env.S, env.offered, env.t, k=cache["n"], env=env)

_ARM_K = 50   # arm_a/arm_b trained at fixed K=50 — keep for round_ratio consistency with training
def _feat_unconstrained(cache, env, k):
    # arms: run_topology_arms.py trained with k=K=50; budget_col frozen at 1.0
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, _ARM_K, env)
    return np.concatenate([base, np.ones((cache["n"],1), dtype=np.float32)], axis=1)

@torch.no_grad()
def eval_policy_episode(policy, G, cache, ei, k, B, seed, device, feat_fn, skip_enforce=False):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device); rev = 0.0
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(feat_fn(cache, env, k)).to(device)
        av = _avail(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = policy.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        if skip_enforce:
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

def eval_policy_k(policy, G, cache, ei, k, B, device, feat_fn, skip_enforce=False):
    return float(np.mean([
        eval_policy_episode(policy, G, cache, ei, k, B, s, device, feat_fn, skip_enforce)
        for s in SEEDS
    ]))

def eval_greedy_k(G, k, B):
    res = greedy_discount_budget(G, B=B, c=C, b=b_RAY, n_trials=N_TRIALS, weight_high=W_HIGH)
    return res["revenue"]["mean"]

def eval_caldp_k(G, k, B):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    r2 = dp_calibrated_v2_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    r3 = dp_calibrated_v3_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    v2 = r2["revenue"]["all"] if isinstance(r2["revenue"], dict) else [r2["revenue"]]
    v3 = r3["revenue"]["all"] if isinstance(r3["revenue"], dict) else [r3["revenue"]]
    composite = [max(a, b) for a, b in zip(v2, v3)]
    return float(np.mean(composite))


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ksweep] k={K_VALUES}  C={C}  N_TRIALS={N_TRIALS}  device={device}")
    for label, ckpt in [("OURS_SMALL", OURS_SMALL), ("OURS_LARGE", OURS_LARGE),
                        ("lstm_v1", LSTM_V1_CKPT), ("arm_a", ARM_A_CKPT), ("arm_b", ARM_B_CKPT)]:
        sha = _sha8(ckpt) if os.path.exists(ckpt) else "MISSING"
        print(f"  {label}: sha8={sha}", flush=True)

    # Load networks once
    print("\n[ksweep] Loading networks...", flush=True)
    networks = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }

    # Pre-compute static features + edge index once per network
    caches, eis = {}, {}
    for name, G in networks.items():
        print(f"  {name}: computing features...", flush=True)
        caches[name] = build_graph_feature_cache(G, compute_static_features(G))
        eis[name]    = _edge_index(G, device)
        print(f"    n={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)

    # Load policies
    print("\n[ksweep] Loading policies...", flush=True)
    pol_small  = _load_policy(OURS_SMALL, device)    # OURS k<20
    pol_large  = _load_policy(OURS_LARGE, device)    # OURS k>=20
    pol_lstmv1 = _load_policy(LSTM_V1_CKPT, device)
    pol_arma   = _load_policy(ARM_A_CKPT, device)
    pol_armb   = _load_policy(ARM_B_CKPT, device)

    # Main sweep
    all_results = {}   # [net_name][k] = {method: value}
    for net_name, G in networks.items():
        all_results[net_name] = {}
        cache = caches[net_name]
        ei    = eis[net_name]

        for k in K_VALUES:
            B = k * C
            t_k = time.time()
            print(f"\n[ksweep] {net_name}  k={k}  B={B:.1f}", flush=True)
            r = {}

            # Select OURS policy
            ours_pol  = pol_small if k < K_SWITCH else pol_large
            ours_label = f"unified(k<{K_SWITCH})" if k < K_SWITCH else f"largek(k>={K_SWITCH})"

            r["greedy_budget"]   = eval_greedy_k(G, k, B)
            print(f"  Greedy+Budget={r['greedy_budget']:.1f}", flush=True)

            r["caldp_composite"] = eval_caldp_k(G, k, B)
            print(f"  Cal-DP composite={r['caldp_composite']:.1f}", flush=True)

            r["ours"]   = eval_policy_k(ours_pol, G, cache, ei, k, B, device, _feat_budget, skip_enforce=True)
            print(f"  OURS({ours_label})={r['ours']:.1f}", flush=True)

            r["lstm_v1"] = eval_policy_k(pol_lstmv1, G, cache, ei, k, B, device, _feat_budget, skip_enforce=True)
            print(f"  lstm_v1={r['lstm_v1']:.1f}", flush=True)

            r["arm_a"] = eval_policy_k(pol_arma, G, cache, ei, k, B, device, _feat_unconstrained)
            print(f"  arm_a(unc)={r['arm_a']:.1f}", flush=True)

            r["arm_b"] = eval_policy_k(pol_armb, G, cache, ei, k, B, device, _feat_unconstrained)
            print(f"  arm_b(unc)={r['arm_b']:.1f}  ({time.time()-t_k:.0f}s)", flush=True)

            all_results[net_name][k] = r

    # Print per-network tables
    wall = time.time() - t_start
    for net_name in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        print(f"\n── {net_name} ──")
        print(f"{'k':>4} | {'B':>5} | {'Greedy+B':>8} | {'Cal-DP':>8} | {'OURS':>8} | "
              f"{'lstm_v1':>8} | {'arm_a(unc)':>10} | {'arm_b(unc)':>10}")
        print(f"{'─'*85}")
        for k in K_VALUES:
            B = k * C
            r = all_results[net_name][k]
            print(f"{k:>4} | {B:>5.1f} | {r['greedy_budget']:>8.1f} | {r['caldp_composite']:>8.1f} | "
                  f"{r['ours']:>8.1f} | {r['lstm_v1']:>8.1f} | "
                  f"{r['arm_a']:>10.1f} | {r['arm_b']:>10.1f}")

    # Summary: OURS vs baselines
    print(f"\n\nSUMMARY — OURS vs Baselines (deployment rule: unified k<{K_SWITCH}, largek k>={K_SWITCH}):")
    for net_name in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        print(f"\n  {net_name}:")
        for k in K_VALUES:
            r = all_results[net_name][k]
            vg = r["ours"] - r["greedy_budget"]
            vd = r["ours"] - r["caldp_composite"]
            print(f"    k={k:2d}: OURS={r['ours']:.1f}  vs_Greedy={vg:+.1f}  vs_CalDP={vd:+.1f}")

    # Observation: unconstrained arms beating BOTH baselines
    print(f"\n\nOBSERVATION — (network, k) where unconstrained arm beats BOTH Greedy+Budget AND Cal-DP:")
    found = False
    for net_name in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        for k in K_VALUES:
            r = all_results[net_name][k]
            for arm_lbl, arm_val in [("arm_a", r["arm_a"]), ("arm_b", r["arm_b"])]:
                if arm_val > r["greedy_budget"] and arm_val > r["caldp_composite"]:
                    print(f"  {net_name} k={k} {arm_lbl}: {arm_val:.1f} "
                          f"> Greedy {r['greedy_budget']:.1f} AND > CalDP {r['caldp_composite']:.1f}")
                    found = True
    if not found:
        print("  None — no (network, k) pair where an unconstrained arm beats both baselines.")

    print(f"\nWall time: {wall:.0f}s  ({wall/60:.1f} min)")

    # Save
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "protocol": {"k_values": K_VALUES, "C": C, "n_trials": N_TRIALS,
                     "weight_high": W_HIGH, "deployment_rule": f"unified k<{K_SWITCH}, largek k>={K_SWITCH}"},
        "results": all_results,
        "wall_s": wall,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {LOG_OUT}")


if __name__ == "__main__":
    main()
