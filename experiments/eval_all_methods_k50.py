#!/usr/bin/env python3
"""eval_all_methods_k50.py — 6-method comparison at k=50, BudgetRevenueEnv, all 5 networks.

Methods (ALL in BudgetRevenueEnv, k=50, B=15, C=0.3):
  1. Greedy+Budget        — Babaei 2013 budget-aware
  2. Cal-DP composite     — max(v2, v3) per trial, fresh calibration per network
  3. OURS                 — deployment rule: unified(k<20) | largek(k>=20)
  4. lstm_v1              — rev_gnn_lstm_budget.pt (per-network released alternative)
  5. arm_a (unconstrained-trained) — rev_gnn_lstm_ba.pt, sha 32a9053a
  6. arm_b (unconstrained-trained) — rev_gnn_lstm_densemix.pt, sha 00368482, ep80

Summary lines cover OURS only. Rows 5-6 reported but excluded from summary.

Run on server:
    python -u experiments/eval_all_methods_k50.py 2>&1 | tee /tmp/all_methods_k50.log

Output: results/logs/all_methods_k50.json
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
K          = 50
C          = 0.3
B          = K * C        # = 15.0
b_RAY      = 1.0
W_HIGH     = 2.0
N_MC       = 5
N_TRIALS   = 3            # seeds 0,1,2 for all methods
SEEDS      = list(range(N_TRIALS))

# ── Checkpoints ───────────────────────────────────────────────────────────────
CKPT_DIR      = "results/checkpoints"
OURS_SMALL    = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")    # k < 20
OURS_LARGE    = os.path.join(CKPT_DIR, "rev_gnn_lstm_largek.pt")     # k >= 20
LSTM_V1_CKPT  = os.path.join(CKPT_DIR, "rev_gnn_lstm_budget.pt")     # a7828957
ARM_A_CKPT    = os.path.join(CKPT_DIR, "rev_gnn_lstm_ba.pt")         # 32a9053a
ARM_B_CKPT    = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")   # 00368482

LOG_OUT = "results/logs/all_methods_k50.json"


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _edge_index(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    nmap = {v:i for i,v in enumerate(G.nodes())}
    src = [nmap[u] for u,_ in edges]+[nmap[v] for _,v in edges]
    dst = [nmap[v] for _,v in edges]+[nmap[u] for u,_ in edges]
    return torch.tensor([src,dst],dtype=torch.long,device=device)

def _load_policy(ckpt, device, in_dim=21):
    enc = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=64, n_layers=2)
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

# ── Budget-aware feature (actual budget_remaining/B as dim 21) ─────────────
def _feat_budget(cache, env, k):
    return compute_budget_node_features_fast(cache, env.S, env.offered, env.t, k=k, env=env)

# ── Unconstrained proxy feature (budget_col=1.0 always) ────────────────────
def _feat_unconstrained(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    return np.concatenate([base, np.ones((cache["n"],1), dtype=np.float32)], axis=1)

@torch.no_grad()
def eval_policy_episode(policy, G, cache, ei, seed, device, feat_fn, skip_enforce=False):
    """Evaluate one episode. skip_enforce: SKIP if price would bankrupt."""
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    n = G.number_of_nodes()
    policy.reset_episode(device); rev = 0.0
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(feat_fn(cache, env, K)).to(device)
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

def eval_policy(policy, G, cache, ei, device, feat_fn, label, skip_enforce=False):
    revs = []
    for s in SEEDS:
        r = eval_policy_episode(policy, G, cache, ei, s, device, feat_fn, skip_enforce)
        revs.append(r)
        print(f"    [{label}] seed={s} rev={r:.1f}", flush=True)
    m = float(np.mean(revs))
    print(f"  → {label} MEAN={m:.1f}", flush=True)
    return m, revs

def eval_greedy_budget(G):
    res = greedy_discount_budget(G, B=B, c=C, b=b_RAY, n_trials=N_TRIALS, weight_high=W_HIGH)
    m = res["revenue"]["mean"]
    print(f"  → Greedy+Budget MEAN={m:.1f}", flush=True)
    return m

def eval_caldp_composite(G):
    """Run v2 and v3, take composite max per trial, return mean."""
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    print(f"    [Cal-DP v2] calibrating...", flush=True)
    r2 = dp_calibrated_v2_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    print(f"    [Cal-DP v3] calibrating...", flush=True)
    r3 = dp_calibrated_v3_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    v2_all = r2["revenue"]["all"] if isinstance(r2["revenue"], dict) else [r2["revenue"]]
    v3_all = r3["revenue"]["all"] if isinstance(r3["revenue"], dict) else [r3["revenue"]]
    composite = [max(a, b) for a, b in zip(v2_all, v3_all)]
    m = float(np.mean(composite))
    print(f"    v2={np.mean(v2_all):.1f}  v3={np.mean(v3_all):.1f}  composite={m:.1f}", flush=True)
    print(f"  → Cal-DP composite MEAN={m:.1f}", flush=True)
    return m


def main():
    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[all-methods-k50] k={K}  B={B:.1f}  C={C}  N_TRIALS={N_TRIALS}  device={device}")

    # Print checkpoint SHAs
    for label, ckpt in [("OURS_LARGE", OURS_LARGE), ("lstm_v1", LSTM_V1_CKPT),
                        ("arm_a", ARM_A_CKPT), ("arm_b", ARM_B_CKPT)]:
        sha = _sha8(ckpt) if os.path.exists(ckpt) else "MISSING"
        print(f"  {label}: sha8={sha}", flush=True)

    # Load networks
    print("\n[all-methods-k50] Loading networks...", flush=True)
    networks = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }
    for name, G in networks.items():
        print(f"  {name}: n={G.number_of_nodes()} edges={G.number_of_edges()}", flush=True)

    # Load NN policies (OURS=largek uses budget feat; arms use unconstrained feat)
    print("\n[all-methods-k50] Loading policies...", flush=True)
    ours   = _load_policy(OURS_LARGE, device)
    lstmv1 = _load_policy(LSTM_V1_CKPT, device)
    arma   = _load_policy(ARM_A_CKPT, device)
    armb   = _load_policy(ARM_B_CKPT, device)

    results = {}
    METHODS = ["greedy_budget", "caldp_composite", "ours", "lstm_v1", "arm_a", "arm_b"]
    for net_name, G in networks.items():
        print(f"\n{'='*60}", flush=True)
        print(f"[all-methods-k50] {net_name} (n={G.number_of_nodes()})", flush=True)
        cache = build_graph_feature_cache(G, compute_static_features(G))
        ei    = _edge_index(G, device)

        r = {}
        # 1. Greedy+Budget
        print(f"\n  [1/6] Greedy+Budget", flush=True)
        r["greedy_budget"] = eval_greedy_budget(G)

        # 2. Cal-DP composite
        print(f"\n  [2/6] Cal-DP composite", flush=True)
        r["caldp_composite"] = eval_caldp_composite(G)

        # 3. OURS (largek, budget features, SKIP enforcement)
        print(f"\n  [3/6] OURS (largek, budget-aware)", flush=True)
        r["ours"], _ = eval_policy(ours, G, cache, ei, device,
                                   _feat_budget, "OURS", skip_enforce=True)

        # 4. lstm_v1 (budget.pt, budget features)
        print(f"\n  [4/6] lstm_v1 (rev_gnn_lstm_budget.pt)", flush=True)
        r["lstm_v1"], _ = eval_policy(lstmv1, G, cache, ei, device,
                                      _feat_budget, "lstm_v1", skip_enforce=True)

        # 5. arm_a (unconstrained-trained, budget_col=1.0)
        print(f"\n  [5/6] arm_a (unconstrained-trained)", flush=True)
        r["arm_a"], _ = eval_policy(arma, G, cache, ei, device,
                                    _feat_unconstrained, "arm_a(unc)")

        # 6. arm_b (unconstrained-trained, budget_col=1.0)
        print(f"\n  [6/6] arm_b (unconstrained-trained)", flush=True)
        r["arm_b"], _ = eval_policy(armb, G, cache, ei, device,
                                    _feat_unconstrained, "arm_b(unc)")

        results[net_name] = r

    # Print table
    wall = time.time() - t_start
    print(f"\n\n{'─'*80}")
    print(f"{'network':<14} | {'Greedy+B':>8} | {'Cal-DP':>8} | {'OURS':>8} | "
          f"{'lstm_v1':>8} | {'arm_a(unc)':>10} | {'arm_b(unc)':>10}")
    print(f"{'─'*80}")
    for net in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        r = results[net]
        print(f"{net:<14} | {r['greedy_budget']:>8.1f} | {r['caldp_composite']:>8.1f} | "
              f"{r['ours']:>8.1f} | {r['lstm_v1']:>8.1f} | "
              f"{r['arm_a']:>10.1f} | {r['arm_b']:>10.1f}")
    print(f"{'─'*80}")

    # Summary lines (OURS only)
    print(f"\nSUMMARY (OURS = largek, k=50, BudgetRevenueEnv):")
    for net in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        r = results[net]
        vs_g = r["ours"] - r["greedy_budget"]
        vs_d = r["ours"] - r["caldp_composite"]
        print(f"  {net:<14}: OURS={r['ours']:.1f}  vs_Greedy={vs_g:+.1f}  vs_CalDP={vs_d:+.1f}")

    # Observation: unconstrained arms beating BOTH baselines
    print(f"\nOBSERVATION — Networks where an unconstrained arm beats BOTH Greedy+Budget AND Cal-DP:")
    found = False
    for net in ["polblogs","FF_1000","Rice_FB","Modular_FF","FF_2000"]:
        r = results[net]
        for arm_label, arm_val in [("arm_a", r["arm_a"]), ("arm_b", r["arm_b"])]:
            if arm_val > r["greedy_budget"] and arm_val > r["caldp_composite"]:
                print(f"  {net} {arm_label}: {arm_val:.1f} > Greedy {r['greedy_budget']:.1f}"
                      f" AND > CalDP {r['caldp_composite']:.1f}")
                found = True
    if not found:
        print("  None — unconstrained arms do NOT beat both baselines on any network.")

    print(f"\nWall time: {wall:.0f}s  ({wall/60:.1f} min)")

    # Save JSON
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "protocol": {"k": K, "B": B, "C": C, "n_trials": N_TRIALS, "weight_high": W_HIGH,
                     "ours_ckpt": "largek (k>=20)", "deployment_rule": "unified k<20, largek k>=20"},
        "results": results,
        "wall_s": wall,
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved → {LOG_OUT}")


if __name__ == "__main__":
    main()
