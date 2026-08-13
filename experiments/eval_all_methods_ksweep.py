#!/usr/bin/env python3
"""eval_all_methods_ksweep.py — 9-method k-sweep [5,10,15,20,30,40], BudgetRevenueEnv, all 5 networks.

ALL methods are evaluated inside BudgetRevenueEnv (B=k*C, C=0.3).
Budget-aware methods (1-5) know B; unconstrained-trained methods (6-9) do NOT
know B but the env still enforces bankruptcy — i.e. they stop when B<0.

Methods:
  1. Greedy+Budget        — Babaei 2013 budget-aware
  2. IE-Strategy          — Babaei et al. IE under §3.2 budget feasibility
  3. Cal-DP composite     — max(v2, v3) per trial, fresh calibration per (network, k)
  4. OURS                 — deployment rule: unified(k<20) | largek(k>=20) [budget-trained]
  5. lstm_v1              — rev_gnn_lstm_budget.pt (a7828957) [budget-trained]
  ── C1 arms: trained on RevenueEnv (NO budget), evaluated in BudgetRevenueEnv ──
  6. arm_a (C1-unc)       — rev_gnn_lstm_ba.pt, sha 32a9053a, in_dim=21+dummy
  7. arm_b (C1-unc)       — rev_gnn_lstm_densemix.pt, sha 00368482 ep80, in_dim=21+dummy
  8. c1_50/50 (C1-unc)    — c1_ffba_50_50_final.pt, sha a190f4e3, in_dim=20, FF:BA=1:1
  9. c1_2:1  (C1-unc)     — c1_ffba_2to1_final.pt,  sha fbea89ca, in_dim=20, FF:BA=2:1

Run on server:
    python -u experiments/eval_all_methods_ksweep.py 2>&1 | tee /tmp/all_methods_ksweep.log

Outputs:
  results/logs/budget_sweep_all_networks.json
  results/logs/network_topology_stats.json
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
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.utils.budget_features import compute_budget_node_features_fast
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget
from src.evaluation.ie_budget import ie_strategy_budget, IE_K_SEEDS

# ── Protocol ──────────────────────────────────────────────────────────────────
K_VALUES   = [5, 10, 15, 20, 30, 40]
C          = 0.3
b_RAY      = 1.0
W_HIGH     = 2.0
N_MC       = 5
N_TRIALS   = 3
SEEDS      = list(range(N_TRIALS))
K_SWITCH   = 20   # OURS: k<20 → unified, k>=20 → largek

# ── Checkpoints ───────────────────────────────────────────────────────────────
CKPT_DIR     = "results/checkpoints"
OURS_SMALL   = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")
OURS_LARGE   = os.path.join(CKPT_DIR, "rev_gnn_lstm_largek.pt")
LSTM_V1_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_budget.pt")
ARM_A_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_ba.pt")
ARM_B_CKPT   = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
C1_50_CKPT   = os.path.join(CKPT_DIR, "c1_ffba_50_50_final.pt")   # sha a190f4e3
C1_2TO1_CKPT = os.path.join(CKPT_DIR, "c1_ffba_2to1_final.pt")    # sha fbea89ca

LOG_OUT   = "results/logs/budget_sweep_all_networks.json"
TOPO_OUT  = "results/logs/network_topology_stats.json"

# Expected SHAs (fail fast if wrong checkpoint loaded)
# unified.pt: original gate sha=00071438; current on both Mac+server=57c23076
# (checkpoint was updated after unified_sweep.json was written; see CLAUDE.md)
EXPECTED_SHAS = {
    OURS_SMALL:   "57c23076",
    OURS_LARGE:   "3033620a",
    LSTM_V1_CKPT: "a7828957",
}


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _verify_shas():
    ok = True
    for ckpt, expected in EXPECTED_SHAS.items():
        actual = _sha8(ckpt) if os.path.exists(ckpt) else "MISSING"
        match = actual == expected
        print(f"  SHA {'OK' if match else 'MISMATCH'}: {os.path.basename(ckpt)} "
              f"actual={actual} expected={expected}", flush=True)
        if not match:
            ok = False
    # Arms: print sha but don't enforce (ep80 known)
    for label, ckpt in [("arm_a", ARM_A_CKPT), ("arm_b", ARM_B_CKPT),
                        ("c1_50_50", C1_50_CKPT), ("c1_2to1", C1_2TO1_CKPT)]:
        sha = _sha8(ckpt) if os.path.exists(ckpt) else "MISSING"
        print(f"  SHA INFO: {label} sha8={sha}", flush=True)
    return ok

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

def _feat_budget_unified(cache, env, k):
    """Convention B: k_feat=budget_k — matches rev_gnn_lstm_unified.pt training."""
    return compute_budget_node_features_fast(cache, env.S, env.offered, env.t, k=k, env=env)

def _feat_budget_largek(cache, env, k):
    """Convention A: k_feat=n_nodes — matches rev_gnn_lstm_largek.pt training."""
    return compute_budget_node_features_fast(cache, env.S, env.offered, env.t, k=cache["n"], env=env)

# lstm_v1 uses same convention as largek (budget-trained with k=n_nodes)
_feat_budget = _feat_budget_largek

_ARM_K = 50
def _feat_unconstrained(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, _ARM_K, env)
    return np.concatenate([base, np.ones((cache["n"],1), dtype=np.float32)], axis=1)

def _load_policy_c1(ckpt, device):
    """Loader for C1 (unconstrained) arms trained with in_dim=20."""
    enc = GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd = torch.load(ckpt, map_location=device, weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

def _feat_c1(cache, env, k):
    """20-feature (no budget dummy) — matches C1 FFBA arms training."""
    return compute_node_features_fast(cache, env.S, env.offered, env.t, _ARM_K, env)

@torch.no_grad()
def eval_policy_episode(policy, G, cache, ei, k, B, seed, device, feat_fn, skip_enforce=False):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    B0 = env.B
    n = G.number_of_nodes()
    policy.reset_episode(device)
    rev = 0.0; step_revs = []; n_accepted = 0; offers = set()
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(feat_fn(cache, env, k)).to(device)
        av = _avail(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = policy.forward(x, ei, av)
        ni = int(sc.argmax().item())
        node_id = env.nodes[ni]
        # Double-offer check
        assert node_id not in offers, f"DOUBLE-OFFER node={node_id}"
        offers.add(node_id)
        d  = float(policy.get_discount_distribution(torch.cat([h[ni],ctx])).mean.item())
        if skip_enforce:
            ev = env._estimate_valuation(env.nodes[ni])
            p  = ev * (1.0 - d)
            if env.B - C + p < -1e-9:
                env.offered.add(env.nodes[ni]); env.t += 1
                env.budget_history.append(env.B)
                policy.update_sequence_state(d, False, 0.0)
                continue
        B_before = env.B
        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            step_rev = info["offered_price"]
            rev += step_rev
            step_revs.append(step_rev)
            n_accepted += 1
        policy.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.0))
        if done: break
    # Accounting identity: B_final = B0 + sum(revenue) - n_accepted*C
    B_final = env.B
    acct_expected = B0 + sum(step_revs) - n_accepted * C
    acct_err = abs(B_final - acct_expected)
    return rev, acct_err

def eval_policy_k(policy, G, cache, ei, k, B, device, feat_fn, skip_enforce=False):
    revs, errs = [], []
    for s in SEEDS:
        r, e = eval_policy_episode(policy, G, cache, ei, k, B, s, device, feat_fn, skip_enforce)
        revs.append(r); errs.append(e)
    return float(np.mean(revs)), float(max(errs))

def eval_greedy_k(G, k, B):
    res = greedy_discount_budget(G, B=B, c=C, b=b_RAY, n_trials=N_TRIALS, weight_high=W_HIGH)
    return res["revenue"]["mean"]

def eval_ie_k(G, k, B):
    res = ie_strategy_budget(G, B=B, c=C, k_seeds=IE_K_SEEDS,
                             n_trials=N_TRIALS, weight_high=W_HIGH)
    return float(res["revenue"]["mean"])

def eval_caldp_k(G, k, B):
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0,
                         weight_high=W_HIGH, n_mc_samples=N_MC)
    r2 = dp_calibrated_v2_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    r3 = dp_calibrated_v3_budget(G, cfg, B=B, c=C, n_trials=N_TRIALS, n_sims=20)
    v2 = r2["revenue"]["all"] if isinstance(r2["revenue"], dict) else [r2["revenue"]]
    v3 = r3["revenue"]["all"] if isinstance(r3["revenue"], dict) else [r3["revenue"]]
    return float(np.mean([max(a, b) for a, b in zip(v2, v3)]))

def compute_topo_stats(G, name):
    degs = [d for _, d in G.degree()]
    cc = nx.average_clustering(G)
    try:
        comms = nx.community.greedy_modularity_communities(G)
        mod = nx.community.modularity(G, comms)
    except Exception:
        mod = None
    return {
        "n": G.number_of_nodes(), "m": G.number_of_edges(),
        "mean_deg": round(float(np.mean(degs)), 2),
        "median_deg": int(np.median(degs)),
        "max_deg": int(np.max(degs)),
        "max_median_ratio": round(float(np.max(degs)/max(1,np.median(degs))), 2),
        "avg_clustering": round(cc, 4),
        "modularity": round(mod, 4) if mod is not None else None,
    }


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", nargs="+", default=None,
                   metavar="NET",
                   help="Subset of networks to run (default: all 5). "
                        "Valid: polblogs FF_1000 Rice_FB Modular_FF FF_2000")
    p.add_argument("--k-values", nargs="+", type=int, default=None,
                   metavar="K",
                   help="Subset of k values to run (default: all 6). "
                        "Valid: 5 10 15 20 30 40")
    return p.parse_args()


def main():
    args = _parse_args()

    # Apply CLI filters (inner logic unchanged — only iterate subset)
    k_filter  = args.k_values if args.k_values else K_VALUES
    net_filter = args.networks if args.networks else None  # None = all

    # Derive a unique output path so parallel workers don't clobber each other
    if net_filter or k_filter != K_VALUES:
        net_tag = "_".join(net_filter) if net_filter else "all"
        k_tag   = "k" + "-".join(str(k) for k in k_filter)
        log_out = f"results/logs/budget_sweep_{net_tag}_{k_tag}.json"
    else:
        log_out = LOG_OUT

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[ksweep] k={k_filter}  nets={net_filter or 'all'}  "
          f"C={C}  N_TRIALS={N_TRIALS}  device={device}", flush=True)
    print(f"[ksweep] output → {log_out}", flush=True)

    # SHA verification
    print("\n[ksweep] Verifying checkpoint SHAs...", flush=True)
    if not _verify_shas():
        print("ABORT: checkpoint SHA mismatch — fix before running.", flush=True)
        sys.exit(1)

    # Load networks
    print("\n[ksweep] Loading networks...", flush=True)
    networks = {
        "polblogs":   load_polblogs(),
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
    }

    # Topology stats
    print("\n[ksweep] Computing topology stats...", flush=True)
    topo = {}
    for name, G in networks.items():
        topo[name] = compute_topo_stats(G, name)
        t = topo[name]
        print(f"  {name}: n={t['n']} m={t['m']} mean_deg={t['mean_deg']} "
              f"max={t['max_deg']} clust={t['avg_clustering']} mod={t['modularity']}", flush=True)
    os.makedirs("results/logs", exist_ok=True)
    with open(TOPO_OUT, "w") as f: json.dump(topo, f, indent=2)
    print(f"  Saved → {TOPO_OUT}", flush=True)

    # Features + edge index
    caches, eis = {}, {}
    for name, G in networks.items():
        caches[name] = build_graph_feature_cache(G, compute_static_features(G))
        eis[name]    = _edge_index(G, device)

    # Load policies
    print("\n[ksweep] Loading policies...", flush=True)
    pol_small  = _load_policy(OURS_SMALL, device)
    pol_large  = _load_policy(OURS_LARGE, device)
    pol_lstmv1 = _load_policy(LSTM_V1_CKPT, device)
    pol_arma     = _load_policy(ARM_A_CKPT, device)
    pol_armb     = _load_policy(ARM_B_CKPT, device)
    pol_c1_50    = _load_policy_c1(C1_50_CKPT, device)   if os.path.exists(C1_50_CKPT)   else None
    pol_c1_2to1  = _load_policy_c1(C1_2TO1_CKPT, device) if os.path.exists(C1_2TO1_CKPT) else None
    if pol_c1_50   is None: print("[ksweep] WARNING: c1_50_50 checkpoint missing — skipping", flush=True)
    if pol_c1_2to1 is None: print("[ksweep] WARNING: c1_2to1 checkpoint missing — skipping",  flush=True)

    # Apply network filter
    if net_filter:
        missing = [n for n in net_filter if n not in networks]
        if missing:
            print(f"ABORT: unknown network(s): {missing}", flush=True)
            sys.exit(1)
        networks = {n: networks[n] for n in net_filter}

    # Main sweep
    all_results = {}
    for net_name, G in networks.items():
        all_results[net_name] = {}
        cache = caches[net_name]; ei = eis[net_name]

        for k in k_filter:
            B = k * C
            t_k = time.time()
            print(f"\n[ksweep] {net_name}  k={k}  B={B:.1f}", flush=True)
            r = {}

            ours_pol  = pol_small if k < K_SWITCH else pol_large
            # unified uses Convention B (k_feat=budget_k); largek uses Convention A (k_feat=n_nodes)
            ours_feat = _feat_budget_unified if k < K_SWITCH else _feat_budget_largek
            r["greedy_budget"]   = eval_greedy_k(G, k, B)
            print(f"  Greedy+Budget={r['greedy_budget']:.1f}", flush=True)

            r["ie_budget"] = eval_ie_k(G, k, B)
            print(f"  IE-Strategy={r['ie_budget']:.1f}", flush=True)

            r["caldp_composite"] = eval_caldp_k(G, k, B)
            print(f"  Cal-DP composite={r['caldp_composite']:.1f}", flush=True)

            r["ours"], r["ours_acct_err"] = eval_policy_k(ours_pol, G, cache, ei, k, B, device, ours_feat, skip_enforce=True)
            print(f"  OURS={r['ours']:.1f}  acct_err={r['ours_acct_err']:.2e}", flush=True)

            r["lstm_v1"], r["lstm_v1_acct_err"] = eval_policy_k(pol_lstmv1, G, cache, ei, k, B, device, _feat_budget, skip_enforce=True)
            print(f"  lstm_v1={r['lstm_v1']:.1f}  acct_err={r['lstm_v1_acct_err']:.2e}", flush=True)

            r["arm_a"], r["arm_a_acct_err"] = eval_policy_k(pol_arma, G, cache, ei, k, B, device, _feat_unconstrained)
            print(f"  arm_a={r['arm_a']:.1f}  acct_err={r['arm_a_acct_err']:.2e}", flush=True)

            r["arm_b"], r["arm_b_acct_err"] = eval_policy_k(pol_armb, G, cache, ei, k, B, device, _feat_unconstrained)
            print(f"  arm_b={r['arm_b']:.1f}  acct_err={r['arm_b_acct_err']:.2e}", flush=True)

            if pol_c1_50 is not None:
                r["c1_50_50"], r["c1_50_50_acct_err"] = eval_policy_k(pol_c1_50, G, cache, ei, k, B, device, _feat_c1)
                print(f"  c1_50_50={r['c1_50_50']:.1f}  acct_err={r['c1_50_50_acct_err']:.2e}", flush=True)
            if pol_c1_2to1 is not None:
                r["c1_2to1"], r["c1_2to1_acct_err"] = eval_policy_k(pol_c1_2to1, G, cache, ei, k, B, device, _feat_c1)
                print(f"  c1_2to1={r['c1_2to1']:.1f}  acct_err={r['c1_2to1_acct_err']:.2e}  ({time.time()-t_k:.0f}s)", flush=True)

            all_results[net_name][k] = r

    wall = time.time() - t_start

    # ── Print tables (only computed networks/k-values) ────────────────────
    METHODS = ["greedy_budget","ie_budget","caldp_composite","ours","lstm_v1","arm_a","arm_b","c1_50_50","c1_2to1"]
    MLABELS = ["Greedy+B","IE-Strat","Cal-DP","OURS","lstm_v1","arm_a(unc)","arm_b(unc)","c1_50/50","c1_2:1"]
    for net_name in all_results:
        print(f"\n── {net_name} ──")
        print(f"{'method':<14}", end="")
        for k in k_filter: print(f"  k={k:2d}", end="")
        print()
        print("─"*60)
        for m, ml in zip(METHODS, MLABELS):
            print(f"{ml:<14}", end="")
            for k in k_filter:
                v = all_results[net_name][k].get(m, None)
                print(f"  {v:>5.0f}" if v is not None else "    --", end="")
            print()

    # ── Summary lines (only over computed cells) ──────────────────────────
    print("\n\nSUMMARY LINES:")
    computed_nets = list(all_results.keys())
    print("(a) Networks where OURS >= Greedy+Budget at EVERY computed k:")
    for net in computed_nets:
        if all(all_results[net][k]["ours"] >= all_results[net][k]["greedy_budget"] for k in k_filter):
            print(f"    {net}")
    print("(b) Networks where OURS >= Cal-DP composite at EVERY computed k:")
    for net in computed_nets:
        if all(all_results[net][k]["ours"] >= all_results[net][k]["caldp_composite"] for k in k_filter):
            print(f"    {net}")
    print("(c) (network,k) where arm_a OR arm_b beats BOTH Greedy+Budget AND Cal-DP:")
    found_c = False
    for net in computed_nets:
        for k in k_filter:
            r = all_results[net][k]
            for arm_lbl, arm_val in [("arm_a",r["arm_a"]),("arm_b",r["arm_b"])]:
                if arm_val > r["greedy_budget"] and arm_val > r["caldp_composite"]:
                    print(f"    {net} k={k} {arm_lbl}: {arm_val:.1f} > G={r['greedy_budget']:.1f} & D={r['caldp_composite']:.1f}")
                    found_c = True
    if not found_c: print("    None")
    if "polblogs" in all_results:
        print("(d) polblogs — OURS(unified,k<20) vs lstm_v1 at k∈{5,10,15}∩computed:")
        for k in [5,10,15]:
            if k in k_filter and k in all_results.get("polblogs",{}):
                r = all_results["polblogs"][k]
                print(f"    k={k}: OURS={r['ours']:.1f}  lstm_v1={r['lstm_v1']:.1f}  delta={r['ours']-r['lstm_v1']:+.1f}")

    print(f"\nWall time: {wall:.0f}s  ({wall/60:.1f} min)")

    # ── Save results ─────────────────────────────────────────────────────────
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "protocol": {"k_values": k_filter, "C": C, "n_trials": N_TRIALS,
                     "seeds": SEEDS, "weight_high": W_HIGH,
                     "deployment_rule": f"unified k<{K_SWITCH}, largek k>={K_SWITCH}",
                     "networks": list(all_results.keys())},
        "shas": {os.path.basename(k): v for k, v in EXPECTED_SHAS.items()},
        "results": all_results,
        "wall_s": wall,
    }
    with open(log_out, "w") as f: json.dump(out, f, indent=2)
    print(f"Saved → {log_out}")


if __name__ == "__main__":
    main()
