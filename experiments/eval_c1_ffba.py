#!/usr/bin/env python3
"""eval_c1_ffba.py — Pre-registered C1 evaluation for the FFBA arms.

Env:     RevenueEnv ONLY — no budget, no production cost, no feasibility.
Seeds:   single-seed 42 AND 5-seed [0,1,2,3,4].
Feat:    in_dim=20 unconstrained features, k=0 (no budget budget-k in vector).
N_MC:    5 (valuation MC samples per step, consistent with training).

Networks: FF_1000, FF_2000, Modular_FF, Rice_FB, polblogs
Models evaluated:
  (A) c1_ffba_50_50_final.pt          sha a190f4e3
  (B) c1_ffba_2to1_final.pt           sha fbea89ca
  (C) c1_ffba_50_50_p1_ep200.pt       P1-end arm_50_50
  (D) c1_ffba_2to1_p1_ep200.pt        P1-end arm_2to1
  (E) c1_ffba_50_50_p1_ep60.pt        ~1/3 Phase-1 arm_50_50
  (F) c1_ffba_50_50_p1_ep120.pt       ~2/3 Phase-1 arm_50_50
  (G) c1_ffba_2to1_p1_ep60.pt         ~1/3 Phase-1 arm_2to1
  (H) c1_ffba_2to1_p1_ep120.pt        ~2/3 Phase-1 arm_2to1
  (R) rev_gnn_lstm.pt                 released sha 8fbc4648 [reference]

Baselines (RevenueEnv, same seeds):
  greedy_discount: Greedy-Discount (all-n run)
  ie_strategy:     IE-Strategy (unconstrained)

Frozen reference (paper, BudgetRevenueEnv k=50; shown for context only):
  FF_1000=448.6, FF_2000=915.0, Modular_FF=414.4, Rice_FB=214.1, polblogs=374.2

Per-arm verdict (pre-registered, 5-seed mean):
  FIX CONFIRMED : polblogs>=525.7 AND FF_1000>=440 AND FF_2000>=900
                  AND Modular_FF>=400 AND Rice_FB>=200
  TRADE-OFF     : polblogs>=525.7 but one or more other floors missed
                  (printed with name + shortfall)
  NO FIX        : polblogs<525.7

Usage:
  python -u experiments/eval_c1_ffba.py [--networks NET ...] [--arm-tag TAG]
  (parallel launch: bash experiments/run_c1_ffba_eval_parallel.sh)

Output: results/logs/c1_ffba_eval.json
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

# Betweenness approximation
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache, compute_node_features_fast
from src.evaluation.baselines import ie_strategy, greedy_discount

# ── Patch: fast neighbour-ratio influence (avoids MC IC per step) ──────────────
def _fast_gci(self, node):
    nb = list(self.graph.neighbors(node))
    if not nb: return 0.
    tw = sum(self._link_weights.get((node,n),0.) for n in nb)
    if tw == 0: return 0.
    return sum(self._link_weights.get((node,n),0.) for n in nb if n in self.S)/tw
RevenueEnv.get_current_influence = _fast_gci

# ── Protocol ───────────────────────────────────────────────────────────────────
W_HIGH   = 2.0
N_MC     = 5
SEEDS_5  = [0, 1, 2, 3, 4]
SEED_42  = [42]
B_RAY    = 1.0          # Rayleigh b param

# ── Frozen reference (from paper, BudgetRevenueEnv k=50) ──────────────────────
FROZEN_REF = {
    "FF_1000":    448.6,
    "FF_2000":    915.0,
    "Modular_FF": 414.4,
    "Rice_FB":    214.1,
    "polblogs":   374.2,
}

# ── Checkpoints ────────────────────────────────────────────────────────────────
CKPT_DIR = "results/checkpoints"
MODELS = {
    "c1_50_50_final":   ("c1_ffba_50_50_final.pt",        "a190f4e3", "fin"),
    "c1_2to1_final":    ("c1_ffba_2to1_final.pt",         "fbea89ca", "fin"),
    "c1_50_50_p1_end":  ("c1_ffba_50_50_p1_ep200.pt",     None,       "p1e"),
    "c1_2to1_p1_end":   ("c1_ffba_2to1_p1_ep200.pt",      None,       "p1e"),
    "c1_50_50_p1_60":   ("c1_ffba_50_50_p1_ep60.pt",      None,       "1/3"),
    "c1_50_50_p1_120":  ("c1_ffba_50_50_p1_ep120.pt",     None,       "2/3"),
    "c1_2to1_p1_60":    ("c1_ffba_2to1_p1_ep60.pt",       None,       "1/3"),
    "c1_2to1_p1_120":   ("c1_ffba_2to1_p1_ep120.pt",      None,       "2/3"),
    "released_lstm":    ("rev_gnn_lstm.pt",                "8fbc4648", "ref"),
}

LOG_OUT = "results/logs/c1_ffba_eval.json"


def _sha8(p):
    return hashlib.sha256(open(p,'rb').read()).hexdigest()[:8]

def _cfg(seed=0):
    """SimpleNamespace cfg compatible with ie_strategy / greedy_discount."""
    inf = SimpleNamespace(b=B_RAY, weight_low=0., weight_high=W_HIGH,
                          n_mc_samples=N_MC, model="monotone",
                          k_seeds=30)
    tr  = SimpleNamespace(imitation_lr=1e-3)
    bud = SimpleNamespace(k=30)
    prj = SimpleNamespace(seed=seed)
    rwd = SimpleNamespace(type="flat", gamma=1.0)
    return SimpleNamespace(influence=inf, training=tr, budget=bud,
                           project=prj, reward=rwd,
                           graph=SimpleNamespace(p=0.37, pb=0.32, n_nodes=500))

def _load_policy(ckpt_path, device):
    """Load any in_dim=20 C1 policy (strict=True)."""
    enc  = GraphSAGEEncoder(in_dim=20, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(ckpt_path, map_location=device, weights_only=True)
    if "policy_state_dict" in sd: sd = sd["policy_state_dict"]
    elif "model_state_dict" in sd: sd = sd["model_state_dict"]
    pol.load_state_dict(sd, strict=True)
    return pol.to(device).eval()

def _edge_index(G, device):
    nmap = {v:i for i,v in enumerate(G.nodes())}
    E    = list(G.edges())
    s    = [nmap[u] for u,_ in E]+[nmap[v] for _,v in E]
    d    = [nmap[v] for _,v in E]+[nmap[u] for u,_ in E]
    return torch.tensor([s,d], dtype=torch.long, device=device)

@torch.no_grad()
def eval_policy_episode(policy, G, cache, ei, seed, device):
    """One episode in RevenueEnv (unconstrained). Returns (revenue, discounts)."""
    env_cfg = RevenueEnvConfig(influence_model="monotone", b=B_RAY,
                               weight_low=0., weight_high=W_HIGH,
                               n_mc_samples=N_MC, reward_type="flat",
                               gamma=1.0, seed=seed)
    env = RevenueEnv(G, env_cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    policy.reset_episode(device)
    rev = 0.0; discounts = []
    offered_set = set()
    for _ in range(n):
        av = torch.tensor([v not in env.offered for v in nodes],
                           dtype=torch.bool, device=device)
        if not av.any(): break
        feats = compute_node_features_fast(cache, env.S, frozenset(env.offered),
                                           env.t, k=0, env=env)
        x  = torch.FloatTensor(feats).to(device)
        ms, h, ctx, _ = policy.forward(x, ei, av)
        ni = int(ms.argmax().item())
        assert nodes[ni] not in offered_set, f"double-offer node={nodes[ni]}"
        offered_set.add(nodes[ni])
        d  = float(policy.get_discount_distribution(
                        torch.cat([h[ni], ctx])).mean.item())
        discounts.append(d)
        _, step_rev, done, info = env.step(ni, d)
        rev += info.get("revenue", 0.0)
        if done: break
    return rev, discounts

def eval_policy(policy, G, cache, ei, seeds, device):
    """Multi-seed eval; returns mean_rev, mean_disc, frac_d_gt_09."""
    all_revs = []; all_discs = []
    for s in seeds:
        r, ds = eval_policy_episode(policy, G, cache, ei, s, device)
        all_revs.append(r); all_discs.extend(ds)
    d_arr = np.array(all_discs, dtype=float)
    return {
        "mean": round(float(np.mean(all_revs)), 2),
        "per_seed": [round(v,2) for v in all_revs],
        "mean_disc": round(float(np.mean(d_arr)) if len(d_arr) else 0., 4),
        "frac_d_gt09": round(float(np.mean(d_arr > 0.9)) if len(d_arr) else 0., 4),
    }

def eval_greedy(G, seeds):
    """Run greedy_discount (unconstrained) for each seed; return stats."""
    revs = []
    for s in seeds:
        cfg = _cfg(s)
        revs.append(float(greedy_discount(G, cfg)))
    return {"mean": round(float(np.mean(revs)),2), "per_seed": [round(v,2) for v in revs]}

def eval_ie(G, seeds):
    """Run ie_strategy (unconstrained) for each seed; return stats."""
    revs = []
    for s in seeds:
        cfg = _cfg(s)
        revs.append(float(ie_strategy(G, cfg)))
    return {"mean": round(float(np.mean(revs)),2), "per_seed": [round(v,2) for v in revs]}

# Pre-registered floors
FLOORS = {
    "polblogs":   525.7,
    "FF_1000":    440.0,
    "FF_2000":    900.0,
    "Modular_FF": 400.0,
    "Rice_FB":    200.0,
}

def verdict(arm_5seed_means):
    """
    FIX CONFIRMED : polblogs>=525.7 AND FF_1000>=440 AND FF_2000>=900
                    AND Modular_FF>=400 AND Rice_FB>=200
    TRADE-OFF     : polblogs>=525.7 but >=1 other floor missed
    NO FIX        : polblogs<525.7
    Returns (verdict_str, detail_str) where detail lists five values and any shortfalls.
    """
    vals     = {net: arm_5seed_means.get(net, 0.0) for net in FLOORS}
    polblogs_ok = vals["polblogs"] >= FLOORS["polblogs"]
    missed   = [(net, vals[net], FLOORS[net], FLOORS[net]-vals[net])
                for net in FLOORS if vals[net] < FLOORS[net]]
    detail   = "  ".join(f"{n}:{vals[n]:.1f}(fl={FLOORS[n]})" for n in FLOORS)
    if polblogs_ok and not missed:
        return "FIX CONFIRMED", detail
    elif polblogs_ok:
        miss_str = ", ".join(f"{n}={v:.1f}<{fl:.1f}(short {s:.1f})"
                             for n,v,fl,s in missed)
        return "TRADE-OFF", f"{detail}  MISSED: {miss_str}"
    else:
        short = FLOORS["polblogs"] - vals["polblogs"]
        return "NO FIX", f"{detail}  polblogs short by {short:.1f}"


def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--networks", nargs="+", default=None)
    p.add_argument("--arm-tag", nargs="+", default=None,
                   help="Subset of model keys (e.g. c1_50_50_final c1_2to1_final)")
    return p.parse_args()


def main():
    args = _parse_args()
    t0   = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[c1-eval] device={device}  feature:in_dim=20,k=0", flush=True)

    # ── Networks ───────────────────────────────────────────────────────────
    all_networks = {
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
        "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
        "Rice_FB":    load_rice_facebook(),
        "polblogs":   load_polblogs(),
    }
    net_filter = args.networks if args.networks else list(all_networks.keys())
    networks   = {k: all_networks[k] for k in net_filter if k in all_networks}
    print(f"[c1-eval] networks: {list(networks.keys())}", flush=True)

    # ── Pre-compute caches + edge indices ──────────────────────────────────
    caches = {n: build_graph_feature_cache(G, compute_static_features(G))
              for n, G in networks.items()}
    eis    = {n: _edge_index(G, device) for n, G in networks.items()}

    # ── Load checkpoints ───────────────────────────────────────────────────
    model_filter = args.arm_tag if args.arm_tag else list(MODELS.keys())
    loaded = {}
    print(f"\n[c1-eval] loading {len(model_filter)} models...", flush=True)
    for key in model_filter:
        if key not in MODELS:
            print(f"  SKIP unknown key: {key}", flush=True); continue
        fname, expected_sha, phase = MODELS[key]
        path = os.path.join(CKPT_DIR, fname)
        if not os.path.exists(path):
            print(f"  MISSING: {fname}", flush=True); continue
        actual_sha = _sha8(path)
        sha_ok = (expected_sha is None) or (actual_sha == expected_sha)
        print(f"  {key}: {fname}  sha={actual_sha}  "
              f"{'OK' if sha_ok else 'MISMATCH vs '+expected_sha}", flush=True)
        try:
            loaded[key] = _load_policy(path, device)
        except Exception as e:
            print(f"  LOAD ERROR {key}: {e}", flush=True)

    # ── Baseline + policy evaluations ──────────────────────────────────────
    results = {}
    for net_name, G in networks.items():
        print(f"\n[c1-eval] ── {net_name} (n={G.number_of_nodes()}) ──", flush=True)
        cache = caches[net_name]; ei = eis[net_name]
        r = {}

        # Baselines (5-seed + seed-42)
        t_bl = time.time()
        for bl_name, bl_fn in [("greedy_discount", eval_greedy), ("ie_strategy", eval_ie)]:
            r5  = bl_fn(G, SEEDS_5)
            r42 = bl_fn(G, SEED_42)
            r[bl_name] = {"5seed": r5, "seed42": r42}
            print(f"  {bl_name}: 5s={r5['mean']:.1f}  s42={r42['mean']:.1f}", flush=True)
        print(f"  baselines done in {time.time()-t_bl:.0f}s", flush=True)

        # Frozen reference row (static — no eval needed)
        ref_val = FROZEN_REF.get(net_name)
        r["frozen_ref"] = {"note": "BudgetRevenueEnv k=50, from paper", "mean": ref_val}

        # Neural models
        for key, pol in loaded.items():
            t_m = time.time()
            r5  = eval_policy(pol, G, cache, ei, SEEDS_5,  device)
            r42 = eval_policy(pol, G, cache, ei, SEED_42, device)
            r[key] = {"5seed": r5, "seed42": r42, "phase": MODELS[key][2]}
            print(f"  {key}: 5s={r5['mean']:.1f}  s42={r42['mean']:.1f}  "
                  f"disc={r5['mean_disc']:.3f}  d>0.9={r5['frac_d_gt09']:.3f}  "
                  f"({time.time()-t_m:.0f}s)", flush=True)

        results[net_name] = r

    # ── Print summary table ─────────────────────────────────────────────────
    net_names = list(networks.keys())
    row_keys  = (["frozen_ref","greedy_discount","ie_strategy"]
                 + [k for k in loaded.keys()])
    print(f"\n\n{'model':<24}", end="")
    for net in net_names: print(f"  {net[:10]:>10}", end="")
    print()
    print("─"*80)
    for rk in row_keys:
        print(f"{rk:<24}", end="")
        for net in net_names:
            cell = results.get(net,{}).get(rk)
            if cell is None: print(f"  {'--':>10}", end=""); continue
            v = cell.get("5seed",{}).get("mean") or cell.get("mean")
            print(f"  {v:>10.1f}" if v is not None else f"  {'--':>10}", end="")
        print()

    # ── Verdicts ────────────────────────────────────────────────────────────
    print("\n\nVERDICTS (pre-registered floors: polblogs≥525.7 FF_1000≥440 FF_2000≥900 Modular_FF≥400 Rice_FB≥200):")
    for key in [k for k in loaded if k.endswith("_final")]:
        arm_means = {net: results.get(net,{}).get(key,{}).get("5seed",{}).get("mean",0)
                     for net in net_names}
        v, detail = verdict(arm_means)
        print(f"  {key:<28}  {v}")
        print(f"    {detail}")

    wall = time.time() - t0
    print(f"\n[c1-eval] wall={wall:.0f}s ({wall/60:.1f} min)", flush=True)

    # ── Save JSON ────────────────────────────────────────────────────────────
    # Derive unique output path if network-sharded
    out_path = LOG_OUT
    if args.networks:
        tag = "_".join(args.networks)
        out_path = f"results/logs/c1_ffba_eval_{tag}.json"
    os.makedirs("results/logs", exist_ok=True)
    out = {
        "protocol": {
            "env": "RevenueEnv (no budget, no cost, no feasibility)",
            "feat": "in_dim=20, k=0, compute_node_features_fast",
            "seeds_5": SEEDS_5,
            "seed_42": SEED_42,
            "w_high": W_HIGH, "n_mc": N_MC,
            "networks": net_filter,
        },
        "frozen_ref": FROZEN_REF,
        "results": results,
        "wall_s": round(wall, 1),
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[c1-eval] saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
