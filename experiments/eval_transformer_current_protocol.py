#!/usr/bin/env python3
"""eval_transformer_current_protocol.py
Evaluate Rev-GNN-Transformer under the CURRENT project protocol:
  monotone+flat, Uniform(0,2), n_mc=200, seeds [0..9], greedy argmax.
  Unconstrained: BudgetRevenueEnv with B=1e7
  Budget: kappa in {5, 20}, profit = R - c*|S_T|, Pi-assert

CHECKPOINT PROVENANCE:
  Gate A checkpoint: results/checkpoints/rev_gnn_transformer.pt  sha=c24215b8
  Gate A protocol (what produced 463.84±5.26):
    - env: RevenueEnv (unconstrained) via baselines._make_env
    - graph: Forest Fire p=0.37, pb=0.32
    - weights: Uniform(0,2) per Babaei model (from YAML config)
    - n_mc: from config (typically 200)
    - seeds: 20 for FF_1000 primary; 5 for OOD
    - NO budget constraint, NO production cost
    - verdict: crit1 (mean>=457.6), crit2 (OOD wins>=2) → GATE A PASS

  This script uses the IDENTICAL feature pipeline (compute_node_features_fast,
  graph_to_pyg_data, greedy select_and_price) but with BudgetRevenueEnv
  to ensure fair comparison with LSTM baselines and profit metrics.

Usage:
  venv/bin/python3 -u experiments/eval_transformer_current_protocol.py \\
    > /tmp/transformer_eval.log 2>&1
"""
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import (generate_forest_fire,
                                       generate_modular_forest_fire,
                                       load_rice_facebook)
from src.utils.helpers import set_seed, load_config_with_base
from src.utils.features import (compute_static_features, build_graph_feature_cache,
                                 compute_node_features_fast)
from src.utils.helpers import graph_to_pyg_data, get_available_mask
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.models.policies.transformer_joint_policy import TransformerJointPolicy
from src.evaluation.budget_baselines import _make_env as _make_budget_env

_ROOT    = str(Path(__file__).parent.parent)
CKPT_TFM = os.path.join(_ROOT, "results", "checkpoints", "rev_gnn_transformer.pt")
CKPT_SHA = "c24215b8"
CFG_PATH = os.path.join(_ROOT, "configs", "experiments", "rev_gnn_transformer_300ep.yaml")
LOG_DIR  = os.path.join(_ROOT, "results", "logs")
OUT      = os.path.join(LOG_DIR, "transformer_current_protocol.json")

C      = 0.3
W_HIGH = 2.0
N_MC   = 200
SEEDS  = list(range(10))
KAPPAS = [5, 20]

NETWORKS = {
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
    "Modular_FF": lambda: generate_modular_forest_fire([200, 300, 500], 0.37, 0.32, 0.01, seed=0),
    "Rice_FB":    load_rice_facebook,
}


# ── Load policy ───────────────────────────────────────────────────────────────

def load_transformer(ckpt_path, device):
    cfg = load_config_with_base(CFG_PATH)
    enc = GraphSAGEEncoder(
        int(cfg.features.dim), int(cfg.encoder.hidden_dim),
        int(cfg.encoder.n_layers), float(cfg.encoder.dropout),
    )
    tfm = EpisodeTransformerSliding.from_config(cfg.transformer)
    pol = TransformerJointPolicy(enc, tfm,
                                  gnn_dim=int(cfg.encoder.hidden_dim),
                                  context_dim=tfm.context_dim)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    pol.load_state_dict(sd)
    pol.to(device).eval()
    return pol


# ── Episode runner ────────────────────────────────────────────────────────────

def run_episode(pol, G, cache, B0, seed, device, budget=True):
    """Run one greedy episode.  Returns (revenue, profit, n_below_cost, S_T)."""
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C if budget else 0.0,
                          seed=seed, weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes = list(G.nodes()); n = len(nodes)
    pol.reset_episode(device)

    revenue = 0.0; n_below = 0
    S_set = frozenset(); off_set = frozenset()

    while True:
        if budget and env._check_bankrupt():
            break
        if len(off_set) >= n:
            break

        feats = compute_node_features_fast(cache, S_set, set(off_set), len(off_set), n, env)
        data  = graph_to_pyg_data(G, feats, device)
        mask  = get_available_mask(n, off_set, nodes, device)
        if mask.sum() == 0:
            break

        with torch.no_grad():
            ni, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)

        node = nodes[int(ni)]
        d    = float(disc)
        val  = env._true_valuation(node)
        price = val * (1 - d)
        acc  = val >= price

        off_set = frozenset(off_set | {node})
        env.offered.add(node); env.t += 1

        if acc:
            S_set = frozenset(S_set | {node}); env.S.add(node)
            env._influence_cache = {}
            revenue += price
            if budget:
                env.B = env.B - C + price
                env.budget_history.append(env.B)
            if price < C:
                n_below += 1

        pol.update_sequence_state(d, acc, price if acc else 0.0)

    S_T    = len(S_set)
    profit = revenue - C * S_T if budget else revenue
    return revenue, profit, n_below, S_T


# ── Cell stats ────────────────────────────────────────────────────────────────

def _cell(revs, profits=None, bcs=None, sts=None):
    def _s(v):
        v = [x for x in v if x == x]
        return (round(float(np.mean(v)),2), round(float(np.std(v)),2)) if v else (float('nan'), float('nan'))
    out = {"rev": _s(revs)}
    if profits is not None: out["profit"] = _s(profits)
    if bcs     is not None: out["n_below"] = _s(bcs)
    if sts     is not None: out["S_T"] = _s(sts)
    return out


# ── Comparison table (load LSTM and CGS from existing logs) ───────────────────

def _load_existing_results():
    """Load LSTM unc results from ksweep or sanity logs for comparison."""
    existing = {}
    sanity = os.path.join(LOG_DIR, "sanity_unified_ff1000_k10.json")
    if os.path.exists(sanity):
        try:
            d = json.load(open(sanity))
            existing["LSTM_FF1000_unc"] = d.get("mean", None)
        except Exception:
            pass
    return existing


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--ckpt",   default=CKPT_TFM)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"=== Transformer eval — current protocol ===")
    print(f"Checkpoint : {args.ckpt}")
    print(f"SHA (first8): {CKPT_SHA}")
    print(f"Gate A score: 463.84±5.26 (FF_1000, 20 seeds)")
    print(f"Gate A protocol:")
    print(f"  env=RevenueEnv (unconstrained, baselines._make_env)")
    print(f"  influence=Babaei monotone-flat, weights Uniform(0,{W_HIGH})")
    print(f"  n_mc=from config, seeds=20 (primary) / 5 (OOD)")
    print(f"  cost=0.0 (no production cost), no budget constraint")
    print(f"Current protocol:")
    print(f"  env=BudgetRevenueEnv, C={C}, B_unc=1e7")
    print(f"  influence=monotone+flat, Uniform(0,{W_HIGH}), n_mc={N_MC}")
    print(f"  seeds={SEEDS}", flush=True)

    pol = load_transformer(args.ckpt, device)
    print(f"Loaded: {sum(p.numel() for p in pol.parameters()):,} params", flush=True)

    all_results = {}

    # ── Header ────────────────────────────────────────────────────────────────
    def hdr(mode): print(f"\n{'─'*80}\n{mode}\n{'─'*80}")

    hdr("=== UNCONSTRAINED (B=1e7) ===")
    print(f"{'net':12s} {'method':12s}  {'rev':>10s} ± {'std':>6s}  {'|S_T|':>6s}", flush=True)

    B_UNC = 1e7
    net_results = {}
    for net_name, gfn in NETWORKS.items():
        G = gfn()
        cache = build_graph_feature_cache(G, compute_static_features(G))
        revs = []
        for seed in SEEDS:
            try:
                rev, _, _, st = run_episode(pol, G, cache, B_UNC, seed, device, budget=False)
                revs.append(rev)
            except Exception as e:
                print(f"  {net_name} seed={seed} error: {e}", flush=True)
                revs.append(float('nan'))
        m, s = (np.nanmean(revs), np.nanstd(revs))
        print(f"{net_name:12s} {'Transformer':12s}  {m:10.2f} ± {s:6.2f}", flush=True)
        net_results[net_name] = {"unc": _cell(revs)}

    hdr("=== BUDGET EVAL (C=0.3, kappa in {5,20}) ===")
    print(f"{'net':12s} {'kap':4s} {'method':12s}  "
          f"{'profit':>9s} ± {'std':>5s}  "
          f"{'revenue':>9s} ± {'std':>5s}  "
          f"{'below':>5s}  {'|S_T|':>5s}", flush=True)

    for net_name, gfn in NETWORKS.items():
        G = gfn()
        cache = build_graph_feature_cache(G, compute_static_features(G))
        for kappa in KAPPAS:
            B0 = kappa * C
            revs=[]; profits=[]; bcs=[]; sts=[]
            for seed in SEEDS:
                try:
                    rev, profit, bc, st = run_episode(pol, G, cache, B0, seed, device, budget=True)
                    revs.append(rev); profits.append(profit)
                    bcs.append(bc); sts.append(st)
                except Exception as e:
                    print(f"  {net_name} kap={kappa} seed={seed} error: {e}", flush=True)
                    revs.append(float('nan')); profits.append(float('nan'))
                    bcs.append(float('nan')); sts.append(float('nan'))
            c_ = _cell(revs, profits, bcs, sts)
            print(f"{net_name:12s} {kappa:4d} {'Transformer':12s}  "
                  f"{c_['profit'][0]:9.2f} ±{c_['profit'][1]:5.2f}  "
                  f"{c_['rev'][0]:9.2f} ±{c_['rev'][1]:5.2f}  "
                  f"{c_['n_below'][0]:5.1f}  {c_['S_T'][0]:5.1f}", flush=True)
            net_results[net_name][f"k{kappa}"] = c_

    all_results["Transformer"] = net_results
    os.makedirs(LOG_DIR, exist_ok=True)
    json.dump(all_results, open(OUT, "w"), indent=2)
    print(f"\nSaved → {OUT}", flush=True)


if __name__ == "__main__":
    main()
