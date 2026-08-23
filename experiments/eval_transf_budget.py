#!/usr/bin/env python3
"""eval_transf_budget.py — Quick eval of any transf_budget_s*_p1_ep*.pt checkpoint.

Runs BudgetRevenueEnv, reports profit / below-cost / |S_T| / revenue.
Can run in parallel while training continues on other GPUs.

Usage:
  venv/bin/python3 -u experiments/eval_transf_budget.py \
    --ckpt results/checkpoints/transf_budget_s0_p1_ep100.pt \
    --label "s0_p1_ep100" \
    --device cuda:3
"""
from __future__ import annotations
import argparse, hashlib, os, sys
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
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.episode_transformer import EpisodeTransformerSliding
from src.models.policies.transformer_joint_policy import TransformerJointPolicy

_ROOT    = str(Path(__file__).parent.parent)
CFG_TFM  = os.path.join(_ROOT, "configs/experiments/rev_gnn_transformer_300ep.yaml")
C        = 0.3
B_MAX    = 40 * C       # 12.0
W_HIGH   = 2.0
N_MC     = 200
SEEDS    = list(range(10))
KAPPAS   = [5, 10, 20, 40]
FF_P, FF_PB = 0.37, 0.32

NETWORKS = {
    "FF_1000":    lambda: generate_forest_fire(1000, FF_P, FF_PB, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, FF_P, FF_PB, seed=1),
    "Modular_FF": lambda: generate_modular_forest_fire([200,300,500], FF_P, FF_PB, 0.01, seed=0),
    "Rice_FB":    load_rice_facebook,
}
try:
    from src.env.graph_generators import load_polblogs
    NETWORKS["polblogs"] = load_polblogs
except ImportError:
    pass

def _sha8(p):
    try: return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]
    except: return "????????"

def _features(cache, env, k):
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k, env)
    bcol = np.full((cache["n"],1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, bcol], axis=1)

def _ei(G, device):
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    m = {v:i for i,v in enumerate(G.nodes())}
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)

def _avail(env, n, device):
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for idx in env.available_nodes: mask[idx]=True
    return mask

def load_model(ckpt_path, device):
    cfg_t  = load_config_with_base(CFG_TFM)
    in_dim = int(cfg_t.features.dim)+1  # 21-dim
    H=int(cfg_t.encoder.hidden_dim); NL=int(cfg_t.encoder.n_layers); DO=float(cfg_t.encoder.dropout)
    enc = GraphSAGEEncoder(in_dim, H, NL, DO)
    tfm = EpisodeTransformerSliding.from_config(cfg_t.transformer)
    pol = TransformerJointPolicy(enc, tfm, gnn_dim=H, context_dim=tfm.context_dim).to(device)
    sd = torch.load(ckpt_path, map_location=device, weights_only=True)
    if isinstance(sd,dict) and 'state_dict' in sd: sd=sd['state_dict']
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    return pol

def run_episode(pol, G, cache, ei_t, B0, k, seed, device):
    """Greedy episode. Returns (revenue, profit, n_below, S_T)."""
    set_seed(seed)
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(G, cfg); env.reset()
    nodes=list(G.nodes()); n=len(nodes)
    pol.reset_episode(device)
    revenue=0.0; n_below=0

    while env.available_nodes and not env._check_bankrupt():
        feats = _features(cache, env, k)
        x = torch.tensor(feats, dtype=torch.float32, device=device)
        av = _avail(env, n, device)
        if not av.any(): break
        with torch.no_grad():
            scores, h, ctx, _ = pol.forward(x, ei_t, av)
            safe = scores.clone(); safe[~av]=-1e9
            ni = int(safe.argmax())
            disc = float(pol.get_discount_distribution(
                torch.cat([h[ni], ctx])).mean.clamp(1e-4,1-1e-4))
        _, r, done, _ = env.step(ni, disc)  # ni is 0-based index
        revenue += r
        if 0 < r < C:  # accepted at below-cost price
            n_below += 1
        pol.update_sequence_state(disc, r>0, r)
        if done: break

    S_T = len(env.S)
    profit = revenue - C * S_T
    return revenue, profit, n_below, S_T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt",   required=True)
    ap.add_argument("--label",  default="")
    ap.add_argument("--device", default="cuda:3" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sha = _sha8(args.ckpt)
    lbl = args.label or os.path.basename(args.ckpt).replace(".pt","")
    print(f"=== eval_transf_budget  ckpt={lbl}  sha={sha}  device={device} ===")
    print(f"Networks: {list(NETWORKS.keys())}")
    print(f"Kappas: {KAPPAS}  Seeds: {SEEDS}  C={C}  N_MC={N_MC}", flush=True)

    pol = load_model(args.ckpt, device)
    print(f"Loaded: {sum(p.numel() for p in pol.parameters()):,} params  in_dim=21", flush=True)

    hdr = f"{'net':12s} {'k':3s} {'profit':>8s}±{'std':>5s}  {'below':>5s}  {'|S_T|':>5s}  {'rev':>8s}±{'std':>5s}"
    print(hdr)
    print("-"*len(hdr))

    results = {}
    for net_name, gfn in NETWORKS.items():
        try:
            G = gfn()
        except Exception as e:
            print(f"[skip] {net_name}: {e}"); continue
        cache = build_graph_feature_cache(G, compute_static_features(G))
        ei_t  = _ei(G, device)
        results[net_name] = {}

        for k in KAPPAS:
            B0 = k * C
            profs=[]; revs=[]; bcs=[]; sts=[]
            for seed in SEEDS:
                try:
                    rev, prof, bc, st = run_episode(pol, G, cache, ei_t, B0, k, seed, device)
                    profs.append(prof); revs.append(rev); bcs.append(bc); sts.append(st)
                except Exception as e:
                    print(f"  ERR {net_name} k={k} s={seed}: {e}", flush=True)
                    profs.append(float('nan')); revs.append(float('nan'))
                    bcs.append(float('nan')); sts.append(float('nan'))
            pm = float(np.nanmean(profs)); ps = float(np.nanstd(profs))
            rm = float(np.nanmean(revs));  rs = float(np.nanstd(revs))
            bc = float(np.nanmean(bcs));   st = float(np.nanmean(sts))
            print(f"{net_name:12s} {k:3d} {pm:8.2f}±{ps:5.2f}  {bc:5.1f}  {st:5.1f}  {rm:8.2f}±{rs:5.2f}", flush=True)
            results[net_name][f"k{k}"] = {"profit":[pm,ps],"rev":[rm,rs],"bc":bc,"st":st}


    import json
    out = os.path.join(_ROOT, f"results/logs/eval_transf_{lbl}.json")
    json.dump({"ckpt": args.ckpt, "sha": sha, "results": results}, open(out,"w"), indent=2)
    print(f"\nSaved → {out}", flush=True)

if __name__ == "__main__":
    main()
