#!/usr/bin/env python3
"""diag_clip_prices.py — Diagnose clip vs env v_hat mismatch.

Q1a: The clip uses f21[ni, 17] = env._apply_influence_model(
         env.get_current_influence(ni)) — deterministic given current S.
     The env.step uses env._estimate_valuation(node) — stochastic MC estimate
     (fresh link weight draws each call). They are NOT the same.
     Price posted = est_val_fresh * (1 - d).

Q1b: Print first 10 posted prices for Modular k=20 seed=0,
     clipped and unclipped, and count below C=0.3.

Usage:
  venv/bin/python3 -u experiments/diag_clip_prices.py \
    --transf_ckpt results/checkpoints/transf_budget_s0_best.pt \
    --device cpu
"""
from __future__ import annotations
import argparse, hashlib, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch

_ROOT = str(Path(__file__).parent.parent)
C      = 0.3
W_HIGH = 2.0
N_MC   = 200
V_HAT_COL = 17


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _ei(G, device):
    m = {v:i for i,v in enumerate(G.nodes())}
    edges = list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)

def _avail(env, n, device):
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for i in env.available_nodes: mask[i]=True
    return mask

def _feats21(cache, env, k):
    from src.utils.features import compute_node_features_fast
    base=compute_node_features_fast(cache,env.S,env.offered,env.t,k,env)
    bcol=np.full((cache["n"],1),env.B/(40*C),dtype=np.float32)
    return np.concatenate([base,bcol],axis=1)

def _clip(d_raw, v_hat_i):
    d_max = max(0.0, 1.0-C/v_hat_i) if v_hat_i>C else 0.0
    return min(float(d_raw), d_max)

def _load_transformer(ckpt_path, device):
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy
    cfg=load_config_with_base(os.path.join(_ROOT,"configs/experiments/rev_gnn_transformer_300ep.yaml"))
    H,NL,DO=int(cfg.encoder.hidden_dim),int(cfg.encoder.n_layers),float(cfg.encoder.dropout)
    enc=GraphSAGEEncoder(int(cfg.features.dim)+1,H,NL,DO)
    tfm=EpisodeTransformerSliding.from_config(cfg.transformer)
    pol=TransformerJointPolicy(enc,tfm,gnn_dim=H,context_dim=tfm.context_dim).to(device)
    sd=torch.load(ckpt_path,map_location=device,weights_only=True)
    if isinstance(sd,dict) and "state_dict" in sd: sd=sd["state_dict"]
    pol.load_state_dict(sd,strict=True); pol.eval()
    return pol

def _rollout_debug(pol, G, cache, ei_t, B0, k, seed, device, do_clip, max_steps=10):
    """Run episode, return list of dicts with price trace."""
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.helpers import set_seed
    set_seed(seed)
    n=G.number_of_nodes()
    cfg=BudgetEnvConfig(budget_B=B0,production_cost=C,seed=seed,
                        weight_high=W_HIGH,n_mc_samples=N_MC)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    pol.reset_episode(device)
    log=[]; step=0

    while env.available_nodes and not env._check_bankrupt() and step<max_steps+20:
        f21=_feats21(cache,env,k)
        x=torch.tensor(f21,dtype=torch.float32,device=device)
        av=_avail(env,n,device)
        if not av.any(): break
        with torch.no_grad():
            sc,h,ctx,_=pol.forward(x,ei_t,av)
        safe=sc.clone(); safe[~av]=-1e9
        ni=int(safe.argmax())

        d_raw=float(pol.get_discount_distribution(
            torch.cat([h[ni],ctx])).mean.clamp(1e-4,1-1e-4).detach())

        # v_hat from features (col 17 = _apply_influence_model(get_current_influence))
        v_hat_feat = float(f21[ni, V_HAT_COL])

        if do_clip:
            d = _clip(d_raw, v_hat_feat)
        else:
            d = d_raw

        # What the env actually uses for price (stochastic MC estimate, fresh draw)
        # We can peek at it by calling _estimate_valuation before step
        nodes = list(G.nodes())
        v_hat_env = float(env._estimate_valuation(nodes[ni]))
        price_actual = v_hat_env * (1.0 - d)

        _,r,done,info=env.step(ni,d)
        offered_price = info.get("offered_price", 0.0)
        pol.update_sequence_state(d,r>0,r)

        row = {
            "step": step,
            "ni": ni,
            "d_raw": round(d_raw,4),
            "d_posted": round(d,4),
            "v_hat_feat(17)": round(v_hat_feat,4),
            "v_hat_env(MC)": round(v_hat_env,4),
            "price_pre": round(price_actual,4),     # v_hat_env_diag * (1-d) (one fresh draw)
            "price_info": round(offered_price,4),   # actual env price (another fresh draw)
            "revenue": round(r,4),
            "below_c": offered_price < C,           # uses actual env price
        }
        if len(log) < max_steps:
            log.append(row)
        step += 1
        if done: break

    return log


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--transf_ckpt",default=os.path.join(
        _ROOT,"results/checkpoints/transf_budget_s0_best.pt"))
    ap.add_argument("--device",default="cpu")
    args=ap.parse_args()
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")

    from src.env.graph_generators import generate_modular_forest_fire
    from src.utils.features import compute_static_features, build_graph_feature_cache
    G=generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0)
    cache=build_graph_feature_cache(G,compute_static_features(G))
    ei_t=_ei(G,device)
    pol=_load_transformer(args.transf_ckpt,device)
    sha=_sha8(args.transf_ckpt)

    k=20; seed=0; B0=k*C

    print(f"=== Q1: v_hat source disambiguation ===")
    print(f"  f21[ni,17] = env._apply_influence_model(env.get_current_influence(ni))")
    print(f"             = DETERMINISTIC given current S and realized link weights")
    print(f"  env.step internally: est_val = env._estimate_valuation(node)")
    print(f"             = MC average over FRESH link weight samples (N_MC={N_MC})")
    print(f"  price posted to buyer = est_val_fresh * (1 - d)")
    print(f"  Clip uses f21[ni,17]; env uses est_val_fresh. These are DIFFERENT.")
    print(f"  Clip guarantees: (1-d) >= C/f21[ni,17]")
    print(f"  But actual price = est_val_fresh*(1-d); may be < C if est_val_fresh < f21[ni,17]")
    print()

    for do_clip in [False, True]:
        label = "CLIPPED  " if do_clip else "UNCLIPPED"
        log=_rollout_debug(pol,G,cache,ei_t,B0,k,seed,device,do_clip,max_steps=10)
        n_below=sum(1 for r in log if r["below_c"])
        print(f"=== Modular_FF k=20 seed=0  {label}  sha={sha} ===")
        print(f"  {'step':>4}  {'ni':>5}  {'d_raw':>6}  {'d_post':>6}  "
              f"{'v_hat_17':>9}  {'v_hat_MC':>9}  {'price_MC':>9}  {'price_env':>9}  {'below_C':>7}")
        for r in log:
            print(f"  {r['step']:>4}  {r['ni']:>5}  {r['d_raw']:>6.4f}  {r['d_posted']:>6.4f}  "
                  f"{r['v_hat_feat(17)']:>9.4f}  {r['v_hat_env(MC)']:>9.4f}  "
                  f"{r['price_pre']:>9.4f}  {r['price_info']:>9.4f}  "
                  f"{'YES' if r['below_c'] else 'no':>7}")
        print(f"  → below_C count (first {len(log)} steps): {n_below}/{len(log)}")
        print()

    print("=== Q2: arm_b profit discrepancy ===")
    print("Scripts compared:")
    print("  eval_clipped.py:         W_HIGH=2.0  N_MC=200  seeds=[0..9]"
          "  profit=revenue-C*|S_T|  sha=0b549f93")
    print("  _arm_b_utils.py/         W_HIGH=1.0  N_MC=5    seeds=[0..9]")
    print("  budget_sweep_10seed.py   profit=revenue-C*|S_T|  sha=0b549f93")
    print()
    print("Same checkpoint, same profit formula, same seeds.")
    print("ONLY difference: W_HIGH=2.0 (eval_clipped) vs W_HIGH=1.0 (_arm_b_utils).")
    print("With W_HIGH=2.0: E[buyer value] = 1.0 >> C=0.3 → budget grows rapidly,")
    print("more rounds, higher cumulative revenue → +91.4 on Rice k=40.")
    print("With W_HIGH=1.0: E[buyer value] = 0.5 > C=0.3 but many draws < C,")
    print("more bankruptcies, fewer accepted → negative profit.")
    print()
    print("eval_clipped.py evaluates arm_b OOD (W_HIGH=2.0 ≠ arm_b training W_HIGH=1.0).")
    print("arm_b Rice k=40 result of +91.4 is NOT comparable to prior results.")


if __name__=="__main__":
    main()
