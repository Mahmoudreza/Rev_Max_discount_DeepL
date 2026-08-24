#!/usr/bin/env python3
"""eval_skip_rice.py — Rice-FB eval for TransformerSkipPolicy vs baselines.

Same protocol as eval_transf_rice.py but loads TransformerSkipPolicy
(z_i = [h, ctx, s] = 132-dim) and also runs the base transformer for
direct side-by-side comparison.

Usage:
  venv/bin/python3 -u experiments/eval_skip_rice.py \
    --skip_ckpt results/checkpoints/transf_skip_s0_best.pt \
    --base_ckpt results/checkpoints/transf_budget_s0_best.pt \
    --device cuda:0 > /tmp/skip_rice.log 2>&1
"""
from __future__ import annotations
import argparse, json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

import hashlib
import numpy as np
import scipy.stats as sc
import torch

_ROOT  = str(Path(__file__).parent.parent)
C      = 0.3; W_HIGH = 2.0; N_MC = 200; SEEDS = list(range(10))
KAPPAS = [5, 10, 20, 40]
SKIP_IDX = [17, 16, 0, 20]   # must match train_skip_transformer.py


# ── Helpers ────────────────────────────────────────────────────────────────────

def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]

def _ei(G, device):
    m={v:i for i,v in enumerate(G.nodes())}
    edges=list(G.edges())
    if not edges: return torch.zeros((2,0),dtype=torch.long,device=device)
    s=[m[u] for u,_ in edges]+[m[v] for _,v in edges]
    d=[m[v] for _,v in edges]+[m[u] for u,_ in edges]
    return torch.tensor([s,d],dtype=torch.long,device=device)

def _avail(env,n,device):
    mask=torch.zeros(n,dtype=torch.bool,device=device)
    for idx in env.available_nodes: mask[idx]=True
    return mask

def _feats21(cache,env,k):
    from src.utils.features import compute_node_features_fast
    base=compute_node_features_fast(cache,env.S,env.offered,env.t,k,env)
    bcol=np.full((cache["n"],1),env.B/(40*C),dtype=np.float32)
    return np.concatenate([base,bcol],axis=1)

def _skip_tensor(f21,device):
    return torch.tensor(f21[:,SKIP_IDX].copy(),dtype=torch.float32,device=device)


# ── Load policies ─────────────────────────────────────────────────────────────

def _load_skip(ckpt_path, device):
    from train_skip_transformer import TransformerSkipPolicy, SKIP_DIM
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    cfg = load_config_with_base(
        os.path.join(_ROOT,"configs/experiments/rev_gnn_transformer_300ep.yaml"))
    H,NL,DO=(int(cfg.encoder.hidden_dim),int(cfg.encoder.n_layers),
             float(cfg.encoder.dropout))
    enc=GraphSAGEEncoder(int(cfg.features.dim)+1,H,NL,DO)
    tfm=EpisodeTransformerSliding.from_config(cfg.transformer)
    pol=TransformerSkipPolicy(enc,tfm,gnn_dim=H,
                               context_dim=tfm.context_dim,skip_dim=SKIP_DIM).to(device)
    sd=torch.load(ckpt_path,map_location=device,weights_only=True)
    pol.load_state_dict(sd,strict=True); pol.eval()
    sha=_sha8(ckpt_path)
    print(f"Skip loaded sha={sha}  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol, sha

def _load_base(ckpt_path, device):
    from src.utils.helpers import load_config_with_base
    from src.models.encoders.graphsage import GraphSAGEEncoder
    from src.models.encoders.episode_transformer import EpisodeTransformerSliding
    from src.models.policies.transformer_joint_policy import TransformerJointPolicy
    cfg = load_config_with_base(
        os.path.join(_ROOT,"configs/experiments/rev_gnn_transformer_300ep.yaml"))
    H,NL,DO=(int(cfg.encoder.hidden_dim),int(cfg.encoder.n_layers),
             float(cfg.encoder.dropout))
    enc=GraphSAGEEncoder(int(cfg.features.dim)+1,H,NL,DO)
    tfm=EpisodeTransformerSliding.from_config(cfg.transformer)
    pol=TransformerJointPolicy(enc,tfm,gnn_dim=H,context_dim=tfm.context_dim).to(device)
    sd=torch.load(ckpt_path,map_location=device,weights_only=True)
    pol.load_state_dict(sd,strict=True); pol.eval()
    sha=_sha8(ckpt_path)
    print(f"Base  loaded sha={sha}  params={sum(p.numel() for p in pol.parameters()):,}")
    return pol, sha


# ── Rollouts ───────────────────────────────────────────────────────────────────

def _rollout_skip(pol, G, cache, ei_t, B0, k, seed, device, deg_order=None):
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.helpers import set_seed
    set_seed(seed)
    cfg=BudgetEnvConfig(budget_B=B0,production_cost=C,seed=seed,
                        weight_high=W_HIGH,n_mc_samples=N_MC)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    nodes=list(G.nodes()); n=len(nodes)
    pol.reset_episode(device); revenue=0.0; n_below=0
    B_start=env.B
    while env.available_nodes and not env._check_bankrupt():
        f21=_feats21(cache,env,k)
        x  =torch.tensor(f21,dtype=torch.float32,device=device)
        sk =_skip_tensor(f21,device)
        av =_avail(env,n,device)
        if not av.any(): break
        with torch.no_grad():
            scores,h,ctx,_=pol.forward(x,ei_t,av,sk)
        if deg_order is not None:
            ni=next((i for i in deg_order if av[i]),None)
            if ni is None: break
        else:
            safe=scores.clone(); safe[~av]=-1e9; ni=int(safe.argmax())
        disc=float(pol.get_discount_distribution(
            torch.cat([h[ni],ctx,sk[ni]])).mean.clamp(1e-4,1-1e-4))
        _,r,done,_=env.step(ni,disc)
        revenue+=r
        if 0<r<C: n_below+=1
        pol.update_sequence_state(disc,r>0,r)
        if done: break
    S_T=len(env.S)
    profit=revenue-C*S_T
    pi_check=env.B-B_start
    if abs(profit-pi_check)>0.01:
        print(f"  ASSERT FAIL profit={profit:.3f} pi_check={pi_check:.3f}")
    return revenue, profit, n_below, S_T

def _rollout_base(pol, G, cache, ei_t, B0, k, seed, device):
    from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
    from src.utils.helpers import set_seed
    set_seed(seed)
    cfg=BudgetEnvConfig(budget_B=B0,production_cost=C,seed=seed,
                        weight_high=W_HIGH,n_mc_samples=N_MC)
    env=BudgetRevenueEnv(G,cfg); env.reset()
    nodes=list(G.nodes()); n=len(nodes)
    pol.reset_episode(device); revenue=0.0
    B_start=env.B
    while env.available_nodes and not env._check_bankrupt():
        f21=_feats21(cache,env,k)
        x  =torch.tensor(f21,dtype=torch.float32,device=device)
        av =_avail(env,n,device)
        if not av.any(): break
        with torch.no_grad():
            scores,h,ctx,_=pol.forward(x,ei_t,av)
        safe=scores.clone(); safe[~av]=-1e9; ni=int(safe.argmax())
        disc=float(pol.get_discount_distribution(
            torch.cat([h[ni],ctx])).mean.clamp(1e-4,1-1e-4))
        _,r,done,_=env.step(ni,disc)
        revenue+=r; pol.update_sequence_state(disc,r>0,r)
        if done: break
    S_T=len(env.S); profit=revenue-C*S_T
    return revenue, profit, S_T


# ── Sweep helpers ─────────────────────────────────────────────────────────────

def _sweep_skip(pol, G, cache, ei_t, k, device, deg_order=None):
    B0=k*C; profs=[]; s_ts=[]
    for seed in SEEDS:
        try:
            rev,prof,bc,s_t=_rollout_skip(pol,G,cache,ei_t,B0,k,seed,device,deg_order)
        except Exception as e:
            print(f"  ERR skip seed={seed}: {e}"); prof=s_t=float("nan")
        profs.append(prof); s_ts.append(s_t)
    return float(np.nanmean(profs)), float(np.nanstd(profs)), profs

def _sweep_base(pol, G, cache, ei_t, k, device):
    B0=k*C; profs=[]
    for seed in SEEDS:
        try:
            _,prof,_=_rollout_base(pol,G,cache,ei_t,B0,k,seed,device)
        except Exception as e:
            print(f"  ERR base seed={seed}: {e}"); prof=float("nan")
        profs.append(prof)
    return float(np.nanmean(profs)), float(np.nanstd(profs)), profs

def _baselines(G, k):
    B0=k*C; out={}
    try:
        from src.evaluation.ie_budget import ie_strategy_budget_aware
        ag=ie_strategy_budget_aware(G,B0,C,n_trials=len(SEEDS),weight_high=W_HIGH)
        rs=ag.get("revenue",{}).get("all",[]); ss=ag.get("n_in_S",{}).get("all",[])
        out["ie"]=float(np.mean([r-C*s for r,s in zip(rs,ss)]))
    except Exception as e: out["ie"]=float("nan")
    try:
        from src.evaluation.greedy_budget_faithful import greedy_discount_budget_faithful
        gf=greedy_discount_budget_faithful(G,B0,C,n_trials=len(SEEDS),weight_high=W_HIGH)
        out["gf"]=gf["profit"]["mean"]
    except Exception as e: out["gf"]=float("nan")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--skip_ckpt",required=True)
    ap.add_argument("--base_ckpt",default="")
    ap.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    args=ap.parse_args()
    device=torch.device(args.device if torch.cuda.is_available() else "cpu")

    from src.env.graph_generators import load_rice_facebook
    from src.utils.features import compute_static_features, build_graph_feature_cache
    G=load_rice_facebook()
    cache=build_graph_feature_cache(G,compute_static_features(G))
    ei_t=_ei(G,device)
    deg_order=sorted(range(G.number_of_nodes()),
                     key=lambda i:G.degree(list(G.nodes())[i]),reverse=True)
    print(f"Rice-FB  n={G.number_of_nodes()}  m={G.number_of_edges()}", flush=True)

    pol_s, sha_s = _load_skip(args.skip_ckpt, device)
    pol_b = None; sha_b = "—"
    if args.base_ckpt and os.path.exists(args.base_ckpt):
        try: pol_b, sha_b = _load_base(args.base_ckpt, device)
        except Exception as e: print(f"Base load failed: {e}")

    results={}
    hdr=f"{'k':3s}  {'skip':>10s}±std   {'base':>10s}±std   {'diff':>7s}  p  {'IE':>8s}  {'GF':>8s}  {'deg(skip)':>10s}"
    print(f"\n=== Rice-FB skip sha={sha_s} vs base sha={sha_b} ===")
    print(hdr); print("-"*len(hdr))

    for k in KAPPAS:
        sm,ss,sprofs=_sweep_skip(pol_s,G,cache,ei_t,k,device)
        pm_deg,_,_=_sweep_skip(pol_s,G,cache,ei_t,k,device,deg_order=deg_order)
        bm=float("nan"); bprofs=[]
        if pol_b: bm,_,bprofs=_sweep_base(pol_b,G,cache,ei_t,k,device)
        bl=_baselines(G,k)
        diff=float("nan"); pval=float("nan")
        if bprofs and len(bprofs)==len(sprofs):
            td,pval=sc.ttest_rel(sprofs,bprofs,alternative="two-sided")
            diff=float(np.mean(sprofs))-float(np.mean(bprofs))
        print(f"  {k:2d}  {sm:+9.2f}±{ss:5.2f}   {bm:+9.2f}        "
              f"{diff:+7.2f}  {pval:.4f}  {bl['ie']:+8.2f}  {bl['gf']:+8.2f}  {pm_deg:+9.2f}",
              flush=True)
        results[f"k{k}"]={"skip_profit":[sm,ss],"base_profit":bm,
                           "diff":diff,"p":pval,
                           "ie":bl["ie"],"gf":bl["gf"],
                           "skip_deg":pm_deg,"all_skip":sprofs}

    # Summary
    print(f"\n=== VERDICT ===")
    positive_k=[k for k in KAPPAS if results[f"k{k}"]["skip_profit"][0]>0]
    positive_vs_base=[k for k in KAPPAS
                      if results[f"k{k}"]["diff"]>0]
    print(f"  Skip profit > 0 at k: {positive_k}")
    print(f"  Skip > base at k:     {positive_vs_base}")
    if not positive_k:
        print("  CONCLUSION: skip connection did not fix Rice-FB negativity.")
        print("  Five interventions failed — report limitation.")
    else:
        print("  CONCLUSION: skip connection fixed k=" + str(positive_k))

    out=os.path.join(_ROOT,"results/logs/skip_rice.json")
    os.makedirs(os.path.dirname(out),exist_ok=True)
    json.dump({"sha_skip":sha_s,"sha_base":sha_b,"results":results},
              open(out,"w"),indent=2)
    print(f"\nSaved → {out}", flush=True)

    import subprocess
    subprocess.run(["git","add","-f",out],cwd=_ROOT)
    subprocess.run(["git","commit","-m",f"skip_rice eval sha={sha_s}"],cwd=_ROOT)
    h=subprocess.run(["git","rev-parse","--short","HEAD"],
                      capture_output=True,text=True,cwd=_ROOT).stdout.strip()
    print(h, flush=True)

if __name__=="__main__":
    main()
