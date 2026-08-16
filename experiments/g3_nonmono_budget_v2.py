#!/usr/bin/env python3
"""
experiments/g3_nonmono_budget_v2.py — G3 (fixed import)
=========================================================
All methods under NON-MONOTONE Rayleigh valuations, budget protocol,
FF_1000 and Rice_FB, k=[5,10,20,40], 10 seeds.

Fix v2: correct ie_strategy_budget import path.
Writes: results/logs/g3_nonmono_budget_10seed.json  (new file on server)
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.budget_baselines import greedy_discount_budget
try:
    from src.evaluation.ie_budget import ie_strategy_budget
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget
from _arm_b_utils import _feat_unconstrained, make_ei as _make_ei
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.helpers import set_seed
import hashlib

C = 0.3; W_HIGH = 1.0
SEEDS = list(range(10))
KS    = [5, 10, 20, 40]
NETS  = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
         "Rice_FB": load_rice_facebook}
ARM_B   = "results/checkpoints/rev_gnn_lstm_densemix.pt"
ARM_SHA = "0b549f93"
NM_KW   = {"acceptance_mode": "nonmonotone"}
OUT     = "results/logs/g3_nonmono_budget_10seed.json"


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]


def _load_policy(device):
    assert _sha8(ARM_B) == ARM_SHA, f"sha mismatch: {_sha8(ARM_B)}"
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lst = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol = SequentialJointPolicy(enc, lst, gnn_dim=64, context_dim=64)
    pol.load_state_dict(torch.load(ARM_B, map_location="cpu"))
    return pol.to(device).eval()


@torch.no_grad()
def _run_policy(pol, graph, cache, ei, k, seed, device):
    set_seed(seed)
    n   = graph.number_of_nodes()
    cfg = BudgetEnvConfig(budget_B=k*C, production_cost=C, seed=seed,
                          weight_high=W_HIGH, **NM_KW)
    env = BudgetRevenueEnv(graph, cfg); env.reset(); pol.reset_episode(device)
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = torch.zeros(n, dtype=torch.bool, device=device)
        for i in env.available_nodes: av[i] = True
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax())
        d  = float(pol.get_discount_distribution(torch.cat([h[ni], ctx])).mean)
        _, _, done, info = env.step(ni, d)
        pol.update_sequence_state(d, info["accepted"], info.get("revenue_step", 0.))
        if done: break
    return float(env.total_revenue)


def _stats(v): a=np.array(v,dtype=float); return {"mean":round(float(a.mean()),2),"std":round(float(a.std()),2),"all":v}


def run_net(net, graph, pol, device):
    ei, cache = _make_ei(graph, device)
    out = {}
    for k in KS:
        B = k*C
        cfg0 = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0, weight_high=W_HIGH, **NM_KW)
        ie_v, gd_v, cdp_v, gnn_v = [], [], [], []
        for s in SEEDS:
            cfgs = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s, weight_high=W_HIGH, **NM_KW)
            try: r=ie_strategy_budget(graph,cfgs,B=B,c=C); ie_v.append(r.get("total_revenue",r.get("revenue",{})).get("mean",0.))
            except: ie_v.append(0.)
            try: r=greedy_discount_budget(graph,cfgs,B=B,c=C); gd_v.append(r.get("total_revenue",r.get("revenue",{})).get("mean",0.))
            except: gd_v.append(0.)
        try:
            rv2=dp_calibrated_v2_obs_budget(graph,cfg0,B=B,c=C,n_trials=10)
            v2=rv2.get("revenue",{}).get("all",[rv2.get("revenue",{}).get("mean",0.)]*10)
        except: v2=[0.]*10
        try:
            rv3=dp_calibrated_v3_obs_budget(graph,cfg0,B=B,c=C,n_trials=10)
            v3=rv3.get("revenue",{}).get("all",[rv3.get("revenue",{}).get("mean",0.)]*10)
        except: v3=[0.]*10
        cdp_v=[max(a,b) for a,b in zip(v2,v3)]
        gnn_v=[_run_policy(pol,graph,cache,ei,k,s,device) for s in SEEDS]
        out[str(k)]={"IE+Budget":_stats(ie_v),"Greedy+Budget":_stats(gd_v),"Cal-DP":_stats(cdp_v),"Rev-GNN-LSTM":_stats(gnn_v)}
        print(f"  {net} k={k:2d}  IE={np.mean(ie_v):.1f}  GD={np.mean(gd_v):.1f}  CDP={np.mean(cdp_v):.1f}  GNN={np.mean(gnn_v):.1f}")
    return out


def main():
    if os.path.exists(OUT):
        print(f"Output already exists: {OUT} — skipping"); return
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pol    = _load_policy(device)
    print("G3 v2: NON-MONOTONE budget, FF_1000+Rice_FB, k=[5,10,20,40], 10 seeds")
    results = {"shas":{"policy":ARM_SHA},"acceptance_mode":"nonmonotone"}
    for net, loader in NETS.items():
        graph = loader()
        print(f"\n=== {net} ===")
        results[net] = run_net(net, graph, pol, device)
    os.makedirs("results/logs", exist_ok=True)
    with open(OUT,"w") as f: json.dump(results,f,indent=2)
    print(f"\nSaved → {OUT}")

if __name__ == "__main__":
    main()
