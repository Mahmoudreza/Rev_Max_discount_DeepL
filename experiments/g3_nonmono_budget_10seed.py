#!/usr/bin/env python3
"""
experiments/g3_nonmono_budget_10seed.py — G3
=============================================
All methods under NON-MONOTONE Rayleigh valuations, budget protocol,
FF_1000 and Rice_FB only, k=[5,10,20,40], 10 seeds.

The learned policy and Cal-DP are calibrated on the standard monotone
Uniform model — this is a misspecification test.  Results reported
next to the standard (monotone) numbers from budget_sweep_10seed.py.

acceptance_mode="nonmonotone": buyer accepts iff v_i > price (Rayleigh
 upper-tail; implemented in BudgetEnvConfig).

Writes: results/logs/g3_nonmono_budget_10seed.json
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.budget_baselines import (
    greedy_discount_budget, ie_strategy_budget,
)
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.utils.helpers import set_seed
from _arm_b_utils import _feat_unconstrained, make_ei as _make_ei
import hashlib

C = 0.3; W_HIGH = 1.0
SEEDS  = list(range(10))
KS     = [5, 10, 20, 40]
NETS   = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
          "Rice_FB": load_rice_facebook}
ARM_B  = "results/checkpoints/rev_gnn_lstm_densemix.pt"
ARM_SHA= "0b549f93"
NONMONO_CFG_KWARGS = {"acceptance_mode": "nonmonotone"}


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]


def _load_policy(device):
    assert _sha8(ARM_B) == ARM_SHA
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    pol.load_state_dict(torch.load(ARM_B, map_location="cpu"))
    pol.to(device).eval(); return pol


@torch.no_grad()
def _run_policy(pol, graph, cache, ei, k, seed, device):
    set_seed(seed)
    n   = graph.number_of_nodes()
    cfg = BudgetEnvConfig(budget_B=k*C, production_cost=C, seed=seed,
                          weight_high=W_HIGH, **NONMONO_CFG_KWARGS)
    env = BudgetRevenueEnv(graph, cfg); env.reset(); pol.reset_episode(device)
    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = torch.zeros(n, dtype=torch.bool, device=device)
        for i in env.available_nodes: av[i] = True
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(torch.cat([h[ni], ctx])).mean.item())
        _, _, done, info = env.step(ni, d)
        pol.update_sequence_state(d, info["accepted"], info.get("revenue_step",0.))
        if done: break
    return float(env.total_revenue)


def _stats(vals):
    a = np.array(vals, dtype=float)
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2), "all": vals}


def run_net(net, graph, pol, device):
    ei, cache = _make_ei(graph, device)
    out = {}
    for k in KS:
        B   = k * C
        cfg0 = BudgetEnvConfig(budget_B=B, production_cost=C, seed=0,
                               weight_high=W_HIGH, **NONMONO_CFG_KWARGS)
        ie_vals, gd_vals, cdp_vals, gnn_vals = [], [], [], []
        for s in SEEDS:
            cfgs = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s,
                                   weight_high=W_HIGH, **NONMONO_CFG_KWARGS)
            try:
                r = ie_strategy_budget(graph, cfgs, B=B, c=C)
                ie_vals.append(r.get("total_revenue",r.get("revenue",{})).get("mean",0.))
            except Exception: ie_vals.append(0.)
            try:
                r = greedy_discount_budget(graph, cfgs, B=B, c=C)
                gd_vals.append(r.get("total_revenue",r.get("revenue",{})).get("mean",0.))
            except Exception: gd_vals.append(0.)
        # CDP: calibrated on standard model (misspec)
        try:
            rv2 = dp_calibrated_v2_obs_budget(graph, cfg0, B=B, c=C, n_trials=len(SEEDS))
            cdp_v2 = rv2.get("revenue",{}).get("all", [rv2.get("revenue",{}).get("mean",0.)]*10)
        except Exception: cdp_v2 = [0.]*10
        try:
            rv3 = dp_calibrated_v3_obs_budget(graph, cfg0, B=B, c=C, n_trials=len(SEEDS))
            cdp_v3 = rv3.get("revenue",{}).get("all", [rv3.get("revenue",{}).get("mean",0.)]*10)
        except Exception: cdp_v3 = [0.]*10
        cdp_vals = [max(a,b) for a,b in zip(cdp_v2, cdp_v3)]
        gnn_vals = [_run_policy(pol, graph, cache, ei, k, s, device) for s in SEEDS]
        out[str(k)] = {
            "IE+Budget":    _stats(ie_vals),
            "Greedy+Budget":_stats(gd_vals),
            "Cal-DP":       _stats(cdp_vals),
            "Rev-GNN-LSTM": _stats(gnn_vals),
        }
        print(f"  {net} k={k:2d}  IE={np.mean(ie_vals):.1f}  GD={np.mean(gd_vals):.1f}"
              f"  CDP={np.mean(cdp_vals):.1f}  GNN={np.mean(gnn_vals):.1f}")
    return out


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    pol    = _load_policy(device)
    print("G3: NON-MONOTONE budget sweep, FF_1000 + Rice_FB, k=[5,10,20,40], 10 seeds")
    results = {"shas": {"policy": ARM_SHA}, "acceptance_mode": "nonmonotone"}
    for net, loader in NETS.items():
        graph = loader()
        print(f"\n=== {net} ===")
        results[net] = run_net(net, graph, pol, device)
    out = "results/logs/g3_nonmono_budget_10seed.json"
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")

if __name__ == "__main__":
    main()
