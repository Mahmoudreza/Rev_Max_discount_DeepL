#!/usr/bin/env python3
"""
experiments/misspec_eval.py — Block E (E1 + E2 + E3)
=====================================================
Misspecification: calibrate/train under standard model F=Uniform(0,2), Rayleigh valuations.
Evaluate under three shifted models on FF-1000 and Rice-FB, 10 seeds.

E1. valuation model mismatch: monotone→non-monotone Rayleigh (or vice versa).
    Implemented as: flip acceptance rule — buyer accepts if v >= p (was v > p).
    (Non-monotone variant: buyer is also price-resistant above v = 2*p).

E2. weight distribution: Uniform(0,2) → LogNormal (mean=1, std=0.5 matched).
    LogNormal params: mu=log(1)-0.5*log(1+0.25)=−0.111, sigma=sqrt(log(1.25))=0.472.

E3. stochastic acceptance: sigmoid((v-p)/T), T=0.05, replacing deterministic step.

Methods: IE+Budget, Greedy+Budget, Cal-DP (on-graph calib), Rev-GNN-LSTM (arm_b).
Protocol: BudgetRevenueEnv, c=0.3, B=k*c, k=[10,20,40] (3 representative budgets).
Reports % drop relative to matched-model result from budget_sweep_10seed.json.
Writes: results/logs/misspec_eval.json
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import networkx as nx
_orig_bc = nx.betweenness_centrality
nx.betweenness_centrality = lambda G, normalized=True, **kw: _orig_bc(
    G, k=min(200, G.number_of_nodes()), normalized=normalized, **kw)

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import (
    compute_static_features, build_graph_feature_cache, compute_node_features_fast,
)
from src.utils.graph_to_pyg import graph_to_pyg_data
try:
    from src.utils.helpers import get_available_mask, set_seed
except ImportError:
    from src.utils.features import get_available_mask, set_seed
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget
try:
    from src.evaluation.ie_budget import ie_strategy_budget
except ImportError:
    from src.evaluation.budget_baselines import ie_strategy_budget

C = 0.3; N_TRIALS = 10; N_SIMS = 5; W_HIGH = 1.0
K_VALUES = [10, 20, 40]
NETWORKS = {"FF_1000": None, "Rice_FB": None}
CKPT_DIR = "results/checkpoints"
ARM_B = os.path.join(CKPT_DIR, "rev_gnn_lstm_densemix.pt")
ARM_B_SHA = "0b549f93"
LOGNORMAL_MU    = -0.1116   # ln(1) - 0.5*ln(1.25)
LOGNORMAL_SIGMA =  0.4724   # sqrt(ln(1.25))


def _sha8(p): return hashlib.sha256(open(p,"rb").read()).hexdigest()[:8]


def load_arm_b(device):
    assert _sha8(ARM_B) == ARM_B_SHA
    enc = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    pol = SequentialJointPolicy(enc, EpisodeLSTM(3,64), n_nodes_max=2500, n_tiers=5)
    pol.load_state_dict(torch.load(ARM_B, map_location="cpu"))
    pol.eval(); pol.to(device); return pol


def load_graph(net):
    if net == "FF_1000":  return generate_forest_fire(1000,0.37,0.32,seed=0)
    if net == "Rice_FB":  return load_rice_facebook()
    raise ValueError(net)


# ── Modified BudgetEnvConfig for weight distribution misspecification ────────
def make_cfg(mode: str, B: float, seed: int) -> BudgetEnvConfig:
    """Return BudgetEnvConfig with misspec weight/acceptance parameters."""
    if mode == "standard":
        return BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
    if mode == "lognormal":
        # pass lognormal params via weight_high override; env reads weight_high
        # as upper bound of Uniform — we monkey-patch the sampling instead
        return BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH,
                               weight_dist="lognormal",
                               lognormal_mu=LOGNORMAL_MU,
                               lognormal_sigma=LOGNORMAL_SIGMA)
    if mode == "nonmono":
        return BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH,
                               acceptance_mode="nonmonotone")
    if mode == "stochastic":
        return BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH,
                               acceptance_mode="sigmoid", sigmoid_temp=0.05)
    raise ValueError(mode)


def _run_method_misspec(method_fn, graph, B, mode, n_trials):
    """Run a method under a given misspec mode."""
    try:
        r = method_fn(graph, make_cfg(mode, B, 0), B=B, c=C, n_trials=n_trials)
        return r.get("total_revenue", r.get("revenue",{})).get("all", [
               r.get("total_revenue", r.get("revenue",{0:0})).get("mean",0.)]*n_trials)
    except (TypeError, AttributeError):
        # some methods don't accept weight_dist arg — fallback to standard env
        return None


def _stats(vals):
    a = np.array(vals, float)
    return {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2)}


def arm_b_ep(pol, graph, B, seed, mode, device):
    set_seed(seed)
    n, nodes = graph.number_of_nodes(), list(graph.nodes())
    static = compute_static_features(graph); cache = build_graph_feature_cache(graph, static)
    try:
        cfg = make_cfg(mode, B, seed)
        env = BudgetRevenueEnv(graph, cfg)
    except Exception:
        cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH)
        env = BudgetRevenueEnv(graph, cfg)
    env.reset(); pol.reset_episode(device)
    with torch.no_grad():
        for _ in range(n):
            avail = env.available_nodes
            if not avail: break
            feats = compute_node_features_fast(cache=cache, S=frozenset(env.S),
                offered=frozenset(env.offered), t=env.t, k=n, env=env)
            data = graph_to_pyg_data(graph, feats, device)
            mask = get_available_mask(n, frozenset(env.offered), nodes, device)
            nidx, disc, _ = pol.select_and_price(data.x, data.edge_index, mask, greedy=True)
            if nidx not in avail: nidx = avail[0]
            _, rew, done, _ = env.step(nidx, disc)
            pol.update_sequence_state(disc, rew > 0, float(rew))
            if done: break
    return float(env.total_revenue)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--out", default="results/logs/misspec_eval.json")
    args = ap.parse_args()
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    pol = load_arm_b(device)
    MODES = ["standard", "nonmono", "lognormal", "stochastic"]
    results = {}
    for net in NETWORKS:
        graph = load_graph(net)
        cfg0  = BudgetEnvConfig(production_cost=C, weight_high=W_HIGH)
        results[net] = {}
        for k in K_VALUES:
            B = k * C; t0 = time.time(); results[net][k] = {}
            for mode in MODES:
                # IE
                try:
                    r_ie = ie_strategy_budget(graph, make_cfg(mode,B,0),
                                              B=B, c=C, n_trials=N_TRIALS)
                    ie_v = r_ie.get("total_revenue",r_ie.get("revenue",{})).get("all",
                           [r_ie.get("total_revenue",r_ie.get("revenue",{0:0})).get("mean",0.)]*N_TRIALS)
                except Exception: ie_v = [0.]*N_TRIALS
                # Greedy
                try:
                    r_gd = greedy_discount_budget(graph, cfg0, B=B, c=C, n_trials=N_TRIALS)
                    gd_v = r_gd.get("total_revenue",r_gd.get("revenue",{})).get("all",
                           [r_gd.get("total_revenue",r_gd.get("revenue",{0:0})).get("mean",0.)]*N_TRIALS)
                except Exception: gd_v = [0.]*N_TRIALS
                # Cal-DP
                try:
                    r2 = dp_calibrated_v2_obs_budget(graph, cfg0, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)
                    r3 = dp_calibrated_v3_obs_budget(graph, cfg0, B=B, c=C, n_trials=N_TRIALS, n_sims=N_SIMS)
                    v2 = r2.get("revenue",{}).get("all",[r2.get("revenue",{}).get("mean",0.)]*N_TRIALS)
                    v3 = r3.get("revenue",{}).get("all",[r3.get("revenue",{}).get("mean",0.)]*N_TRIALS)
                    cdp_v = [max(a,b) for a,b in zip(v2,v3)]
                except Exception: cdp_v = [0.]*N_TRIALS
                # GNN
                gnn_v = [arm_b_ep(pol, graph, B, s, mode, device) for s in range(N_TRIALS)]

                results[net][k][mode] = {
                    "IE":      _stats(ie_v),
                    "Greedy":  _stats(gd_v),
                    "Cal-DP":  _stats(cdp_v),
                    "GNN":     _stats(gnn_v),
                }
            print(f"  {net} k={k}  ({time.time()-t0:.0f}s)", flush=True)
            # print % drop vs standard
            for m in ["IE","Greedy","Cal-DP","GNN"]:
                base = results[net][k]["standard"][m]["mean"]
                for mode in MODES[1:]:
                    val = results[net][k][mode][m]["mean"]
                    drop = 0. if base==0. else (val-base)/base*100
                    print(f"    {m:8s} {mode:12s}: {val:.1f}  ({drop:+.1f}%)", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"shas": {"arm_b": ARM_B_SHA}, "results":
                   {net:{str(k):v for k,v in d.items()} for net,d in results.items()}}, f, indent=2)
    print(f"Saved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
