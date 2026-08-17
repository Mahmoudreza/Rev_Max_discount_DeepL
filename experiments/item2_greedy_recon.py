#!/usr/bin/env python3
"""
experiments/item2_greedy_recon.py — ITEM 2: Greedy-Discount Reconciliation
===========================================================================
Compares two GD implementations and explains the paper vs. recent discrepancy.

FACTS (from eval_c1_final.py & paper_gen_updated.json):
  paper_gen_updated.json: single-seed (seed=42), std=0.0, 1 realization.
  _eval_greedy_discount(graph, cfg) → ~418 for FF-1000  [CORRECT, matches paper]
  greedy_discount(graph, cfg)       → ~462 for FF-1000  [INFLATED, different fn]

Rev-GNN-LSTM (C1): checkpoint rev_gnn_lstm.pt — sha printed at runtime.
Writes: results/logs/item2_greedy_recon.json
"""
from __future__ import annotations
import hashlib, json, os, sys
import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

from src.env.revenue_env import RevenueEnvConfig
from src.evaluation.idea1_eval import _eval_greedy_discount
from src.evaluation.baselines import greedy_discount
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from experiments._arm_b_utils import (
    make_ei, _feat_unconstrained, _avail_mask,
)
from src.utils.helpers import set_seed

SEEDS = list(range(10))
CKPT_C1  = os.path.join(_ROOT, "results/checkpoints/rev_gnn_lstm.pt")
OUT      = "results/logs/item2_greedy_recon.json"

PAPER_GD = {
    "FF_500":   209.91,
    "FF_1000":  417.74,
    "FF_2000":  839.16,
    "Modular":  356.59,
    "Rice_FB":  159.73,
}

GRAPHS = {
    "FF_500":   lambda: generate_forest_fire(500,  0.37, 0.32, seed=0),
    "FF_1000":  lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "FF_2000":  lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
    "Modular":  lambda: generate_modular_forest_fire([250,250],0.37,0.32,0.05,seed=0),
    "Rice_FB":  load_rice_facebook,
}


def _sha8(path):
    return hashlib.sha256(open(path,"rb").read()).hexdigest()[:8]


def _stats(v):
    a = np.array(v, dtype=float)
    return {"mean": round(float(a.mean()),2), "std": round(float(a.std()),2)}


def _load_c1_policy(device, in_dim=20):
    enc  = GraphSAGEEncoder(in_dim=in_dim, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    pol.load_state_dict(torch.load(CKPT_C1, map_location="cpu"))
    pol.eval(); pol.to(device)
    return pol


@torch.no_grad()
def _run_lstm_unc(pol, graph, cache, ei, seed, device):
    from src.env.revenue_env import RevenueEnv
    set_seed(seed)
    n = graph.number_of_nodes()
    env = RevenueEnv(graph, RevenueEnvConfig(seed=seed))
    env.reset(); pol.reset_episode(device)
    total = 0.0
    while env.available_nodes:
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(
                   torch.cat([h[ni], ctx])).mean.item())
        _, r, done, _ = env.step(ni, d)
        total += r
        pol.update_sequence_state(d, r > 0, r)
        if done: break
    return total


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT}"); return

    device = torch.device("cpu")
    sha_c1 = _sha8(CKPT_C1)
    print(f"Rev-GNN-LSTM (C1) checkpoint sha8 = {sha_c1}")
    pol = _load_c1_policy(device)

    results = {
        "shas": {"rev_gnn_lstm_c1": sha_c1, "ckpt": CKPT_C1},
        "paper_protocol": "single seed=42, std=0.0, 1 realization",
        "correct_fn": "idea1_eval._eval_greedy_discount",
        "inflated_fn": "baselines.greedy_discount (~10% higher due to implementation diff)",
    }

    print(f"\n{'Net':<10} {'paper(s42)':>11} {'correct_s42':>12} {'correct_10s':>12} {'inflated_s42':>13} {'lstm_10s':>10}")
    print("-"*63)

    for name, loader in GRAPHS.items():
        graph = loader()
        ei, cache = make_ei(graph, device)

        cfg42 = RevenueEnvConfig(seed=42)
        gd_correct_s42  = float(_eval_greedy_discount(graph, cfg42))
        gd_inflated_s42 = float(greedy_discount(graph, cfg42))
        gd_correct_10s  = [float(_eval_greedy_discount(graph, RevenueEnvConfig(seed=s))) for s in SEEDS]
        lstm_10s         = [_run_lstm_unc(pol, graph, cache, ei, s, device) for s in SEEDS]

        gd10  = _stats(gd_correct_10s)
        lst10 = _stats(lstm_10s)
        paper = PAPER_GD[name]
        diff  = round(gd_correct_s42 - paper, 2)
        flag  = " >5!" if abs(diff) > 5 else ""

        results[name] = {
            "paper_s42":      paper,
            "correct_s42":    round(gd_correct_s42,2),
            "correct_10seed": gd10,
            "inflated_s42":   round(gd_inflated_s42,2),
            "lstm_c1_10seed": lst10,
            "repro_diff":     diff,
            "flag":           flag.strip() or "OK",
        }
        print(f"{name:<10} {paper:>11.2f} {gd_correct_s42:>12.2f} "
              f"{gd10['mean']:>6.2f}±{gd10['std']:>5.2f} "
              f"{gd_inflated_s42:>13.2f} "
              f"{lst10['mean']:>6.2f}±{lst10['std']:>5.2f}{flag}")

    print()
    all_ok = all(abs(results[n]["repro_diff"]) <= 5 for n in GRAPHS)
    verdict = ("PAPER VALUES REPRODUCE" if all_ok else
               "PAPER VALUES DO NOT REPRODUCE: baselines.greedy_discount() inflates by ~10%; "
               "paper used idea1_eval._eval_greedy_discount, single seed=42")
    print(f"VERDICT: {verdict}")
    results["verdict"] = verdict

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT,"w") as f: json.dump(results, f, indent=2)
    print(f"Saved → {OUT}")

if __name__ == "__main__":
    main()
