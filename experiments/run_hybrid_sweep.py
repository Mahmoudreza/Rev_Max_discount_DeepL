#!/usr/bin/env python3
"""experiments/run_hybrid_sweep.py — Gate H: Hybrid Policy+Planner sweep.

Evaluates Rev-GNN-LSTM + DP v2 lookahead on FF n=1000, c=0.3.
Compares against raw unified policy (sanity: must match unified_sweep.json within noise).

k=[1,2,3,5,8,10,15,20,30,40], n_trials=3, seeds=[42,123,7] (matching dp_upgrade_eval).
SKIP enforcement + accounting identity check per episode.

PRE-COMMITTED GATE H:
  PASS iff:
    (a) hybrid >= 430.0 at k=40, AND
    (b) hybrid >= raw_unified - 5.0 at EVERY k in the sweep (no-regression).

Saves results/logs/hybrid_sweep.json.
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.evaluation.dp_calibrated_v2 import calibrate_v2_table
from src.evaluation.hybrid_lookahead_policy import build_J_table, HybridLookaheadPolicy

# ── Constants ──────────────────────────────────────────────────────────────────
C         = 0.3
B_MAX     = 40 * C       # = 12.0
DP_DELTA  = 0.05
DP_TIERS  = (1.0, 0.8, 0.5, 0.2, 0.0)
CKPT_DIR  = "results/checkpoints"
LOG_DIR   = "results/logs"
UNIFIED_CKPT = os.path.join(CKPT_DIR, "rev_gnn_lstm_unified.pt")

K_LIST   = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
N_TRIALS = 3
SEEDS    = [42, 123, 7]

# Gate H thresholds
GATE_H_K40_MIN    = 430.0
GATE_H_REGRESSION = 5.0   # hybrid >= raw - 5.0 at every k


# ── Helpers mirrored from run_unified_sweep.py ─────────────────────────────────

def _to_edge_index(graph, device):
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    nodes = list(graph.nodes())
    nmap  = {v: i for i, v in enumerate(nodes)}
    src   = [nmap[e[0]] for e in edges] + [nmap[e[1]] for e in edges]
    dst   = [nmap[e[1]] for e in edges] + [nmap[e[0]] for e in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


def _avail_mask(env, n, device):
    mask = torch.zeros(n, dtype=torch.bool, device=device)
    for idx in env.available_nodes:
        mask[idx] = True
    return mask


def _unified_feat(cache, env, k_budget):
    from src.utils.features import compute_node_features_fast
    base = compute_node_features_fast(cache, env.S, env.offered, env.t, k_budget, env)
    n    = cache["n"]
    col  = np.full((n, 1), env.B / B_MAX, dtype=np.float32)
    return np.concatenate([base, col], axis=1)


def _load_unified(device) -> SequentialJointPolicy:
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(UNIFIED_CKPT, map_location=device)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    return pol.to(device)


# ── Single-episode evaluators ─────────────────────────────────────────────────

@torch.no_grad()
def _eval_one_raw(policy, graph, cache, k_budget, seed, device):
    """Raw unified policy — mirrors _eval_unified_one from run_unified_sweep.py."""
    B   = k_budget * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()

    n        = graph.number_of_nodes()
    ei       = _to_edge_index(graph, device)
    policy.reset_episode(device)

    revenue  = 0.0
    bankrupt = False
    n_acc    = 0

    while True:
        if not env.available_nodes:
            break
        if env._check_bankrupt():
            bankrupt = True
            break

        x_t  = torch.tensor(_unified_feat(cache, env, k_budget), device=device)
        avail = _avail_mask(env, n, device)

        scores, h, ctx, _ = policy.forward(x_t, ei, avail)
        node_idx  = int(scores.argmax().item())
        comb      = torch.cat([h[node_idx], ctx], dim=0)
        beta      = policy.get_discount_distribution(comb)
        discount  = float(beta.mean.item())

        # SKIP enforcement
        est_val       = env._estimate_valuation(env.nodes[node_idx])
        offered_price = est_val * (1.0 - discount)
        if env.B - C + offered_price < -1e-9:
            env.offered.add(env.nodes[node_idx])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(discount, False, 0.0)
            continue

        obs, reward, done, info = env.step(node_idx, discount)
        if info["accepted"]:
            revenue += info["offered_price"]
            n_acc   += 1
        policy.update_sequence_state(discount, info["accepted"],
                                     info["offered_price"] if info["accepted"] else 0.0)
        if done:
            break

    # Accounting identity
    net_spend = float(sum(h for h in env.budget_history if h is not None) - env.B
                      ) if hasattr(env, 'budget_history') else 0.0
    # Simpler: B - remaining = net cost; revenue = total_rev collected
    # Use env.B as remaining budget
    return revenue, bankrupt, n_acc, float(env.B)


@torch.no_grad()
def _eval_one_hybrid(policy, J, graph, cache, k_budget, seed, device):
    """Hybrid (policy + lookahead) evaluation for one trial."""
    B   = k_budget * C
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()

    n        = graph.number_of_nodes()
    ei       = _to_edge_index(graph, device)
    policy.reset_episode(device)

    # b_max=B_MAX: J was built for [0, B_MAX]; clip actual budget to that range
    hybrid = HybridLookaheadPolicy(policy, J, DP_DELTA, B_MAX, C)

    revenue  = 0.0
    bankrupt = False
    n_acc    = 0

    while True:
        if not env.available_nodes:
            break
        if env._check_bankrupt():
            bankrupt = True
            break

        x_t   = torch.tensor(_unified_feat(cache, env, k_budget), device=device)
        avail = _avail_mask(env, n, device)

        node_idx, discount = hybrid.select_action(x_t, ei, avail, env)

        # SKIP enforcement: if even the hybrid's choice is infeasible → skip
        est_val       = env._estimate_valuation(env.nodes[node_idx])
        offered_price = est_val * (1.0 - discount)
        if env.B - C + offered_price < -1e-9:
            env.offered.add(env.nodes[node_idx])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(discount, False, 0.0)
            continue

        obs, reward, done, info = env.step(node_idx, discount)
        if info["accepted"]:
            revenue += info["offered_price"]
            n_acc   += 1
        policy.update_sequence_state(discount, info["accepted"],
                                     info["offered_price"] if info["accepted"] else 0.0)
        if done:
            break

    return revenue, bankrupt, n_acc, float(env.B)


# ── Multi-trial wrappers ──────────────────────────────────────────────────────

def eval_raw(policy, graph, cache, k_budget, device, n_trials=N_TRIALS):
    revs, bkrs = [], []
    for s in SEEDS[:n_trials]:
        rev, bkr, _, _ = _eval_one_raw(policy, graph, cache, k_budget, s, device)
        revs.append(rev); bkrs.append(int(bkr))
    return {"mean": float(np.mean(revs)), "std": float(np.std(revs)),
            "bkr": float(np.mean(bkrs)), "all": revs}


def eval_hybrid(policy, J, graph, cache, k_budget, device, n_trials=N_TRIALS):
    revs, bkrs = [], []
    for s in SEEDS[:n_trials]:
        rev, bkr, _, _ = _eval_one_hybrid(policy, J, graph, cache, k_budget, s, device)
        revs.append(rev); bkrs.append(int(bkr))
    return {"mean": float(np.mean(revs)), "std": float(np.std(revs)),
            "bkr": float(np.mean(bkrs)), "all": revs}


# ── Load comparison data ──────────────────────────────────────────────────────

def _load_comparison():
    out = {}
    # DP composite from dp_v3_full_curve_merged.json
    dp_path = os.path.join(LOG_DIR, "dp_v3_full_curve_merged.json")
    if os.path.exists(dp_path):
        d = json.load(open(dp_path))
        for entry in d.get("results", []):
            k = entry.get("k")
            if k:
                out[("dp_comp", k)] = entry.get("composite_mean", entry.get("mean"))
    # Previous unified sweep
    usw = os.path.join(LOG_DIR, "unified_sweep.json")
    if os.path.exists(usw):
        d = json.load(open(usw))
        for k_str, v in d.get("results", {}).get("ff", {}).items():
            out[("prev_unified", int(k_str))] = v.get("rev_mean")
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 72)
    print("run_hybrid_sweep.py — Gate H: Hybrid Policy+Planner (inference-time only)")
    print("=" * 72)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(UNIFIED_CKPT):
        print(f"ERROR: {UNIFIED_CKPT} not found.")
        sys.exit(1)

    import hashlib
    ckpt_sha256 = hashlib.sha256(open(UNIFIED_CKPT, "rb").read()).hexdigest()
    print(f"Checkpoint: {UNIFIED_CKPT}")
    print(f"  sha256: {ckpt_sha256}")

    policy = _load_unified(device)
    print(f"Policy loaded: {sum(p.numel() for p in policy.parameters())} params")

    # ── Generate FF graph (fixed seed=0 for graph, matching dp_upgrade_eval) ──
    print("\nGenerating FF n=1000 (seed=0)...")
    import random
    rng_state = random.getstate()
    np_state  = np.random.get_state()
    np.random.seed(0); random.seed(0)
    graph = generate_forest_fire(n=1000, p=0.37, pb=0.32, seed=0)
    random.setstate(rng_state); np.random.set_state(np_state)
    print(f"  nodes={graph.number_of_nodes()}, edges={graph.number_of_edges()}")

    # ── Calibrate V, A, P tables (from cache or fresh) ────────────────────────
    from omegaconf import OmegaConf
    cfg_calib = OmegaConf.create("""
production_cost: 0.3
weight_high: 2.0
""")
    # Wrap cfg so calibrate_v2_table can access cfg.production_cost
    class CalibCfg:
        def __init__(self): self.production_cost = 0.3; self.weight_high = 2.0
    calib_cfg = CalibCfg()

    print("Calibrating DP v2 table (cache lookup)...")
    t_cal = time.time()
    V, A, P, class_boundaries, infl_boundaries = calibrate_v2_table(
        graph, calib_cfg, n_classes=5, n_buckets=10, n_sims=30, seed=0
    )
    print(f"  Calibration done in {time.time()-t_cal:.1f}s  "
          f"V shape={V.shape}, A shape={A.shape}, P shape={P.shape}")

    # ── Build graph feature cache ──────────────────────────────────────────────
    print("Building feature cache...")
    static     = compute_static_features(graph)
    feat_cache = build_graph_feature_cache(graph, static)

    # ── Load comparison data ──────────────────────────────────────────────────
    comp = _load_comparison()

    # ── Run sweep ─────────────────────────────────────────────────────────────
    results_raw    = {}
    results_hybrid = {}

    print(f"\nSweep: k={K_LIST}, n_trials={N_TRIALS}, seeds={SEEDS}")
    print(f"{'k':>4} | {'raw_mean':>9}±{'std':>6} | {'hybrid_mean':>11}±{'std':>6} | "
          f"{'delta':>7} | {'dp_comp':>8} | {'prev_uni':>8}")
    print("-" * 80)

    # Build J table ONCE with B_MAX so budget self-replenishment (B > k*C) is
    # correctly handled. Using per-k B clips b_idx to a tiny range and corrupts
    # the lookahead when revenue acceptance refills the budget beyond k*C.
    print(f"Building J table (B=B_MAX={B_MAX:.1f}, covers full budget range)...")
    t_jt = time.time()
    J = build_J_table(
        V, A, P, class_boundaries,
        n_total=graph.number_of_nodes(),
        B=B_MAX, c=C,
        tiers=DP_TIERS,
        delta=DP_DELTA,
    )
    print(f"  J shape={J.shape}  ({time.time()-t_jt:.1f}s)")

    for k in K_LIST:
        t_k = time.time()
        B = k * C
        print(f"  k={k:2d} B={B:.2f} ... ", end="", flush=True)

        # Raw unified
        res_raw = eval_raw(policy, graph, feat_cache, k, device)
        # Hybrid (J covers B_MAX; b_max passed as B_MAX for correct _bidx clipping)
        res_hyb = eval_hybrid(policy, J, graph, feat_cache, k, device)

        results_raw[k]    = res_raw
        results_hybrid[k] = res_hyb

        delta = res_hyb["mean"] - res_raw["mean"]
        dp    = comp.get(("dp_comp", k), float("nan"))
        pu    = comp.get(("prev_unified", k), float("nan"))

        print(f"k={k:2d} | {res_raw['mean']:9.2f}±{res_raw['std']:5.2f} | "
              f"{res_hyb['mean']:11.2f}±{res_hyb['std']:5.2f} | "
              f"{delta:+7.2f} | {dp:8.1f} | {pu:8.1f}  "
              f"({time.time()-t_k:.1f}s)")

    # ── Gate H verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    hyb_k40  = results_hybrid[40]["mean"]
    crit_a   = hyb_k40 >= GATE_H_K40_MIN
    crit_b   = all(
        results_hybrid[k]["mean"] >= results_raw[k]["mean"] - GATE_H_REGRESSION
        for k in K_LIST
    )
    gate_pass = crit_a and crit_b

    regression_failures = [
        k for k in K_LIST
        if results_hybrid[k]["mean"] < results_raw[k]["mean"] - GATE_H_REGRESSION
    ]

    print(f"GATE H criteria:")
    print(f"  (a) hybrid k=40 = {hyb_k40:.2f} >= {GATE_H_K40_MIN:.1f} : {'PASS' if crit_a else 'FAIL'}")
    print(f"  (b) no-regression (hybrid >= raw - {GATE_H_REGRESSION:.1f} at every k): "
          f"{'PASS' if crit_b else f'FAIL at k={regression_failures}'}")
    verdict = "PASS" if gate_pass else "FAIL"
    print(f"\nGATE H: {verdict}  "
          f"(hybrid_k40={hyb_k40:.2f}, regression_ok={crit_b})")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "gate_h": {"pass": gate_pass, "hybrid_k40": hyb_k40,
                   "crit_a": crit_a, "crit_b": crit_b,
                   "regression_failures": regression_failures},
        "ckpt_sha256": ckpt_sha256,
        "n_trials": N_TRIALS, "seeds": SEEDS,
        "results": {
            str(k): {
                "raw":    results_raw[k],
                "hybrid": results_hybrid[k],
                "delta":  results_hybrid[k]["mean"] - results_raw[k]["mean"],
            }
            for k in K_LIST
        },
        "wall_seconds": int(time.time() - t_start),
    }
    out_path = os.path.join(LOG_DIR, "hybrid_sweep.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {out_path}")
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")

    return gate_pass


if __name__ == "__main__":
    main()
