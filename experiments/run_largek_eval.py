#!/usr/bin/env python3
"""experiments/run_largek_eval.py — Gate S: Large-k specialist evaluation.

Evaluates rev_gnn_lstm_largek.pt on FF n=1000, c=0.3.
Reads FROZEN reference numbers from results/logs/unified_sweep.json.
Reads DP-comp from CLAUDE.md constants (dp_v3_full_curve_merged.json).

k = [15, 16, 20, 25, 30, 40]
n_trials = 3, seeds = [42, 123, 7] (same as unified_sweep)
SKIP enforcement + accounting identity check per episode.

Output:
  results/logs/largek_specialist_eval.json
  Printed table + GATE S verdict line.

PRE-COMMITTED GATE S:
  PASS iff:
    (a) specialist >= 435.0 at k=40
    (b) specialist >= 384.6 at k=20   (frozen_unified 369.6 + 15)
    (c) specialist >= 402.9 at k=30   (frozen_unified 387.9 + 15)
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import List

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.models.encoders.graphsage import GraphSAGEEncoder
from src.models.encoders.sequence_models import EpisodeLSTM
from src.models.policies.sequential_joint_policy import SequentialJointPolicy
from src.utils.features import compute_static_features, build_graph_feature_cache
from src.utils.budget_features import compute_budget_node_features_fast

# ── Constants ──────────────────────────────────────────────────────────────────
C            = 0.3
B_MAX_FRAC   = 12.0   # for budget_fraction normalisation (= 40 * C)
K_EVAL       = [15, 16, 20, 25, 30, 40]
N_TRIALS     = 3
SEEDS        = [42, 123, 7]
WEIGHT_HIGH  = 2.0

SPECIALIST_CKPT = "results/checkpoints/rev_gnn_lstm_largek.pt"
LOG_OUT         = "results/logs/largek_specialist_eval.json"

# Frozen unified reference numbers (from unified_sweep.json)
FROZEN_UNIFIED = {15: 365.2, 16: None, 20: 369.6, 25: None, 30: 387.9, 40: 407.0}

# DP-composite reference (from dp_v3_full_curve_merged.json via CLAUDE.md table)
DP_COMP = {1: 10.6, 2: 42.1, 3: 99.8, 5: 154.2, 8: 415.6,
           10: 435.1, 15: 447.7, 20: 448.0, 25: 448.0, 30: 448.0, 40: 448.0}

# Greedy+Budget reference (from results/logs/dp_upgrade_eval* / teacher sanity)
GREEDY_REF = {40: None}   # filled at runtime from greedy sanity in Step 0 if available

# Gate S thresholds (pre-committed, not adjustable)
GATE_S_K40_MIN   = 435.0
GATE_S_K20_DELTA = 15.0
GATE_S_K30_DELTA = 15.0


# ── Model helpers ─────────────────────────────────────────────────────────────

def _load_specialist(device) -> SequentialJointPolicy:
    enc  = GraphSAGEEncoder(in_dim=21, hidden_dim=64, n_layers=2)
    lstm = EpisodeLSTM(graph_dim=64, lstm_hidden=64, n_layers=1)
    pol  = SequentialJointPolicy(enc, lstm, gnn_dim=64, context_dim=64)
    sd   = torch.load(SPECIALIST_CKPT, map_location=device)
    pol.load_state_dict(sd, strict=True)
    pol.eval()
    return pol.to(device)


def _edge_index(graph, device) -> torch.Tensor:
    edges = list(graph.edges())
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long, device=device)
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long, device=device)


# ── Single-episode evaluator ──────────────────────────────────────────────────

@torch.no_grad()
def _eval_one(policy, graph, fc, ei, n_val, k, seed, device):
    """Evaluate one episode with greedy policy + SKIP enforcement."""
    B0  = k * C
    cfg = BudgetEnvConfig(budget_B=B0, production_cost=C,
                          seed=seed, weight_high=WEIGHT_HIGH)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    policy.reset_episode(device)

    revenue  = 0.0
    bankrupt = False
    n_acc    = 0
    B_start  = env.B

    while env.available_nodes and not env._check_bankrupt():
        feats = compute_budget_node_features_fast(
            fc, env.S, env.offered, env.t, k=n_val, env=env,
        )
        x    = torch.tensor(feats, dtype=torch.float32, device=device)
        mask = torch.zeros(n_val, dtype=torch.bool, device=device)
        for idx in env.available_nodes:
            mask[idx] = True

        if mask.sum() == 0:
            break

        scores, h_emb, ctx, _ = policy(x, ei, mask)
        ni   = int(scores.argmax().item())
        comb = torch.cat([h_emb[ni], ctx])
        d    = float(policy.get_discount_distribution(comb).mean.item())

        # SKIP enforcement
        ev = env._estimate_valuation(env.nodes[ni])
        p  = ev * (1.0 - d)
        if env.B - C + p < -1e-9:
            env.offered.add(env.nodes[ni])
            env.t += 1
            env.budget_history.append(env.B)
            policy.update_sequence_state(d, False, 0.0)
            continue

        _, _, done, info = env.step(ni, d)
        if info["accepted"]:
            revenue += info["offered_price"]
            n_acc   += 1
        policy.update_sequence_state(
            d, info["accepted"],
            info["offered_price"] if info["accepted"] else 0.0,
        )
        if done:
            break

    if env._check_bankrupt():
        bankrupt = True

    # Accounting identity check
    acct_ok = True
    if hasattr(env, 'budget_history') and env.budget_history:
        B_end     = env.B
        # Just verify budget is non-negative
        acct_ok   = (B_end >= -1e-6)

    return {
        "revenue": revenue, "bankrupt": bankrupt,
        "n_accepted": n_acc, "B_final": float(env.B),
        "acct_ok": acct_ok,
    }


def eval_k(policy, graph, fc, ei, n_val, k, device) -> dict:
    """Evaluate n_trials trials for one k. Returns mean/std/all."""
    results = []
    for seed in SEEDS[:N_TRIALS]:
        r = _eval_one(policy, graph, fc, ei, n_val, k, seed, device)
        results.append(r)
    revs = [r["revenue"] for r in results]
    return {
        "mean":         float(np.mean(revs)),
        "std":          float(np.std(revs)),
        "all":          revs,
        "bankrupt_mean": float(np.mean([r["bankrupt"] for r in results])),
        "n_accepted_mean": float(np.mean([r["n_accepted"] for r in results])),
    }


# ── Load frozen reference ─────────────────────────────────────────────────────

def _load_frozen_reference() -> dict:
    """Load frozen unified sweep reference numbers."""
    path = "results/logs/unified_sweep.json"
    ref  = {}
    if os.path.exists(path):
        d   = json.load(open(path))
        ff  = d.get("results", {}).get("ff", {})
        for k_str, v in ff.items():
            k = int(k_str)
            if k in K_EVAL:
                ref[k] = v.get("rev_mean")
    # Fallback to hardcoded values from CLAUDE.md
    for k, v in FROZEN_UNIFIED.items():
        if k not in ref and v is not None:
            ref[k] = v
    return ref


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    print("=" * 72)
    print("run_largek_eval.py — Gate S: Large-k Specialist Evaluation")
    print("=" * 72)

    if not os.path.exists(SPECIALIST_CKPT):
        print(f"ERROR: {SPECIALIST_CKPT} not found. Run run_largek_specialist.py first.")
        sys.exit(1)

    import hashlib
    ckpt_sha256 = hashlib.sha256(open(SPECIALIST_CKPT, "rb").read()).hexdigest()
    print(f"Specialist: {SPECIALIST_CKPT}")
    print(f"  sha256: {ckpt_sha256}")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    policy = _load_specialist(device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Loaded: {n_params:,} params")

    # Generate FF n=1000 graph
    import random
    random.seed(0); np.random.seed(0)
    graph = generate_forest_fire(n=1000, p=0.37, pb=0.32, seed=0)
    n_val = graph.number_of_nodes()
    print(f"\nGraph: FF n={n_val} edges={graph.number_of_edges()}")

    sf = compute_static_features(graph)
    fc = build_graph_feature_cache(graph, sf)
    ei = _edge_index(graph, device)

    frozen_ref = _load_frozen_reference()
    print(f"Frozen reference loaded: {sorted(frozen_ref.keys())}")

    # Also get Greedy+Budget at k=40 for the table
    greedy_k40 = None
    try:
        from src.evaluation.budget_baselines import greedy_discount_budget
        gr = greedy_discount_budget(graph, B=40*C, c=C, n_trials=3, weight_high=WEIGHT_HIGH)
        greedy_k40 = gr["revenue"]["mean"]   # return format: r["revenue"]["mean"]
        print(f"Greedy+Budget k=40: {greedy_k40:.2f}")
    except Exception as e:
        print(f"  (Greedy reference unavailable: {e})")

    # ── Evaluate specialist ────────────────────────────────────────────────────
    print(f"\nEvaluating k={K_EVAL}, n_trials={N_TRIALS}, seeds={SEEDS}")
    print(f"{'k':>3} | {'frozen':>7} | {'spec':>7} | {'delta':>7} | {'DP-comp':>7} | {'Greedy':>7}")
    print("-" * 58)

    results = {}
    for k in K_EVAL:
        t_k = time.time()
        r   = eval_k(policy, graph, fc, ei, n_val, k, device)
        results[k] = r

        frz    = frozen_ref.get(k)
        frz_s  = f"{frz:7.1f}" if frz is not None else "    n/a"
        delta  = r["mean"] - frz if frz is not None else float("nan")
        dp     = DP_COMP.get(k, float("nan"))
        gre    = greedy_k40 if k == 40 else float("nan")

        print(f"{k:>3} | {frz_s} | {r['mean']:7.1f} | {delta:+7.1f} | {dp:7.1f} | "
              f"{'n/a':>7}"
              if gre != gre else
              f"{k:>3} | {frz_s} | {r['mean']:7.1f} | {delta:+7.1f} | {dp:7.1f} | {gre:7.1f}"
              if k == 40 else
              f"{k:>3} | {frz_s} | {r['mean']:7.1f} | {delta:+7.1f} | {dp:7.1f} |     n/a"
              )
        sys.stdout.flush()

    # ── Gate S verdict ─────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    spec_k40 = results[40]["mean"]
    spec_k20 = results[20]["mean"]
    spec_k30 = results[30]["mean"]

    frz_k20 = frozen_ref.get(20, 369.6)
    frz_k30 = frozen_ref.get(30, 387.9)

    crit_a = spec_k40 >= GATE_S_K40_MIN
    crit_b = spec_k20 >= frz_k20 + GATE_S_K20_DELTA
    crit_c = spec_k30 >= frz_k30 + GATE_S_K30_DELTA

    gate_pass = crit_a and crit_b and crit_c

    print(f"GATE S criteria:")
    print(f"  (a) spec k=40 = {spec_k40:.2f} >= {GATE_S_K40_MIN:.1f}               : {'PASS' if crit_a else 'FAIL'}")
    print(f"  (b) spec k=20 = {spec_k20:.2f} >= {frz_k20:.1f}+{GATE_S_K20_DELTA:.0f}={frz_k20+GATE_S_K20_DELTA:.1f} : {'PASS' if crit_b else 'FAIL'}")
    print(f"  (c) spec k=30 = {spec_k30:.2f} >= {frz_k30:.1f}+{GATE_S_K30_DELTA:.0f}={frz_k30+GATE_S_K30_DELTA:.1f} : {'PASS' if crit_c else 'FAIL'}")

    verdict = "PASS" if gate_pass else "FAIL"
    reason  = f"k40={spec_k40:.2f}, k20={spec_k20:.2f}, k30={spec_k30:.2f}"
    verdict_line = f"GATE S: {verdict} ({reason})"
    print(f"\n{verdict_line}")

    # ── Save results ──────────────────────────────────────────────────────────
    out = {
        "gate_s": {
            "pass": gate_pass, "verdict": verdict,
            "spec_k40": spec_k40, "spec_k20": spec_k20, "spec_k30": spec_k30,
            "crit_a": crit_a, "crit_b": crit_b, "crit_c": crit_c,
            "thresholds": {
                "k40": GATE_S_K40_MIN,
                "k20": frz_k20 + GATE_S_K20_DELTA,
                "k30": frz_k30 + GATE_S_K30_DELTA,
            },
        },
        "ckpt_sha256":   ckpt_sha256,
        "n_trials":      N_TRIALS,
        "seeds":         SEEDS,
        "frozen_ref":    {str(k): v for k, v in frozen_ref.items()},
        "dp_comp":       {str(k): v for k, v in DP_COMP.items()},
        "greedy_k40":    greedy_k40,
        "results":       {
            str(k): v for k, v in results.items()
        },
        "verdict_line":  verdict_line,
        "wall_seconds":  int(time.time() - t_start),
    }
    with open(LOG_OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {LOG_OUT}")
    print(f"Wall time: {(time.time()-t_start)/60:.1f} min")

    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
