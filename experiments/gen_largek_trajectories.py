#!/usr/bin/env python3
"""experiments/gen_largek_trajectories.py — Generate large-k specialist trajectories.

TEACHER EQUIVALENCE VERIFIED 2026-08-08: custom extractor matches frozen
greedy_discount_budget exactly (diff=0.00e+00, 6/6 seed pairs, first-50-step
node/price/budget identical).

PROTOCOL (zero-shot): training graphs must be DISTINCT from the evaluation
graph (FF n=1000, seed=0). Using the same 5 training graphs as the unified
model (n∈{200,260,320,380,440}, p=0.37, pb=0.32, seed=42).
The n=1000 eval-graph cache is quarantined in largek_traj_cache_EVALGRAPH_DO_NOT_TRAIN/.

Teacher: greedy_discount_budget logic (direct env manipulation, exact Babaei tiers).
Graphs:  FF n∈{200,260,320,380,440}, p=0.37, pb=0.32, GRAPH_SEED=42
k:       {16, 20, 25, 30, 40}
Volume:  200 episodes per (graph, k) = 5,000 trajectories total.

Each step dict: node_idx, discount, accepted, price, B_after.
  Free tier  → discount=1.0, price=0.0, accepted=True
  Tier1/Tier2 → discount=0.0, accepted=(true_val >= price)
  SKIP nodes (B-c+price < -1e-9) are NOT recorded.

Full-length episodes — no MAX_TRAJ_STEPS truncation. The harvest phase
(steps 50+, where paid offers dominate) carries the CE signal needed for
large-k imitation. Episode lengths ~100-300 steps on training graphs.

Cache: results/logs/largek_traj_cache/<graphhash>_k<k>_s<seed>.pkl
"""

from __future__ import annotations

import hashlib
import os
import pickle
import random
import sys
import time
from typing import List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.evaluation.baselines import _rayleigh_price

# ── Constants ──────────────────────────────────────────────────────────────────
C             = 0.3
GRAPH_SIZES   = [200, 260, 320, 380, 440]   # TRAINING graphs (NOT eval graph n=1000)
GRAPH_P       = 0.37
GRAPH_PB      = 0.32
GRAPH_SEED    = 42     # matches Stage-B / unified model provenance
N_EPISODES    = 200    # episodes per (graph, k) pair
K_LIST        = [16, 20, 25, 30, 40]
WEIGHT_HIGH   = 2.0
RAYLEIGH_B    = 1.0
CACHE_DIR     = "results/logs/largek_traj_cache"

os.makedirs(CACHE_DIR, exist_ok=True)

TIER1_PRICE = _rayleigh_price(2.0 / 6.0, RAYLEIGH_B)   # ≈ 0.5338
TIER2_PRICE = _rayleigh_price(4.0 / 6.0, RAYLEIGH_B)   # ≈ 0.5481


def _graph_hash(graph) -> str:
    sig = f"{sorted(graph.nodes())}|{sorted(graph.edges())}"
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def _traj_path(gh: str, k: int, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"{gh}_k{k}_s{seed}.pkl")


def _capture(graph, k: int, seed: int) -> List[dict]:
    """Record full-length greedy_discount_budget episode as trajectory.

    Mirrors greedy_discount_budget exactly (verified: diff=0.00e+00).
    SKIP nodes (B-c+price < -1e-9) are excluded from the trajectory.
    """
    B0  = k * C
    env = BudgetRevenueEnv(graph, BudgetEnvConfig(
        budget_B=B0, production_cost=C, seed=seed, weight_high=WEIGHT_HIGH,
    ))
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

    def _inv(node_):
        for nb in graph.neighbors(node_):
            env._influence_cache.pop(nb, None)
            env._true_val_cache.pop(nb, None)
            env._est_val_cache.pop(nb, None)

    traj = []
    for node in ordering:
        if node in env.offered:
            continue
        if env._check_bankrupt():
            break
        infl  = env.get_current_influence(node)
        price = 0.0 if infl < 2/6 else (TIER1_PRICE if infl < 4/6 else TIER2_PRICE)

        if env.B - C + price < -1e-9:   # SKIP
            env.offered.add(node); env.t += 1; env.budget_history.append(env.B)
            continue

        ni = env.node_to_idx[node]
        if price == 0.0:
            env.S.add(node); env.B -= C; _inv(node)
            env.offered.add(node); env.t += 1; env.budget_history.append(env.B)
            traj.append({"node_idx": ni, "discount": 1.0, "accepted": True,
                         "price": 0.0, "B_after": env.B})
        else:
            tv  = env._true_valuation(node)
            acc = (tv >= price)
            if acc:
                env.S.add(node); env.B = env.B - C + price; _inv(node)
            env.offered.add(node); env.t += 1; env.budget_history.append(env.B)
            traj.append({"node_idx": ni, "discount": 0.0, "accepted": acc,
                         "price": price if acc else 0.0, "B_after": env.B})
    return traj


def _acct_ok(traj: List[dict], B0: float) -> bool:
    B = B0
    for s in traj:
        if s["accepted"] and s["price"] == 0.0:
            B_exp = B - C
        elif s["accepted"]:
            B_exp = B - C + s["price"]
        else:
            B_exp = B
        if abs(B_exp - s["B_after"]) > 1e-6:
            return False
        B = s["B_after"]
    return True


def generate_for_graph(graph, gh: str, n: int, k: int) -> dict:
    B0 = k * C
    revenues = []; n_bad = 0
    for seed in range(N_EPISODES):
        path = _traj_path(gh, k, seed)
        if os.path.exists(path):
            traj = pickle.load(open(path, "rb"))
        else:
            traj = _capture(graph, k, seed)
            if not _acct_ok(traj, B0):
                print(f"  ACCOUNTING FAIL n={n} k={k} s={seed}")
                n_bad += 1; continue
            with open(path, "wb") as f:
                pickle.dump(traj, f)
        revenues.append(sum(s["price"] for s in traj if s["accepted"]))
    return {"n": n, "k": k, "n_ok": len(revenues), "n_bad": n_bad,
            "rev_mean": float(np.mean(revenues)) if revenues else 0.0,
            "rev_std":  float(np.std(revenues))  if revenues else 0.0,
            "avg_steps": float(np.mean([len(pickle.load(open(_traj_path(gh,k,s),"rb"))) for s in range(min(5,N_EPISODES))]))}


def main():
    t0 = time.time()
    print("=" * 68)
    print("gen_largek_trajectories.py — training graph trajectories")
    print(f"Graphs: n∈{GRAPH_SIZES} seed={GRAPH_SEED}   k={K_LIST}")
    print(f"Episodes/pair: {N_EPISODES}  Total: {len(GRAPH_SIZES)*len(K_LIST)*N_EPISODES:,}")
    print(f"Tier: tier1={TIER1_PRICE:.4f}  tier2={TIER2_PRICE:.4f}")
    print(f"Protocol: training graphs DISTINCT from eval (n=1000 quarantined)")
    print("=" * 68)

    all_stats = []
    for n in GRAPH_SIZES:
        random.seed(GRAPH_SEED); np.random.seed(GRAPH_SEED)
        g  = generate_forest_fire(n=n, p=GRAPH_P, pb=GRAPH_PB, seed=GRAPH_SEED)
        gh = _graph_hash(g)
        print(f"\nGraph n={n} hash={gh} edges={g.number_of_edges()}")
        for k in K_LIST:
            t_k = time.time()
            print(f"  k={k:2d} B={k*C:.1f} ... ", end="", flush=True)
            s = generate_for_graph(g, gh, n, k)
            all_stats.append(s)
            print(f"rev={s['rev_mean']:.1f}±{s['rev_std']:.1f}  "
                  f"avg_steps={s['avg_steps']:.0f}  "
                  f"({s['n_ok']} ok, {s['n_bad']} bad)  {time.time()-t_k:.1f}s")

    total = sum(s["n_ok"] for s in all_stats)
    print(f"\nTotal cached: {total:,}  wall={time.time()-t0:.1f}s")

    # Timing estimate for Phase 1 training
    avg_steps = np.mean([s["avg_steps"] for s in all_stats])
    ms_per_step = 2.0  # MPS estimate for n=200-440
    epoch_sec = 300 * avg_steps * ms_per_step / 1000.0
    print(f"Phase-1 timing estimate: 300 eps/epoch × {avg_steps:.0f} avg steps × "
          f"{ms_per_step}ms = {epoch_sec:.0f}s/epoch × 150 epochs = "
          f"{epoch_sec*150/3600:.1f}h")
    return 0


if __name__ == "__main__":
    sys.exit(main())
