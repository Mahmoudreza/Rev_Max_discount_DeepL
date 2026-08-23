#!/usr/bin/env python3
"""diag_budget_binding.py — Check if budget actually binds in faithful Greedy.

Run on server (no GPU needed):
  venv/bin/python3 -u experiments/diag_budget_binding.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.greedy_budget_faithful import (
    greedy_discount_budget_faithful, _make_env, _tier_price,
    _compute_normalized_infl, _inv_budget, TIER1_PRICE, TIER2_PRICE)

C = 0.3; W_HIGH = 2.0; N_MC = 200
G = generate_forest_fire(1000, 0.37, 0.32, seed=0)

print("=== Budget env.B diagnostics ===")

# First: confirm env.B is set correctly at different kappas
for kappa in [5, 10, 20, 40]:
    B0 = kappa * C
    env = _make_env(G, B0, C, seed=0, weight_high=W_HIGH, n_mc=N_MC)
    env.reset()
    print(f"kappa={kappa:2d}  B0_requested={B0:.2f}  env.B_after_reset={env.B:.4f}")

print()
print("=== Trajectory trace for seed=0 at each kappa ===")
print(f"  B0  accepted_so_far  env.B  infl  price  skip?")

for kappa in [5, 10, 20, 40]:
    B0 = kappa * C
    env = _make_env(G, B0, C, seed=0, weight_high=W_HIGH, n_mc=N_MC)
    env.reset()
    lw = env._link_weights
    ev_cache = {v: env._estimate_valuation(v) for v in env.nodes}
    revenue = 0.0; n_acc = 0; n_skip = 0
    min_B = env.B; B_final = env.B
    step = 0

    print(f"\n--- kappa={kappa}  B0={B0:.2f}  env.B_init={env.B:.4f} ---")

    while True:
        remaining = [v for v in env.nodes if v not in env.offered]
        if not remaining:
            break
        target = max(remaining, key=lambda v: ev_cache[v])
        infl = _compute_normalized_infl(G, target, env.S, lw)
        price = _tier_price(infl)
        feasible = env.B - C + price >= -1e-9
        skip = not feasible

        if step < 5:
            print(f"  step={step}  B={env.B:.4f}  infl={infl:.4f}  price={price:.4f}"
                  f"  skip={skip}  feasible_check={env.B-C+price:.4f}")

        if skip:
            env.offered.add(target); env.t += 1; n_skip += 1
            continue

        true_val = env._true_valuation(target)
        env.offered.add(target); env.t += 1

        if price == 0.0:
            env.S.add(target); env.B -= C
            _inv_budget(env, target)
            for nb in G.neighbors(target):
                ev_cache[nb] = env._estimate_valuation(nb)
        elif true_val >= price:
            env.S.add(target); env.B = env.B - C + price
            _inv_budget(env, target)
            for nb in G.neighbors(target):
                ev_cache[nb] = env._estimate_valuation(nb)
            revenue += price; n_acc += 1

        min_B = min(min_B, env.B)
        step += 1

    B_final = env.B
    S_T = len(env.S)
    print(f"  SUMMARY: B0={B0:.2f}  min_B={min_B:.4f}  final_B={B_final:.4f}"
          f"  |S_T|={S_T}  revenue={revenue:.3f}  skips={n_skip}  paid={n_acc}")
