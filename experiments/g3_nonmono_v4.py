#!/usr/bin/env python3
"""
experiments/g3_nonmono_v4.py — G3 v4 (standalone, no NM env class)
====================================================================
Non-monotone Rayleigh/Gaussian acceptance, budget protocol.
FULLY STANDALONE: no BudgetRevenueEnvNM dependency.
Directly uses BudgetRevenueEnv.step() but overrides acceptance in a wrapper.

P(accept | price, w_true) = exp(-(price - w_true)^2 / (2*(w_true/2)^2))
=> peaks at price=w_true, falls for both lower AND higher prices.

FF_1000 + Rice_FB, k=[5,10,20,40], 10 seeds, 3 methods.
Writes: results/logs/g3_nonmono_v4.json
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook

C = 0.3; W_HIGH = 1.0
SEEDS = list(range(10))
KS    = [5, 10, 20, 40]
NETS  = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
         "Rice_FB": load_rice_facebook}
OUT   = "results/logs/g3_nonmono_v4.json"


def nm_accept_prob(price: float, w: float) -> float:
    """P(accept) = exp(-(price-w)^2 / (2*(w/2)^2)), peaked at price=w."""
    if w <= 0: return 0.5
    sigma = w / 2.0
    return math.exp(-((price - w) ** 2) / (2.0 * sigma * sigma))


def run_nm_episode(graph, cfg: BudgetEnvConfig, disc: float, rng) -> float:
    """Run one NM episode: use env for state/budget tracking only;
    override acceptance decision with the NM rule (ignore env's true acceptance).
    Reward = price when NM-accepted AND affordable.
    """
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    total = 0.0

    for node in ordering:
        if len(env.offered) >= env.n: break
        if node in env.offered: continue

        ni = env.node_to_idx.get(node)
        if ni is None: continue

        est_val = env._estimate_valuation(node)
        price   = float(est_val) * (1.0 - disc)

        # Budget check (same as env.step)
        if env.B - C + price < -1e-9:
            env.offered.add(node); env.t += 1
            continue

        # True valuation for NM acceptance
        w_true = float(env._true_valuation(node))
        p_acc  = nm_accept_prob(price, w_true)
        accepted = bool(rng.random() < p_acc)

        # Step the env to keep state consistent (use disc=1.0=free to avoid
        # env's own acceptance from interfering; we track reward ourselves)
        # Actually: call env.step with disc so budget updates if env accepts.
        # But env uses MONOTONE acceptance internally.
        # Workaround: call step, then OVERRIDE budget + reward if our NM decision differs.
        obs, env_reward, done, info = env.step(ni, disc)

        # Override: use our NM accepted decision
        if accepted and not info.get("accepted", False):
            # NM says accept but env said reject: manually update budget + revenue
            env.B = env.B - C + price
            total += price
        elif not accepted and info.get("accepted", False):
            # NM says reject but env accepted: undo budget update
            env.B = env.B + C - price   # reverse env's budget change
            # no revenue
        elif accepted and info.get("accepted", False):
            # Both agree: accept; env already updated budget; add price to total
            total += price
        # else: both reject, nothing to do

        if done: break

    return total


def _stats(v):
    a = np.array(v, dtype=float)
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2), "all": list(v)}


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT} — remove to rerun"); return

    print("G3 v4: NM Gaussian acceptance (standalone), FF_1000+Rice_FB, k=[5,10,20,40], 10 seeds")
    results = {
        "acceptance_model": "Gaussian peaked: P(accept)=exp(-(p-w)^2/(2*(w/2)^2))",
        "note": "price=est_val*(1-disc); IE disc=0.5, GD disc=0.2, CDP-NM disc=0 (optimal for NM)"
    }

    for net, loader in NETS.items():
        graph = loader()
        results[net] = {}
        for k in KS:
            B = k * C
            ie_v, gd_v, cdp_v = [], [], []
            for s in SEEDS:
                cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=s, weight_high=W_HIGH)
                rng = np.random.default_rng(s + 77777)
                ie_v.append(run_nm_episode(graph, cfg, disc=0.5, rng=rng))
                rng = np.random.default_rng(s + 88888)
                gd_v.append(run_nm_episode(graph, cfg, disc=0.2, rng=rng))
                rng = np.random.default_rng(s + 99999)
                cdp_v.append(run_nm_episode(graph, cfg, disc=0.0, rng=rng))  # optimal NM disc

            results[net][str(k)] = {
                "IE+Budget":     _stats(ie_v),
                "Greedy+Budget": _stats(gd_v),
                "CDP-NM-optimal":_stats(cdp_v),
            }
            print(f"  {net} k={k:2d}  IE={np.mean(ie_v):.1f}  GD={np.mean(gd_v):.1f}  CDP-NM={np.mean(cdp_v):.1f}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
