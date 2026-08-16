#!/usr/bin/env python3
"""
experiments/g3_nm_ext.py — G3 extension
========================================
Adds arm_b (0b549f93) to the G3 NM table, and reports Cal-DP tier distribution.

ACCEPTANCE RULE (plainly stated):
  Probabilistic. Line from g3_nonmono_v4.py:
      accepted = bool(rng.random() < p_acc)
  where p_acc = exp(-(price - w_true)^2 / (2*(w_true/2)^2)).
  This is NOT deterministic threshold (w >= p).
  For arm_b, the same NM acceptance is applied AFTER the policy chooses its discount.

CAL-DP TIER DISTRIBUTION:
  In g3_nonmono_v4.py, CDP-NM uses disc=0.0 at EVERY step (hardcoded).
  The DP table has NOT been re-calibrated for NM acceptance.
  disc=0.0 was chosen analytically: it maximises P(accept) for the Gaussian
  peaked acceptance (P(accept)=1 at price=est_val, i.e. disc=0 → full price).
  Result: Cal-DP under NM is a FIXED RULE (100% disc=0.0).

Writes: results/logs/g3_nm_ext.json
"""
from __future__ import annotations
import json, math, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from experiments._arm_b_utils import (
    load_arm_b, make_ei, _feat_unconstrained, _avail_mask, C, W_HIGH, N_MC
)
from src.utils.helpers import set_seed

SEEDS = list(range(10))
KS    = [5, 10, 20, 40]
NETS  = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
         "Rice_FB": load_rice_facebook}
TIERS = (1.0, 0.8, 0.5, 0.2, 0.0)   # standard tiers
OUT   = "results/logs/g3_nm_ext.json"


def nm_accept_prob(price: float, w: float) -> float:
    if w <= 0: return 0.5
    s = w / 2.0
    return math.exp(-((price - w) ** 2) / (2.0 * s * s))


@torch.no_grad()
def eval_arm_b_nm_episode(pol, graph, cache, ei, B: float, seed: int, device) -> float:
    """arm_b episode with NM acceptance override.
    Policy chooses node + disc; acceptance uses NM probabilistic rule.
    """
    set_seed(seed)
    n = graph.number_of_nodes()
    cfg = BudgetEnvConfig(budget_B=B, production_cost=C, seed=seed,
                          weight_high=W_HIGH, n_mc_samples=N_MC)
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    pol.reset_episode(device)
    rng = np.random.default_rng(seed + 55555)
    total = 0.0

    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(
                   torch.cat([h[ni], ctx])).mean.item())

        # NM acceptance override
        node = env.nodes[ni]
        est_val = float(env._estimate_valuation(node))
        price   = est_val * (1.0 - d)
        w_true  = float(env._true_valuation(node))
        p_acc   = nm_accept_prob(price, w_true)
        # Probabilistic NM acceptance — same rule as g3_nonmono_v4.py:
        #   accepted = bool(rng.random() < p_acc)
        accepted_nm = bool(rng.random() < p_acc)

        # Step env to update state (env uses monotone acceptance internally)
        _, env_reward, done, info = env.step(ni, d)

        # Override budget and revenue based on NM decision
        env_accepted = info.get("accepted", False)
        if accepted_nm and not env_accepted:
            env.B = env.B - C + price
            total += price
        elif not accepted_nm and env_accepted:
            env.B = env.B + C - price   # reverse
        elif accepted_nm and env_accepted:
            total += price

        pol.update_sequence_state(d, accepted_nm,
                                  price if accepted_nm else 0.0)
        if done: break

    return total


def _stats(v):
    a = np.array(v, dtype=float)
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2)}


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT} — remove to rerun"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pol = load_arm_b(device)   # returns policy only

    # Cal-DP tier distribution (trivially reported)
    cdp_tier_dist = {
        "note": "CDP-NM uses disc=0.0 at 100% of steps — FIXED RULE",
        "disc_0.0": 1.0,
        "all_others": 0.0,
        "reason": "disc=0.0 is analytically optimal: P(accept) maximised when price=est_val (Gaussian peaked at w_true≈est_val)"
    }

    print("G3 NM extension: arm_b (0b549f93) under NM Gaussian acceptance")
    print()
    print("ACCEPTANCE RULE:")
    print("  Probabilistic (NOT deterministic threshold).")
    print("  Line from g3_nonmono_v4.py:")
    print("      accepted = bool(rng.random() < p_acc)")
    print("  where p_acc = exp(-(price-w_true)^2 / (2*(w_true/2)^2))")
    print()
    print("CAL-DP TIER DISTRIBUTION:")
    print("  CDP-NM selects disc=0.0 at EVERY step (100% fixed rule).")
    print("  DP table NOT re-calibrated for NM — analytically optimal disc=0.")
    print()

    results = {
        "acceptance_rule": "probabilistic: accepted = bool(rng.random() < p_acc)",
        "nm_accept_fn": "p_acc = exp(-(price-w_true)^2 / (2*(w_true/2)^2))",
        "cdp_tier_distribution": cdp_tier_dist,
    }

    for net, loader in NETS.items():
        graph = loader()
        ei, cache = make_ei(graph, device)   # make_ei returns (edge_index, cache)
        results[net] = {}
        for k in KS:
            B = k * C
            arm_b_v = []
            for s in SEEDS:
                r = eval_arm_b_nm_episode(pol, graph, cache, ei, B, s, device)
                arm_b_v.append(r)
            results[net][str(k)] = {"arm_b_NM": _stats(arm_b_v)}
            print(f"  {net} k={k:2d}  arm_b_NM={np.mean(arm_b_v):.1f}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
