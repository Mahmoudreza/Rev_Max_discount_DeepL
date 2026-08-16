#!/usr/bin/env python3
"""
experiments/g3_rayleigh_nm.py — G3 REDO (correct specification)
=================================================================
Change ONLY the valuation function: influence_model="non_monotone"
(raw Rayleigh PDF, no ceiling clamp) vs "monotone" (clamped).

Acceptance rule: DETERMINISTIC (Def 3.3).
  Quote from revenue_env.py:
      true_val = self._true_valuation(node)
      accepted = (true_val >= offered_price)
  No probability — buyer accepts iff true valuation >= posted price.

Cal-DP RECALIBRATED: V and A tables built with influence_model="non_monotone".

Methods: IE+Budget, Greedy+Budget, Cal-DP (recalibrated NM), arm_b (0b549f93).
Networks: FF_1000, Rice_FB. k=[5,10,20,40]. 10 seeds.
Writes: results/logs/g3_rayleigh_nm.json
"""
from __future__ import annotations
import json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated import _deg_class
from experiments._arm_b_utils import (
    load_arm_b, make_ei, _feat_unconstrained, _avail_mask,
    C as C_ARM, W_HIGH as W_HIGH_ARM, N_MC
)
from src.utils.helpers import set_seed

C = 0.3; W_HIGH = 1.0
SEEDS  = list(range(10))
KS     = [5, 10, 20, 40]
TIERS  = (1.0, 0.8, 0.5, 0.2, 0.0)   # standard discount tiers
NETS   = {"FF_1000": lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
          "Rice_FB": load_rice_facebook}
OUT    = "results/logs/g3_rayleigh_nm.json"

# Acceptance rule (quoted for reference):
ACCEPT_RULE = "accepted = (true_val >= offered_price)  # revenue_env.py, deterministic"


def _nm_cfg(B: float, seed: int) -> BudgetEnvConfig:
    """Config with non-monotone Rayleigh (raw density, no ceiling clamp)."""
    return BudgetEnvConfig(
        budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH,
        influence_model="non_monotone",   # ← the ONLY change vs monotone baseline
    )


def _infl_bucket(x, ib_arr):
    for i in range(len(ib_arr)-2, 0, -1):
        if x >= ib_arr[i]: return i
    return 0


# ── Baseline episodes (use env directly — acceptance handled by NM model) ──────

def run_fixed_disc_episode(graph, cfg: BudgetEnvConfig, disc: float) -> float:
    """Offer all nodes in degree order at fixed discount."""
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    total = 0.0
    for node in ordering:
        if node in env.offered: continue
        if env._check_bankrupt(): break
        ni = env.node_to_idx.get(node)
        if ni is None: continue
        _, reward, done, _ = env.step(ni, disc)
        total += reward
        if done: break
    return total


# ── Cal-DP (recalibrated NM) episode ─────────────────────────────────────────

def run_caldp_nm_episode(graph, cfg: BudgetEnvConfig,
                          V, A, class_bnd, infl_bnd) -> tuple[float, list]:
    """Cal-DP episode with NM-calibrated tables. Returns (revenue, tier_choices)."""
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    total = 0.0; tier_choices = []

    for k_rel, node in enumerate(ordering):
        if node in env.offered: continue
        if env._check_bankrupt(): break
        ni = env.node_to_idx.get(node)
        if ni is None: continue

        cls = int(_deg_class(int(graph.degree(node)), class_bnd))
        try: infl = float(env.get_current_influence(node))
        except: infl = 0.0
        ib  = min(_infl_bucket(infl, infl_bnd), A.shape[1]-1)

        # DP value function lookup
        b_step = min(int(env.B / 0.05), V.shape[-1]-1)

        best_val = -1e18; best_disc = 0.0; best_ti = 0
        for ti, d in enumerate(TIERS):
            est = float(env._estimate_valuation(node))
            price = est * (1.0 - d)
            if env.B - C + price < -1e-9: continue
            p_acc = float(A[cls, ib, ti])
            b_next = min(int((env.B - C + price) / 0.05), V.shape[-1]-1) if p_acc > 0 else b_step
            # Immediate + continuation
            val = p_acc * price + float(V[cls, ib, b_next])
            if val > best_val:
                best_val = val; best_disc = d; best_ti = ti

        tier_choices.append(best_ti)
        _, reward, done, _ = env.step(ni, best_disc)
        total += reward
        if done: break

    return total, tier_choices


# ── arm_b episode (NM env) ────────────────────────────────────────────────────

@torch.no_grad()
def run_arm_b_nm_episode(pol, graph, cache, ei, B: float, seed: int, device) -> float:
    """arm_b policy in NM env. Env handles acceptance (deterministic, NM model)."""
    set_seed(seed)
    n = graph.number_of_nodes()
    cfg = BudgetEnvConfig(
        budget_B=B, production_cost=C, seed=seed, weight_high=W_HIGH,
        influence_model="non_monotone", n_mc_samples=N_MC,
    )
    env = BudgetRevenueEnv(graph, cfg)
    env.reset()
    pol.reset_episode(device)
    total = 0.0

    while env.available_nodes and not env._check_bankrupt():
        x  = torch.FloatTensor(_feat_unconstrained(cache, env, n)).to(device)
        av = _avail_mask(env, n, device)
        if not av.any(): break
        sc, h, ctx, _ = pol.forward(x, ei, av)
        ni = int(sc.argmax().item())
        d  = float(pol.get_discount_distribution(
                   torch.cat([h[ni], ctx])).mean.item())
        _, reward, done, info = env.step(ni, d)
        total += reward
        pol.update_sequence_state(d, info.get("accepted", False),
                                  info.get("revenue_step", 0.0))
        if done: break

    return total


def _stats(v):
    a = np.array(v, dtype=float)
    return {"mean": round(float(a.mean()), 2), "std": round(float(a.std()), 2)}


def main():
    if os.path.exists(OUT):
        print(f"Output exists: {OUT} — remove to rerun"); return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pol    = load_arm_b(device)

    print("G3 REDO: NM Rayleigh valuation (non_monotone), deterministic acceptance")
    print(f"ACCEPT LINE: {ACCEPT_RULE}")
    print()

    results = {
        "acceptance_rule": ACCEPT_RULE,
        "valuation_change": "influence_model='non_monotone' (raw Rayleigh PDF, no ceiling)",
        "vs_monotone":      "influence_model='monotone' (Rayleigh PDF clipped at peak)",
    }

    for net, loader in NETS.items():
        graph = loader()
        ei, cache = make_ei(graph, device)
        results[net] = {}

        # Calibrate Cal-DP tables WITH NM model
        print(f"\n=== {net}: calibrating NM Cal-DP tables... ===")
        cal_cfg0 = BudgetEnvConfig(budget_B=1.5, production_cost=C, seed=0,
                                   weight_high=W_HIGH, influence_model="non_monotone")
        V, A, P, class_bnd, infl_bnd = calibrate_v2_obs_table(
            graph, cal_cfg0, n_sims=30, seed=0)
        print(f"  Tables: V{V.shape} A{A.shape}")

        for k in KS:
            B = k * C
            ie_v, gd_v, cdp_v, arm_v = [], [], [], []
            cdp_tiers_all = []

            for s in SEEDS:
                cfg = _nm_cfg(B, s)
                # IE: disc=0.5 (discounted seeding)
                ie_v.append(run_fixed_disc_episode(graph, cfg, 0.5))
                # Greedy: disc=0.2
                gd_v.append(run_fixed_disc_episode(graph, cfg, 0.2))
                # Cal-DP (recalibrated)
                r, tc = run_caldp_nm_episode(graph, cfg, V, A, class_bnd, infl_bnd)
                cdp_v.append(r); cdp_tiers_all.extend(tc)
                # arm_b
                arm_v.append(run_arm_b_nm_episode(pol, graph, cache, ei, B, s, device))

            # Tier distribution for Cal-DP
            tier_hist = {}
            for ti, d in enumerate(TIERS):
                n_ti = cdp_tiers_all.count(ti)
                tier_hist[f"disc_{d}"] = round(n_ti / max(len(cdp_tiers_all), 1), 3)

            results[net][str(k)] = {
                "IE+Budget":     _stats(ie_v),
                "Greedy+Budget": _stats(gd_v),
                "Cal-DP-NM":     _stats(cdp_v),
                "arm_b_NM":      _stats(arm_v),
                "cdp_tier_dist": tier_hist,
            }
            print(f"  k={k:2d}  IE={np.mean(ie_v):.1f}  GD={np.mean(gd_v):.1f}"
                  f"  CDP={np.mean(cdp_v):.1f}  arm_b={np.mean(arm_v):.1f}"
                  f"  | cdp_tiers={tier_hist}")

    os.makedirs("results/logs", exist_ok=True)
    with open(OUT, "w") as f: json.dump(results, f, indent=2)
    print(f"\nSaved → {OUT}")


if __name__ == "__main__":
    main()
