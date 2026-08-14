"""experiments/run_stage_a_rollout_gate.py — Stage A: rollout expert quality gate (R0).

STAGE A spec:
  Graph:   FF n=200, c=0.3, k in {1, 3, 10}, 3 seeds each.
  Rollout expert run AS a policy (evaluate_rollout_expert).
  Current regime teachers (for comparison on SAME graphs, same seeds):
    k=1,3  → DP-Cal v3 executor
    k=10   → DP-Cal v2 executor
    plus Greedy+Budget as reference

GATE R0 (pre-committed):
  rollout expert beats the current per-regime teacher by >= 5% at >= 2 of the 3 k values.
  FAIL → stop pilot, report table, idea goes to future work.
  PASS → Stage B.

Wall-clock gate:
  If mean wall-clock per episode for n=200 > 600 s → flag scale-up as infeasible.

Output: results/logs/stage_a_rollout_gate.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np

# ── Ensure project root on path ───────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.graph_generators import generate_forest_fire
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.rollout_expert import (
    RolloutExpertConfig,
    evaluate_rollout_expert,
)
from src.evaluation.budget_baselines import greedy_discount_budget
from src.evaluation.dp_calibrated import dp_calibrated_budget as dp_cal_v1
from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget

# ── Parameters ─────────────────────────────────────────────────────────────────
N_NODES   = 200
P_FF      = 0.37
PB_FF     = 0.32
FF_SEED   = 42         # deterministic graph topology (reproducible)
C         = 0.3
K_VALUES  = [1, 3, 10]
N_TRIALS  = 3
WALL_CLOCK_LIMIT_SEC = 600.0   # 10 min per episode → flag infeasible

RESULTS_PATH = "results/logs/stage_a_rollout_gate.json"

# Regime teacher mapping (pre-committed, matches mixed_expert_trajectories.py):
#   k <=  5 : DP-Cal v3
#   k >= 6  : DP-Cal v2
#   all k   : Greedy+Budget (reference, not the gate teacher)
def _teacher_name(k: int) -> str:
    return "DP-Cal-v3" if k <= 5 else "DP-Cal-v2"


def _run_teacher(graph, k: int, c: float, n_trials: int) -> dict:
    """Run the regime teacher for this k value."""
    B = float(k * c)
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=0, weight_high=2.0)
    if k <= 5:
        return dp_calibrated_v3_budget(graph, cfg=cfg, B=B, c=c, n_trials=n_trials)
    else:
        return dp_calibrated_v2_budget(graph, cfg=cfg, B=B, c=c, n_trials=n_trials)


def _mean_rev(result: dict) -> float:
    """Extract mean revenue from an aggregated result dict."""
    rv = result.get("revenue", {})
    if isinstance(rv, dict):
        return float(rv.get("mean", rv.get("all", [0.0])[0] if rv.get("all") else 0.0))
    return float(rv)


def main() -> None:
    os.makedirs("results/logs", exist_ok=True)

    print("=" * 64)
    print("STAGE A — Rollout Expert Quality Gate (R0)")
    print(f"  Graph: FF n={N_NODES}, p={P_FF}, pb={PB_FF}, seed={FF_SEED}")
    print(f"  k values: {K_VALUES},  n_trials={N_TRIALS},  c={C}")
    print("=" * 64)

    # ── Generate graph ─────────────────────────────────────────────────────────
    graph = generate_forest_fire(N_NODES, p=P_FF, pb=PB_FF, seed=FF_SEED)
    print(f"  Graph: n={graph.number_of_nodes()}, m={graph.number_of_edges()}\n")

    base_cfg = BudgetEnvConfig(
        budget_B=0.3, production_cost=C, seed=0, weight_high=2.0,
        n_mc_samples=200,
    )
    expert_cfg = RolloutExpertConfig(
        c=C,
        rollout_H=20,
        n_mc_rollout=30,
        discount_grid=(0.0, 0.5, 1.0),
        n_cand_degree=5,
        n_cand_val=3,
        j_delta=0.1,
    )

    results: Dict[str, dict] = {}
    gate_row: dict = {}      # k → {"rollout": rev, "teacher": rev, "pct_diff": float}

    for k in K_VALUES:
        print(f"\n── k={k}  (B_0={k*C:.2f}) ──")
        teacher_name = _teacher_name(k)

        # ── Rollout expert ────────────────────────────────────────────────────
        print(f"  Running rollout expert (n_trials={N_TRIALS}) ...")
        rollout_res = evaluate_rollout_expert(
            graph, k=k, c=C, n_trials=N_TRIALS,
            expert_cfg=expert_cfg, base_cfg=base_cfg, verbose=True,
        )
        roll_rev  = rollout_res["revenue"]["mean"]
        wall_mean = rollout_res["wall_clock_sec"]
        print(f"  Rollout expert:  rev={roll_rev:.2f}  "
              f"wall={wall_mean:.1f}s/episode")

        # ── Regime teacher ────────────────────────────────────────────────────
        print(f"  Running {teacher_name} (n_trials={N_TRIALS}) ...")
        teacher_res  = _run_teacher(graph, k=k, c=C, n_trials=N_TRIALS)
        teacher_rev  = _mean_rev(teacher_res)
        print(f"  {teacher_name}: rev={teacher_rev:.2f}")

        # ── Greedy+Budget reference ───────────────────────────────────────────
        greedy_res = greedy_discount_budget(
            graph, B=k * C, c=C, n_trials=N_TRIALS
        )
        greedy_rev = _mean_rev(greedy_res)
        print(f"  Greedy+Budget:   rev={greedy_rev:.2f}")

        # ── Gate metric ───────────────────────────────────────────────────────
        if teacher_rev > 1e-9:
            pct_diff = (roll_rev - teacher_rev) / teacher_rev * 100.0
        else:
            pct_diff = float("inf") if roll_rev > 0 else 0.0
        gate_pass = pct_diff >= 5.0
        print(f"  Rollout vs {teacher_name}: {pct_diff:+.1f}%  "
              f"→ gate={'PASS ✓' if gate_pass else 'FAIL ✗'}")

        wall_infeasible = wall_mean > WALL_CLOCK_LIMIT_SEC
        if wall_infeasible:
            print(f"  ⚠ WALL-CLOCK FLAG: {wall_mean:.0f}s > {WALL_CLOCK_LIMIT_SEC:.0f}s "
                  f"→ scale-up to n=1000 flagged infeasible")

        gate_row[k] = {
            "rollout_rev":       roll_rev,
            "rollout_std":       rollout_res["revenue"]["std"],
            "rollout_all":       rollout_res["revenue"]["all"],
            "teacher_name":      teacher_name,
            "teacher_rev":       teacher_rev,
            "greedy_rev":        greedy_rev,
            "pct_diff":          pct_diff,
            "gate_pass":         gate_pass,
            "wall_clock_sec":    wall_mean,
            "wall_infeasible":   wall_infeasible,
            "bankrupt_count":    rollout_res["bankrupt_count"],
        }
        results[f"k={k}"] = {
            "rollout": rollout_res,
            "teacher": teacher_res,
            "greedy":  greedy_res,
        }

    # ── R0 gate decision ───────────────────────────────────────────────────────
    n_gate_pass = sum(1 for row in gate_row.values() if row["gate_pass"])
    r0_pass = n_gate_pass >= 2
    any_infeasible = any(row["wall_infeasible"] for row in gate_row.values())

    print("\n" + "=" * 64)
    print("GATE R0 VERDICT")
    print(f"  k-values passing (>=5% over teacher): {n_gate_pass}/3  "
          f"(need >= 2)")
    print(f"  R0: {'PASS ✓ — proceed to Stage B' if r0_pass else 'FAIL ✗ — stop pilot, idea to future work'}")
    if any_infeasible:
        print("  ⚠ Wall-clock flag raised: scale-up to n=1000 may be infeasible.")

    print("\nSummary table (FF n=200, c=0.3):")
    print(f"  {'k':>4}  {'Teacher':>12}  {'Teacher rev':>11}  "
          f"{'Rollout rev':>11}  {'Δ%':>7}  {'Pass?':>6}  {'Wall(s)':>8}")
    print("  " + "-" * 72)
    for k in K_VALUES:
        row = gate_row[k]
        verdict = "PASS" if row['gate_pass'] else "FAIL"
        print(f"  {k:>4}  {row['teacher_name']:>12}  "
              f"{row['teacher_rev']:>11.2f}  "
              f"{row['rollout_rev']:>11.2f}  "
              f"{row['pct_diff']:>+7.1f}%  "
              f"{verdict:>6}  "
              f"{row['wall_clock_sec']:>8.1f}")

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "meta": {
            "n_nodes":     N_NODES,
            "p_ff":        P_FF,
            "pb_ff":       PB_FF,
            "ff_seed":     FF_SEED,
            "c":           C,
            "k_values":    K_VALUES,
            "n_trials":    N_TRIALS,
        },
        "gate_r0": {
            "verdict":         "PASS" if r0_pass else "FAIL",
            "n_pass":          n_gate_pass,
            "wall_flag":       any_infeasible,
            "per_k":           gate_row,
        },
        "raw_results": results,
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Exit code: 0 = pass, 1 = R0 fail
    sys.exit(0 if r0_pass else 1)


if __name__ == "__main__":
    main()
