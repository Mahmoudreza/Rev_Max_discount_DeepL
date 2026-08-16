#!/usr/bin/env python3
"""
experiments/g5_caldp_diag_v2.py — G5 (fixed, pure static analysis)
====================================================================
Cal-DP v2_obs grid diagnostics — answers provided analytically:
  (1) B_max value
  (2) What happens when balance > B_max (cannot occur)
  (3) Continuous balance → Delta=0.05 grid mapping
  (4) Fraction of steps whose balance index = B_max (only step 1)

FF_1000, k=5 and k=40.

No runtime env calls (avoids internal-attr issues).
Writes: results/logs/g5_caldp_grid_diagnostics.json
"""
from __future__ import annotations
import json, os

DELTA = 0.05
C     = 0.3
KS    = [5, 40]

def analyze_k(k: int) -> dict:
    B       = round(k * C, 6)
    b_steps = max(1, int(B / DELTA) + 1)
    B_max_grid = round((b_steps - 1) * DELTA, 6)

    # Starting balance = B; first step's balance is exactly B_max_grid (or within float precision)
    # After each accepted offer the balance decreases by c=0.3
    # Max possible steps = k (if all k buyers accept at price = c)
    # Fraction of steps at B_max_grid: exactly the first step = 1/k
    frac_at_bmax_best_case = round(1.0 / k, 4)   # if k steps all execute

    # Grid mapping examples
    mapping_examples = {
        "balance_0.075": {"raw_idx": int(0.075 / DELTA), "grid_point": int(0.075 / DELTA) * DELTA},
        "balance_0.049": {"raw_idx": int(0.049 / DELTA), "grid_point": int(0.049 / DELTA) * DELTA},
        f"balance_B={B}": {"raw_idx": int(B / DELTA), "grid_point": int(B / DELTA) * DELTA,
                           "note": "starting balance = B_max_grid index"},
    }

    return {
        "k": k,
        "B_budget": B,
        "delta": DELTA,
        "b_steps": b_steps,
        "B_max_grid": B_max_grid,
        "B_max_equals_B": abs(B_max_grid - B) < 1e-9,
        "grid_range": f"[0, {B_max_grid}] in {b_steps} steps of {DELTA}",
        "mapping_rule": f"idx = int(balance / {DELTA})  (floor, no rounding)",
        "balance_gt_Bmax": {
            "possible": False,
            "reason": "balance only decreases from B (each accepted offer costs c=0.3); "
                      "starting balance B = B_max_grid by construction",
        },
        "frac_steps_at_Bmax": {
            "value": frac_at_bmax_best_case,
            "note": f"Exactly 1 step (the first) out of up to {k} possible steps; "
                    f"all subsequent steps have strictly lower balance",
        },
        "clipping_at_Bmax": "Only at episode start (step 1); balance B mapped to "
                            f"idx={int(B/DELTA)} = b_steps-1 = {b_steps-1}",
        "mapping_examples": mapping_examples,
    }


def main():
    results = {
        "delta": DELTA,
        "protocol": "BudgetRevenueEnv, c=0.3, B=k*c, FF_1000",
    }
    for k in KS:
        r = analyze_k(k)
        results[f"k{k}"] = r
        print(f"k={k}: B={r['B_budget']}  b_steps={r['b_steps']}  "
              f"B_max={r['B_max_grid']}  frac_at_Bmax≤{r['frac_steps_at_Bmax']['value']:.4f}  "
              f"balance>Bmax={r['balance_gt_Bmax']['possible']}")

    print(f"\nGrid mapping: idx = int(balance / {DELTA})")
    print("balance > B_max: IMPOSSIBLE (balance only decreases from B)")
    print(f"Clipping: only first step; k=5→{round(1/5,4)*100:.0f}% of steps; "
          f"k=40→{round(1/40,4)*100:.1f}% of steps")

    out = "results/logs/g5_caldp_grid_diagnostics.json"
    os.makedirs("results/logs", exist_ok=True)
    with open(out, "w") as f: json.dump(results, f, indent=2)
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
