"""src/evaluation/bmin_feasibility.py — Minimum-budget feasibility analysis.

Post-hoc analysis over budget trajectories stored in eval JSON files.
A strategy that dips below bmin_frac * B0 at any point is declared infeasible
(solvency floor violation), regardless of final revenue.

Public API:
  feasibility_from_trajectory   — single-trajectory predicate
  apply_bmin_analysis           — batch analysis over a results JSON file
"""

from __future__ import annotations

import json
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

_CACHE_DIR = "results/logs"


def feasibility_from_trajectory(
    budget_trajectory: List[float],
    B0: float,
    bmin_frac: float,
) -> bool:
    """Return True iff min(budget_trajectory) >= bmin_frac * B0.

    Args:
        budget_trajectory: List of budget levels recorded at each step.
        B0:                Initial budget.
        bmin_frac:         Minimum acceptable budget fraction (0.0→no floor,
                           0.25→25% floor, 0.5→50% floor).

    Returns:
        True if all budget levels satisfy the solvency floor.
    """
    if not budget_trajectory:
        return bmin_frac <= 0.0
    if B0 <= 0:
        return True
    return float(min(budget_trajectory)) >= bmin_frac * B0 - 1e-9


def apply_bmin_analysis(
    results_json_path: str,
    bmin_fracs: Tuple[float, ...] = (0.0, 0.25, 0.5),
    c: float = 0.3,
    out_path: Optional[str] = None,
) -> dict:
    """Load a budget_eval JSON and compute feasibility + revenue metrics.

    The JSON is expected to have structure:
      data[k_label][method_name] = agg_dict
    where agg_dict contains:
      "budget_trajectory": list of lists (one per trial)
      "revenue":           {"mean": float, "std": float, "all": [float, ...]}

    For each bmin_frac, method, and budget level:
      feasible_rate    = fraction of trials with min_B >= bmin_frac * B0
      revenue_feasible = mean revenue over feasible trials (NaN if none)

    Saves to results/logs/bmin_analysis.json.

    Args:
        results_json_path: Path to a budget_eval_*.json file.
        bmin_fracs:        Tuple of solvency-floor fractions to evaluate.
        c:                 Production cost (used to infer B0 from k_label).

    Returns:
        Nested dict: analysis[frac_key][k_label][method_name] = metrics_dict.
    """
    os.makedirs(_CACHE_DIR, exist_ok=True)

    with open(results_json_path) as f:
        data = json.load(f)

    analysis: dict = {}

    for bmin_frac in bmin_fracs:
        frac_key = f"bmin_{int(round(bmin_frac * 100))}"
        analysis[frac_key] = {}

        for k_label, methods in data.items():
            analysis[frac_key][k_label] = {}

            # Infer B0 from k_label ("k=X" → B0 = X * c)
            try:
                k_val = float(k_label.split("=")[1])
                B0    = k_val * c
            except Exception:
                B0 = 1.0

            for method_name, agg in methods.items():
                trajs = agg.get("budget_trajectory", None)
                revs  = agg.get("revenue", {})
                all_revs = revs.get("all", []) if isinstance(revs, dict) else []

                if trajs is None or len(trajs) == 0:
                    analysis[frac_key][k_label][method_name] = {
                        "feasible_rate":    None,
                        "revenue_feasible": None,
                        "n_trials":         0,
                        "n_feasible":       0,
                    }
                    continue

                # Each element of trajs is a list of budget values for one trial.
                feasible_mask = [
                    feasibility_from_trajectory(
                        traj if isinstance(traj, list) else [], B0, bmin_frac
                    )
                    for traj in trajs
                ]

                n_feasible    = int(sum(feasible_mask))
                feasible_rate = n_feasible / len(feasible_mask)

                if n_feasible > 0 and len(all_revs) == len(feasible_mask):
                    rev_feasible = float(
                        np.mean([all_revs[i] for i, f in enumerate(feasible_mask) if f])
                    )
                elif n_feasible > 0:
                    # Fallback: use overall mean if per-trial revenues unavailable
                    rev_feasible = (
                        float(revs.get("mean", math.nan))
                        if isinstance(revs, dict)
                        else math.nan
                    )
                else:
                    rev_feasible = math.nan

                analysis[frac_key][k_label][method_name] = {
                    "feasible_rate":    feasible_rate,
                    "revenue_feasible": rev_feasible,
                    "n_trials":         len(feasible_mask),
                    "n_feasible":       n_feasible,
                }

    if out_path is None:
        out_path = os.path.join(_CACHE_DIR, "bmin_analysis.json")
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2, default=lambda x: None if math.isnan(x) else x)

    return analysis
