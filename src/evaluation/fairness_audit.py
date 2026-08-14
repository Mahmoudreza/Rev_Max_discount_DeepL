"""src/evaluation/fairness_audit.py

Fairness metrics for revenue-maximization trajectories.
NEW FILE — does not modify any existing src/ file.

Trajectory format: list of dicts, one per ACCEPTED step, in acceptance order:
    {"node": int, "price": float, "est_val": float}
"""

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

FREE_EPS    = 1e-3   # price below this = "free"
SUBSIDY_FRAC = 0.5   # price < 0.5 * est_val = "subsidized"


def group_metrics_at_checkpoints(
    traj: List[dict],
    labels: np.ndarray,
    checkpoints: List[int],
) -> Dict:
    """Compute per-group fairness metrics at given acceptance-count checkpoints.

    Args:
        traj: List of accepted-step dicts in acceptance order, each with
              {"node": int, "price": float, "est_val": float}.
              est_val may be np.nan if not available.
        labels: np.array of 0/1 per node (1 = minority group B).
        checkpoints: List of K values (K = number of acceptances).

    Returns:
        Dict {K: {metric: value}} with metrics:
          rho_A, rho_B    adoption rate per group |S_K ∩ g| / |g|
          min_rho         min(rho_A, rho_B)
          gap             rho_A - rho_B
          price_ratio_BA  mean price paid by B / by A in S_K
                          (np.nan if either group has no accepted buyers)
          sub_share_B     share of subsidized acceptances going to B
          free_share_B    share of free items going to B
          node_share_B    |B| / n (constant; reference line)
        Plus "final" key with the above + revenue.
    """
    nA = int((labels == 0).sum())
    nB = int((labels == 1).sum())

    accA = accB = 0
    pricesA: List[float] = []
    pricesB: List[float] = []
    subA = subB = freeA = freeB = 0
    ci = 0
    cps = sorted(checkpoints)
    out: Dict = {}

    for t, step in enumerate(traj, start=1):
        node = step["node"]
        p    = float(step["price"])
        ev   = float(step.get("est_val", float("nan")))

        g = int(labels[node])
        if g == 0:
            accA += 1
            pricesA.append(p)
        else:
            accB += 1
            pricesB.append(p)

        # Corrected subsidy definition (2026-08-07):
        # Free items (price=0) ALWAYS count as subsidized — seller forgoes price.
        # est_val guard still applies for non-zero discounts.
        subsidized = (
            p < 1e-3
        ) or (
            not np.isnan(ev) and ev > 1e-6 and p < SUBSIDY_FRAC * ev
        )
        if subsidized:
            if g == 0: subA += 1
            else:      subB += 1
        if p < FREE_EPS:
            if g == 0: freeA += 1
            else:      freeB += 1

        while ci < len(cps) and t == cps[ci]:
            rA = accA / max(nA, 1)
            rB = accB / max(nB, 1)
            out[cps[ci]] = dict(
                rho_A=rA, rho_B=rB,
                min_rho=min(rA, rB),
                gap=rA - rB,
                price_ratio_BA=(
                    float(np.mean(pricesB)) / float(np.mean(pricesA))
                    if pricesA and pricesB and float(np.mean(pricesA)) > 1e-9
                    else float("nan")
                ),
                sub_share_B=subB / max(subA + subB, 1),
                free_share_B=freeB / max(freeA + freeB, 1),
                node_share_B=nB / (nA + nB),
            )
            ci += 1

    # Final state
    rA = accA / max(nA, 1)
    rB = accB / max(nB, 1)
    out["final"] = dict(
        rho_A=rA, rho_B=rB,
        min_rho=min(rA, rB),
        gap=rA - rB,
        price_ratio_BA=(
            float(np.mean(pricesB)) / float(np.mean(pricesA))
            if pricesA and pricesB and float(np.mean(pricesA)) > 1e-9
            else float("nan")
        ),
        sub_share_B=subB / max(subA + subB, 1),
        free_share_B=freeB / max(freeA + freeB, 1),
        node_share_B=nB / (nA + nB),
        revenue=float(sum(s["price"] for s in traj)),
    )
    return out


def aggregate_trials(
    trial_results: List[Dict],
    checkpoints: List[int],
) -> Dict:
    """Aggregate per-trial metric dicts into mean ± std.

    Args:
        trial_results: List of outputs from group_metrics_at_checkpoints.
        checkpoints: Same checkpoints list used when producing trial_results.

    Returns:
        {K: {metric: {"mean": float, "std": float}}} for K in checkpoints+["final"].
    """
    all_keys = list(checkpoints) + ["final"]
    agg: Dict = {}
    for K in all_keys:
        rows = [tr[K] for tr in trial_results if K in tr]
        if not rows:
            continue
        metric_names = list(rows[0].keys())
        agg[K] = {}
        for m in metric_names:
            vals = [r[m] for r in rows if not np.isnan(float(r[m])) if not isinstance(r[m], str)]
            if vals:
                agg[K][m] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
            else:
                agg[K][m] = {"mean": float("nan"), "std": float("nan")}
    return agg


def assert_trajectory_consistent(traj: List[dict], reported_revenue: float) -> None:
    """Assert sum(prices) == reported_revenue and no node appears twice.

    Args:
        traj: List of accepted-step dicts.
        reported_revenue: Revenue value to check against.

    Raises:
        AssertionError: On any violation (loud failure as required).
    """
    nodes_seen = set()
    total = 0.0
    for step in traj:
        node = step["node"]
        assert node not in nodes_seen, (
            f"Accounting violation: node {node} appears more than once in trajectory"
        )
        nodes_seen.add(node)
        total += float(step["price"])

    assert abs(total - reported_revenue) < 1e-6, (
        f"Accounting violation: sum(prices)={total:.8f} != reported_revenue={reported_revenue:.8f}"
    )
