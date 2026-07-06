"""
src/evaluation/tc_evaluation.py — Time-Critical Revenue Evaluation (Idea 2).

Sequential model only (Babaei et al.). No IC cascade. No TimeCriticalRevenueEnv.

A "deadline" τ means: revenue from the first τ ACCEPTANCES (|S| = τ).
This is identical to revenue_at_k(cum_rev_by_S, τ) from paper_eval.py.

The same acceptance-curve trajectories from Idea 1 can be evaluated at
multiple τ values without re-running any simulation.

Key functions:
    revenue_at_checkpoints  — read cum_rev curve at τ values
    revenue_area_under_curve — total front-loaded revenue score
    evaluate_tc_comparison  — compare 4 methods across τ checkpoints
    make_latex_table        — generate paper-ready LaTeX table
"""

import numpy as np
from typing import Dict, List, Optional, Tuple


# ── Core checkpoint functions ─────────────────────────────────────────────────

def revenue_at_checkpoints(
    cum_rev_by_S: List[float],
    checkpoints: List[int],
) -> Dict[int, float]:
    """Revenue at each deadline τ from an acceptance-indexed curve.

    τ = number of ACCEPTANCES (buyers who joined S).
    cum_rev_by_S[i] = cumulative revenue after the (i+1)-th acceptance.
    This is the same indexing as paper_eval.revenue_at_k().

    Args:
        cum_rev_by_S: Acceptance-indexed cumulative revenue list.
                      Length = total acceptances in the episode.
        checkpoints:  List of τ values, e.g. [50, 100, 200, 300, 500, 1000].
                      τ values beyond episode length clamp to final revenue.

    Returns:
        Dict mapping τ → cumulative revenue at that checkpoint.
    """
    if not cum_rev_by_S:
        return {tau: 0.0 for tau in checkpoints}
    n = len(cum_rev_by_S)
    result: Dict[int, float] = {}
    for tau in checkpoints:
        idx = min(tau - 1, n - 1)
        result[tau] = float(cum_rev_by_S[idx]) if idx >= 0 else 0.0
    return result


def cumulative_revenue_curve(
    cum_rev_by_S: List[float],
) -> List[Tuple[int, float]]:
    """Full acceptance-step → cumulative-revenue curve.

    Args:
        cum_rev_by_S: Acceptance-indexed cumulative revenue.

    Returns:
        List of (k, revenue) tuples, one per acceptance event.
    """
    return [(i + 1, float(v)) for i, v in enumerate(cum_rev_by_S)]


def revenue_area_under_curve(
    cum_rev_by_S: List[float],
    max_k: Optional[int] = None,
) -> float:
    """Area under the cumulative revenue curve (front-loading score).

    Area = Σ_{k=1}^{max_k} revenue_at_k(k)  (discrete integral)

    Higher area = more revenue earned early.
    A method earning $100 at k=1 contributes 100 × max_k to area.
    A method earning $100 at k=max_k contributes 100 × 1 to area.

    Args:
        cum_rev_by_S: Acceptance-indexed cumulative revenue.
        max_k:        Integration limit (default: len(cum_rev_by_S)).

    Returns:
        Total area (float).
    """
    if not cum_rev_by_S:
        return 0.0
    n = len(cum_rev_by_S)
    if max_k is None:
        max_k = n
    area = 0.0
    for k in range(1, max_k + 1):
        idx = min(k - 1, n - 1)
        area += float(cum_rev_by_S[idx])
    return area


# ── Multi-method comparison ───────────────────────────────────────────────────

def evaluate_tc_comparison(
    methods_curves: Dict[str, List[List[float]]],
    checkpoints: List[int],
) -> Dict[str, Dict]:
    """Compare multiple methods at multiple deadline checkpoints.

    Args:
        methods_curves: Dict of method_name → list of cum_rev_by_S curves,
                        one curve per seed/trial.
        checkpoints:    List of τ values, e.g. [50, 100, 200, 300, 500, 1000].

    Returns:
        Dict of method_name → {
            "checkpoints":     {τ: mean_revenue},
            "checkpoints_std": {τ: std_revenue},
            "area":            float (mean area under curve),
            "area_std":        float,
            "n_trials":        int,
        }
    """
    results: Dict[str, Dict] = {}
    for method, curves in methods_curves.items():
        if not curves:
            results[method] = {
                "checkpoints":     {t: 0.0 for t in checkpoints},
                "checkpoints_std": {t: 0.0 for t in checkpoints},
                "area": 0.0, "area_std": 0.0, "n_trials": 0,
            }
            continue

        all_cp: Dict[int, List[float]] = {tau: [] for tau in checkpoints}
        all_areas: List[float] = []

        for curve in curves:
            cp_revs = revenue_at_checkpoints(curve, checkpoints)
            for tau in checkpoints:
                all_cp[tau].append(cp_revs[tau])
            all_areas.append(revenue_area_under_curve(curve))

        results[method] = {
            "checkpoints":     {tau: float(np.mean(vs)) for tau, vs in all_cp.items()},
            "checkpoints_std": {tau: float(np.std(vs))  for tau, vs in all_cp.items()},
            "area":            float(np.mean(all_areas)),
            "area_std":        float(np.std(all_areas)),
            "n_trials":        len(curves),
        }
    return results


def compute_tc_reward(
    cum_rev_by_S: List[float],
    training_checkpoints: List[int],
    training_weights: List[float],
) -> float:
    """Multi-checkpoint training reward for TC-REINFORCE.

    R = Σ_i  w_i × Revenue(τ_i) / τ_i

    Each term is normalised by its OWN checkpoint τ_i (not episode length n).
    This ensures early-deadline terms have comparable magnitude to
    full-episode terms:
      - Rev(100)/100 ≈ 0.05–0.15   (avg revenue per acceptance at τ=100)
      - Rev(1000)/1000 ≈ 0.05–0.45 (avg revenue per acceptance at τ=1000)

    Using n as denominator instead makes Rev(100)/n ≈ 0.01 vs Rev(1000)/n ≈ 0.45,
    so the early-deadline signal contributes < 5% of the gradient — nearly invisible.
    Per-τ normalisation makes τ=100 contribute ~28% with default weights [0.5, 0.3, 0.2].

    Args:
        cum_rev_by_S:          Acceptance-indexed cumulative revenue from episode.
        training_checkpoints:  τ values, e.g. [100, 300, 1000].
        training_weights:      Weights summing to 1.0, e.g. [0.5, 0.3, 0.2].

    Returns:
        Scalar reward (float).
    """
    cp_revs = revenue_at_checkpoints(cum_rev_by_S, training_checkpoints)
    reward = 0.0
    for tau, w in zip(training_checkpoints, training_weights):
        # Divide by τ so each term is O(avg_price) regardless of deadline length
        reward += w * cp_revs.get(tau, 0.0) / max(tau, 1)
    return reward


# ── Profit analysis ───────────────────────────────────────────────────────────

def cum_rev_to_trajectory(cum_rev_by_S: List[float]) -> List[dict]:
    """Convert acceptance-indexed cumulative revenue to step-dict trajectory.

    Each entry in cum_rev_by_S represents cumulative revenue after the k-th
    acceptance. This helper extracts the individual prices as differences.

    Args:
        cum_rev_by_S: [rev_after_1st_accept, rev_after_2nd_accept, ...]

    Returns:
        List of dicts: [{"accepted": True, "price": price_k}, ...]
    """
    trajectory = []
    prev = 0.0
    for rev in cum_rev_by_S:
        trajectory.append({"accepted": True, "price": float(rev - prev)})
        prev = rev
    return trajectory


def profit_at_checkpoints(
    trajectory: List[dict],
    checkpoints: List[int],
    production_cost: float,
) -> Dict[int, float]:
    """Cumulative profit (revenue - cost) at each checkpoint τ.

    τ = number of ACCEPTANCES.
    Each acceptance costs production_cost to fulfill.
    Rejections cost nothing (item not produced).

    Args:
        trajectory:       List of step dicts with "accepted" and "price" keys.
                          Accepted steps: {"accepted": True,  "price": float}.
                          Rejected steps: {"accepted": False, "price": 0.0}.
        checkpoints:      List of τ values (acceptance counts).
        production_cost:  Cost c per item delivered.

    Returns:
        Dict mapping τ → cumulative profit at that acceptance checkpoint.
    """
    n_accepted = 0
    cum_profit = 0.0
    curve: List[tuple] = []

    for step in trajectory:
        if step.get("accepted", False):
            n_accepted += 1
            profit_step = float(step.get("price", 0.0)) - production_cost
            cum_profit += profit_step
            curve.append((n_accepted, cum_profit))

    result: Dict[int, float] = {}
    for tau in checkpoints:
        matching = [p for (acc, p) in curve if acc <= tau]
        result[tau] = matching[-1] if matching else 0.0
    return result


def breakeven_point(
    trajectory: List[dict],
    production_cost: float,
) -> Optional[int]:
    """Find first τ where cumulative profit becomes positive.

    Args:
        trajectory:      List of step dicts (see profit_at_checkpoints).
        production_cost: Cost c per item delivered.

    Returns:
        int: Number of acceptances to break even, or None if never profitable.
    """
    n_accepted = 0
    cum_profit = 0.0
    for step in trajectory:
        if step.get("accepted", False):
            n_accepted += 1
            cum_profit += float(step.get("price", 0.0)) - production_cost
            if cum_profit > 0:
                return n_accepted
    return None


def profit_curve(
    trajectory: List[dict],
    production_cost: float,
) -> List[Tuple[int, float]]:
    """Full cumulative profit curve for plotting.

    Args:
        trajectory:      List of step dicts (see profit_at_checkpoints).
        production_cost: Cost c per item delivered.

    Returns:
        List of (n_accepted, cumulative_profit) tuples — one per acceptance.
    """
    n_accepted = 0
    cum_profit = 0.0
    curve: List[Tuple[int, float]] = []
    for step in trajectory:
        if step.get("accepted", False):
            n_accepted += 1
            cum_profit += float(step.get("price", 0.0)) - production_cost
            curve.append((n_accepted, cum_profit))
    return curve


# ── LaTeX table generation ────────────────────────────────────────────────────

def make_latex_table(
    tc_results: Dict[str, Dict],
    checkpoints: List[int],
    method_order: Optional[List[str]] = None,
    caption: str = (
        "Time-critical revenue at deadline $\\tau$ (FF $n{=}1000$, 10 seeds). "
        "$\\tau$ = number of acceptances before which revenue is counted."
    ),
    label: str = "tab:time_critical",
) -> str:
    """Generate LaTeX table of revenue-at-deadline for paper.

    Args:
        tc_results:   Output of evaluate_tc_comparison().
        checkpoints:  List of τ column values.
        method_order: Row order (default: all methods in dict order).
        caption:      LaTeX \\caption text.
        label:        LaTeX \\label text.

    Returns:
        Complete LaTeX table string, ready to paste into paper.
    """
    if method_order is None:
        method_order = list(tc_results.keys())

    # Best value per column (for bold)
    col_bests: Dict[int, float] = {}
    for tau in checkpoints:
        col_bests[tau] = max(
            (tc_results[m]["checkpoints"].get(tau, 0.0)
             for m in method_order if m in tc_results),
            default=0.0,
        )

    OURS = {"Rev-GNN-IM-RL", "Rev-GNN-LSTM", "Rev-GNN-LSTM-TC"}
    col_header = " & ".join(f"$\\tau{{{{{t}}}}}$" for t in checkpoints)
    col_spec = "l" + "c" * len(checkpoints)

    rows: List[str] = []
    in_ours = False
    for m in method_order:
        if m not in tc_results:
            continue
        is_ours = m in OURS
        if is_ours and not in_ours:
            rows.append("\\midrule")
            in_ours = True

        vals: List[str] = []
        for tau in checkpoints:
            mv = tc_results[m]["checkpoints"].get(tau, 0.0)
            sv = tc_results[m]["checkpoints_std"].get(tau, 0.0)
            cell = f"{mv:.1f}"
            if abs(mv - col_bests[tau]) < 0.05 and mv > 0.05:
                cell = f"\\textbf{{{cell}}}"
            vals.append(cell)

        rows.append(f"  {m} & " + " & ".join(vals) + " \\\\")

    tex = (
        "\\begin{table}[t]\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        "\\centering\\small\n"
        f"\\begin{{tabular}}{{{col_spec}}}\n\\toprule\n"
        f"Method & {col_header} \\\\\n\\midrule\n"
        + "\n".join(rows) + "\n"
        "\\bottomrule\n\\end{tabular}\n\\end{table}\n"
    )
    return tex
