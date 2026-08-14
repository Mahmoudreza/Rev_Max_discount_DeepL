"""experiments/run_fair_greedy_gate_f1.py

Gate F1 — Evaluate Fair-Greedy on Rice-FB and SBM, compare to Greedy-Discount.
NEW FILE only — does NOT modify any existing file.

Gate F1 criterion (pre-committed):
  PASS iff on Rice-FB:
    Fair-Greedy gap <= 0.5 * Greedy-Discount gap
    AND Fair-Greedy revenue >= 0.85 * Greedy-Discount revenue

Writes results/logs/fairness_f1.json.
"""
from __future__ import annotations
import sys, json, os
sys.path.insert(0, ".")

import numpy as np
import networkx as nx
from omegaconf import OmegaConf

from src.evaluation.baselines import greedy_discount_trajectory
from src.evaluation.fair_baselines import fair_greedy_discount_trajectory
from src.evaluation.fairness_audit import (
    group_metrics_at_checkpoints,
    aggregate_trials,
    assert_trajectory_consistent,
)
from src.env.sbm_generators import two_block_graph, load_rice_facebook_with_labels

# ── Config ────────────────────────────────────────────────────────────────────
BASE_CFG_YAML = """
project:
  name: revmax-aaai2027
  seed: 0
graph:
  type: rice_facebook
  n_nodes: 443
features:
  dim: 16
encoder:
  hidden_dim: 64
  n_layers: 2
  dropout: 0.0
influence:
  model: monotone
  n_mc_samples: 10
  b: 1.0
  weight_low: 0.0
  weight_high: 2.0
reward:
  type: revenue
  gamma: 1.0
budget:
  k: 443
env:
  k: 443
  budget: 0.0
"""

SEEDS = [0, 1, 2, 3, 4]
CHECKPOINTS_K     = {25, 50, 100, 200, 300, 443}
CHECKPOINTS_SBM   = {50, 100, 200, 500, 750, 1000}


def traj_to_audit(raw_traj: list) -> tuple[list, float]:
    """Convert greedy_discount_trajectory or fair_greedy output to audit format."""
    audit = []
    rev = 0.0
    for step in raw_traj:
        if step.get("accepted", False):
            p   = float(step["price"])
            ev  = float(step.get("est_val", p))
            nid = int(step["node_idx"])
            audit.append({"node": nid, "price": p, "est_val": ev})
            rev += p
    return audit, rev


def eval_two_methods(graph: nx.Graph, labels: np.ndarray,
                     checkpoints_K: set, cfg) -> tuple[dict, dict]:
    """Run Greedy-Discount and Fair-Greedy on graph for SEEDS trials.

    Returns (greedy_agg, fair_agg) where each is aggregate_trials output.
    """
    n   = graph.number_of_nodes()
    cps = sorted(k for k in checkpoints_K if k <= n)

    greedy_trials, fair_trials = [], []
    for seed in SEEDS:
        cfg_s = OmegaConf.merge(cfg, OmegaConf.create({"project": {"seed": seed}}))

        # Greedy-Discount
        raw_g = greedy_discount_trajectory(graph, cfg_s)
        audit_g, rev_g = traj_to_audit(raw_g)
        assert_trajectory_consistent(audit_g, rev_g)
        greedy_trials.append(group_metrics_at_checkpoints(audit_g, labels, cps))

        # Fair-Greedy
        raw_f = fair_greedy_discount_trajectory(graph, labels, cfg_s, seed=seed)
        audit_f, rev_f = traj_to_audit(raw_f)
        assert_trajectory_consistent(audit_f, rev_f)
        fair_trials.append(group_metrics_at_checkpoints(audit_f, labels, cps))

    return (
        aggregate_trials(greedy_trials, cps),
        aggregate_trials(fair_trials,   cps),
    )


def print_comparison_table(graph_name: str,
                            greedy_agg: dict, fair_agg: dict,
                            node_share_B: float):
    print(f"\n{'='*72}")
    print(f"  {graph_name}  |  node_share_B={node_share_B:.3f}")
    print(f"{'='*72}")
    hdr = f"  {'Method':<20} {'min_rho':>8} {'gap':>8} {'price_ratio_BA':>14} {'sub_share_B':>12} {'revenue':>10}"
    print(hdr)
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*14} {'-'*12} {'-'*10}")
    for label, agg in [("Greedy-Discount", greedy_agg), ("Fair-Greedy", fair_agg)]:
        fm  = agg.get("final", {})
        mr  = fm.get("min_rho",        {}).get("mean", float("nan"))
        gap = fm.get("gap",             {}).get("mean", float("nan"))
        pr  = fm.get("price_ratio_BA",  {}).get("mean", float("nan"))
        sb  = fm.get("sub_share_B",     {}).get("mean", float("nan"))
        rv  = fm.get("revenue",         {}).get("mean", float("nan"))
        print(f"  {label:<20} {mr:>8.3f} {gap:>8.3f} {pr:>14.3f} {sb:>12.3f} {rv:>10.1f}")
    print()


def gate_f1_check(greedy_agg: dict, fair_agg: dict) -> tuple[bool, str]:
    """Pre-committed Gate F1 rule on Rice-FB final metrics."""
    g_fm  = greedy_agg.get("final", {})
    f_fm  = fair_agg.get("final",   {})

    greedy_gap = g_fm.get("gap",     {}).get("mean", float("nan"))
    fair_gap   = f_fm.get("gap",     {}).get("mean", float("nan"))
    greedy_rev = g_fm.get("revenue", {}).get("mean", float("nan"))
    fair_rev   = f_fm.get("revenue", {}).get("mean", float("nan"))

    crit_gap = (not np.isnan(fair_gap) and not np.isnan(greedy_gap) and
                fair_gap <= 0.5 * greedy_gap)
    crit_rev = (not np.isnan(fair_rev) and not np.isnan(greedy_rev) and
                fair_rev >= 0.85 * greedy_rev)

    detail = (
        f"gap: Fair={fair_gap:.3f} vs 0.5*Greedy={0.5*greedy_gap:.3f} "
        f"({'PASS' if crit_gap else 'FAIL'}) | "
        f"rev: Fair={fair_rev:.1f} vs 0.85*Greedy={0.85*greedy_rev:.1f} "
        f"({'PASS' if crit_rev else 'FAIL'})"
    )
    passed = crit_gap and crit_rev
    return passed, detail


def main():
    cfg = OmegaConf.create(BASE_CFG_YAML)
    all_results: dict = {}

    # ── Rice-FB ──
    print("\n[Gate F1] Loading Rice-FB...")
    G_rice, lbl_rice = load_rice_facebook_with_labels()
    nsB_rice = float((lbl_rice == 1).sum()) / len(lbl_rice)
    cfg_rice = OmegaConf.merge(cfg, OmegaConf.create({
        "graph": {"n_nodes": G_rice.number_of_nodes()}
    }))
    print("[Gate F1] Evaluating Rice-FB...")
    greedy_rice, fair_rice = eval_two_methods(G_rice, lbl_rice, CHECKPOINTS_K, cfg_rice)
    all_results["rice_fb"] = {
        "node_share_B":  nsB_rice,
        "Greedy-Discount": greedy_rice,
        "Fair-Greedy":     fair_rice,
    }
    print_comparison_table("Rice-FB n=443", greedy_rice, fair_rice, nsB_rice)

    # ── SBM graphs ──
    for h in [0.5, 0.7, 0.9]:
        gname = f"SBM_h{h:.1f}"
        print(f"[Gate F1] Evaluating {gname}...")
        G_sbm, lbl_sbm = two_block_graph(n=1000, frac_minority=0.3,
                                          avg_degree=5.0, homophily=h, seed=0)
        nsB_sbm = float((lbl_sbm == 1).sum()) / len(lbl_sbm)
        cfg_sbm = OmegaConf.merge(cfg, OmegaConf.create({
            "graph": {"n_nodes": 1000}, "budget": {"k": 1000}
        }))
        greedy_sbm, fair_sbm = eval_two_methods(G_sbm, lbl_sbm, CHECKPOINTS_SBM, cfg_sbm)
        all_results[gname] = {
            "node_share_B":  nsB_sbm,
            "Greedy-Discount": greedy_sbm,
            "Fair-Greedy":     fair_sbm,
        }
        print_comparison_table(f"{gname} n=1000", greedy_sbm, fair_sbm, nsB_sbm)

    # ── Save ──
    out_path = "results/logs/fairness_f1.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: float(x) if hasattr(x, "__float__") else str(x))
    print(f"\nResults saved to {out_path}")

    # ── GATE F1 ──
    print("\n" + "="*70)
    print("  GATE F1 CHECK (Rice-FB)")
    print("="*70)
    passed, detail = gate_f1_check(
        all_results["rice_fb"]["Greedy-Discount"],
        all_results["rice_fb"]["Fair-Greedy"],
    )
    print(f"  {detail}")
    verdict = "PASS" if passed else "FAIL"
    print(f"\n  GATE F1: {verdict}")

    if not passed:
        print("\n  Gate F1 FAILED. Fair-Greedy does not meet fairness-efficiency trade-off.")
        print("  ACTION REQUIRED: re-examine Fair-Greedy ordering or relax criterion.")
        sys.exit(1)

    print("\n  Gate F1 PASSED → Fair-Greedy promoted; proceed to Phase 5 or paper table.")


if __name__ == "__main__":
    main()
