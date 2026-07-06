#!/usr/bin/env python
"""
experiments/run_time_critical.py — Evaluate all methods under time-critical cascade.

Runs all τ values [1..10, 9999] in ONE pass:
  • Cascade runs with max_tau=10, revenue_per_day saved for each trial.
  • Any τ query = sum(revenue_per_day[0..τ]) — no recomputation.
  • Results saved to results/logs/tc_all_taus.json (master cache).
  • Budget sweeps (at τ=3) saved separately.

Usage:
    python experiments/run_time_critical.py \\
        --config configs/experiments/time_critical.yaml
    python experiments/run_time_critical.py \\
        --config configs/experiments/time_critical.yaml --force
"""

import argparse, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from src.utils.helpers import load_config_with_base, set_seed, ensure_dir
from src.env.graph_generators import generate_forest_fire, load_rice_facebook
from src.evaluation.paper_eval import load_lstm, load_im
from src.evaluation.tc_baselines import (
    run_time_critical_comparison, run_tc_budget_sweep,
    save_tc_results, query_revenue_at_tau,
)

import os, json, numpy as np
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LSTM_CKPT = "results/checkpoints/rev_gnn_lstm.pt"
IMRL_CKPT = "results/checkpoints/rev_gnn_im_rl.pt"
LOG_DIR   = "results/logs"
FIG_DIR   = "results/figures"
MASTER    = f"{LOG_DIR}/tc_all_taus.json"


def _load_json(p):
    with open(p) as f: return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",        default="configs/experiments/time_critical.yaml")
    p.add_argument("--n-graph-seeds", type=int, default=5)
    p.add_argument("--force",         action="store_true", help="Ignore caches, rerun all")
    args = p.parse_args()

    ensure_dir(LOG_DIR); ensure_dir(FIG_DIR)
    cfg    = load_config_with_base(args.config)
    set_seed(cfg.project.seed)
    device = torch.device("cpu")
    t0     = time.time()

    print("=" * 60)
    print("run_time_critical.py — τ ∈ {1..10, ∞}")
    print("=" * 60)

    lstm_pol = load_lstm(LSTM_CKPT, cfg, device)
    im_pol   = load_im(IMRL_CKPT,  cfg, device)
    print(f"Loaded LSTM ({sum(p.numel() for p in lstm_pol.parameters()):,}) "
          f"and IM-RL ({sum(p.numel() for p in im_pol.parameters()):,}) policies")

    p_ff, pb_ff = cfg.graph.p, cfg.graph.pb
    FF1000 = lambda s: generate_forest_fire(1000, p_ff, pb_ff, seed=s)

    # ── Tau sweep: FF n=1000 ──────────────────────────────────────────────────
    tau_ff_path = f"{LOG_DIR}/tc_ff_tau_sweep.json"
    if not args.force and os.path.exists(tau_ff_path):
        print(f"[cache] Loading {tau_ff_path}")
        tau_ff = _load_json(tau_ff_path)
    else:
        print(f"[run] FF n=1000 τ-sweep ({args.n_graph_seeds} graphs × 10 trials each)...")
        tau_ff = {}
        for gs in range(args.n_graph_seeds):
            graph = FF1000(gs)
            print(f"  graph_seed={gs+1}/{args.n_graph_seeds} (n={graph.number_of_nodes()})...",
                  end=" ", flush=True)
            t1 = time.time()
            res = run_time_critical_comparison(
                graph, cfg,
                lstm_policy=lstm_pol, imrl_policy=im_pol, device=device,
            )
            tau_ff[str(gs)] = res
            print(f"{time.time()-t1:.0f}s")
        save_tc_results(tau_ff, tau_ff_path)
        print(f"  Saved {tau_ff_path}")

    # ── Tau sweep: Rice-Facebook ──────────────────────────────────────────────
    tau_rice_path = f"{LOG_DIR}/tc_rice_tau_sweep.json"
    tau_rice      = None
    try:
        rf = load_rice_facebook(data_dir="data/raw")
        if not args.force and os.path.exists(tau_rice_path):
            print(f"[cache] Loading {tau_rice_path}")
            tau_rice = _load_json(tau_rice_path)
        else:
            print(f"[run] Rice-Facebook τ-sweep...")
            tau_rice = {"0": run_time_critical_comparison(
                rf, cfg, lstm_policy=lstm_pol, imrl_policy=im_pol, device=device,
            )}
            save_tc_results(tau_rice, tau_rice_path)
            print(f"  Saved {tau_rice_path}")
    except FileNotFoundError as e:
        print(f"[SKIP] Rice-Facebook: {e}")

    # ── Budget sweep τ=3, FF ─────────────────────────────────────────────────
    budget_ff_path = f"{LOG_DIR}/tc_budget_ff_tau3.json"
    k_values       = list(cfg.evaluation.k_values_ff)
    if not args.force and os.path.exists(budget_ff_path):
        print(f"[cache] {budget_ff_path}")
        budget_ff = _load_json(budget_ff_path)
    else:
        print(f"[run] Budget sweep FF τ=3 ...")
        budget_ff = run_tc_budget_sweep(
            FF1000, cfg, k_values=k_values, tau_deadline=3,
            n_seeds=args.n_graph_seeds,
            lstm_policy=lstm_pol, imrl_policy=im_pol, device=device,
        )
        save_tc_results(budget_ff, budget_ff_path)
        print(f"  Saved {budget_ff_path}")

    # ── Budget sweep τ=3, Rice-FB ────────────────────────────────────────────
    budget_rice_path = f"{LOG_DIR}/tc_budget_rice_tau3.json"
    budget_rice      = None
    if tau_rice is not None:
        k_rf = list(cfg.evaluation.k_values_rice)
        if not args.force and os.path.exists(budget_rice_path):
            print(f"[cache] {budget_rice_path}")
            budget_rice = _load_json(budget_rice_path)
        else:
            print(f"[run] Budget sweep Rice-FB τ=3 ...")
            try:
                budget_rice = run_tc_budget_sweep(
                    lambda _: load_rice_facebook(data_dir="data/raw"),
                    cfg, k_values=k_rf, tau_deadline=3,
                    n_seeds=1,
                    lstm_policy=lstm_pol, imrl_policy=im_pol, device=device,
                )
                save_tc_results(budget_rice, budget_rice_path)
                print(f"  Saved {budget_rice_path}")
            except Exception as e:
                print(f"  [SKIP] {e}")

    # ── Master cache: aggregate all taus across graph seeds ──────────────────
    print("\nBuilding master tau-summary...")
    TAU_VALUES = list(cfg.evaluation.tau_values)   # [1,2,...,10,9999]
    METHODS    = ["IE-Strategy", "Greedy-Discount", "Rev-GNN-IM-RL", "Rev-GNN-LSTM"]

    master = {"tau_values": TAU_VALUES, "datasets": {}}

    for dname, raw_data in [("FF n=1000", tau_ff), ("Rice-FB n=443", tau_rice)]:
        if raw_data is None:
            continue
        by_tau = {}
        for tau in TAU_VALUES:
            by_tau[str(tau)] = {}
            for m in METHODS:
                all_trial_revs = []
                for gs_key, gs_data in raw_data.items():
                    if m in gs_data:
                        mean, std = query_revenue_at_tau(gs_data, m, tau)
                        all_trial_revs.append(
                            float(sum(
                                sum(rpd[:min(tau + 1, len(rpd))])
                                for rpd in gs_data[m]["per_trial_rpd"]
                            ) / max(len(gs_data[m]["per_trial_rpd"]), 1))
                        )
                if all_trial_revs:
                    arr = np.array(all_trial_revs)
                    by_tau[str(tau)][m] = {"mean": float(arr.mean()), "std": float(arr.std())}
        master["datasets"][dname] = by_tau
    save_tc_results(master, MASTER)
    print(f"Saved master cache: {MASTER}")

    # ── Results table ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    print(f"\n{'='*60}\nAll done in {elapsed/60:.1f} min")
    print(f"\nKey results (FF n=1000):")
    print(f"{'Method':<22}  " + "   ".join(f"τ={t:<5}" for t in [1, 3, 5, 10]))
    print("-" * 70)
    ff_by_tau = master["datasets"].get("FF n=1000", {})
    for m in METHODS:
        vals = []
        for tau in [1, 3, 5, 10]:
            mv = ff_by_tau.get(str(tau), {}).get(m, {}).get("mean", float("nan"))
            vals.append(f"{mv:7.2f}")
        print(f"  {m:<22}: " + "   ".join(vals))

    print(f"\nCaches:")
    print(f"  τ-sweep FF   → {tau_ff_path}")
    print(f"  τ-sweep Rice → {tau_rice_path}")
    print(f"  Budget FF    → {budget_ff_path}")
    print(f"  Budget Rice  → {budget_rice_path}")
    print(f"  Master       → {MASTER}")
    print(f"\nNext: run experiments/run_time_critical.py then src/utils/tc_visualization.py")


if __name__ == "__main__":
    main()
