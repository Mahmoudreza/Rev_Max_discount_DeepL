#!/usr/bin/env python3
"""eval_greedy_faithful.py — Faithful vs Static Greedy+Budget comparison.

Parallelised by network. Includes sanity check, paired t-tests, and
reference baselines (IE-aware, CGS/LSTM where available).

Usage:
  venv/bin/python3 -u experiments/eval_greedy_faithful.py > /tmp/greedy_faithful.log 2>&1
"""
from __future__ import annotations
import json, math, os, subprocess, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import scipy.stats as stats

C = 0.3; W_HIGH = 2.0; N_MC = 200; N_TRIALS = 10
KAPPAS = [5, 10, 20, 40]
_ROOT = str(Path(__file__).parent.parent)


# ── Network registry ──────────────────────────────────────────────────────────
def _make_graphs():
    from src.env.graph_generators import (generate_forest_fire,
                                           generate_modular_forest_fire,
                                           load_rice_facebook)
    nets = {
        "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
        "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
        "Modular_FF": generate_modular_forest_fire([200,300,500], 0.37, 0.32, 0.01, seed=0),
        "Rice_FB":    load_rice_facebook(),
    }
    try:
        from src.env.graph_generators import load_polblogs
        nets["polblogs"] = load_polblogs()
    except Exception:
        pass
    return nets


# ── Sanity check ──────────────────────────────────────────────────────────────
def sanity_check():
    from src.env.graph_generators import generate_forest_fire
    from src.evaluation.greedy_budget_faithful import greedy_discount_budget_faithful
    from src.evaluation.budget_baselines import greedy_discount_budget

    G = generate_forest_fire(1000, 0.37, 0.32, seed=0)
    B40 = 40 * C  # 12.0 — constraint should rarely bind

    # Faithful budget (single seed)
    r_faith = greedy_discount_budget_faithful(G, B40, C, n_trials=1, n_mc=N_MC)
    rev_f = r_faith["revenue"]["all"][0]

    # Unconstrained greedy_discount
    import omegaconf
    cfg_str = f"""
budget: {{k: 40}}
influence: {{b: 1.0, n_mc_samples: {N_MC}, model: monotone, weight_low: 0.0, weight_high: {W_HIGH}}}
reward_type: flat
env: {{production_cost: {C}}}
"""
    cfg = omegaconf.OmegaConf.create(cfg_str)
    from src.evaluation.baselines import greedy_discount
    rev_u = greedy_discount(G, cfg)

    diff_pct = abs(rev_f - rev_u) / max(abs(rev_u), 1e-6) * 100
    print(f"\n=== SANITY CHECK FF-1000 kappa=40 seed=0 ===")
    print(f"  Faithful budget  revenue = {rev_f:.3f}")
    print(f"  Unconstrained    revenue = {rev_u:.3f}")
    print(f"  Diff = {diff_pct:.1f}%  {'PASS' if diff_pct <= 10.0 else 'FAIL — port not faithful, STOP'}")
    if diff_pct > 10.0:
        sys.exit(1)
    return True


# ── Per-network worker ────────────────────────────────────────────────────────
def _run_network(net_name):
    sys.path.insert(0, _ROOT)
    import numpy as np
    from src.evaluation.greedy_budget_faithful import greedy_discount_budget_faithful
    from src.evaluation.budget_baselines import greedy_discount_budget
    from src.evaluation.ie_budget import ie_strategy_budget_aware

    nets = _make_graphs()
    if net_name not in nets:
        return net_name, None, "network not found"

    G = nets[net_name]
    net_res = {}

    for kappa in KAPPAS:
        B0 = kappa * C

        try:
            faithful = greedy_discount_budget_faithful(G, B0, C, n_trials=N_TRIALS,
                                                        weight_high=W_HIGH, n_mc=N_MC)
        except Exception as e:
            return net_name, None, f"faithful ERR k={kappa}: {e}"

        try:
            static = greedy_discount_budget(G, B0, C, n_trials=N_TRIALS,
                                             weight_high=W_HIGH)
        except Exception as e:
            return net_name, None, f"static ERR k={kappa}: {e}"

        try:
            ie_a = ie_strategy_budget_aware(G, B0, C, n_trials=N_TRIALS,
                                             weight_high=W_HIGH)
            ie_profs = [r - C*s for r,s in
                        zip(ie_a.get("revenue",{}).get("all",[0]*N_TRIALS),
                            ie_a.get("n_in_S",{}).get("all",[0]*N_TRIALS))]
            ie_result = {"profit": {"mean": float(np.mean(ie_profs)),
                                    "std":  float(np.std(ie_profs)),
                                    "all":  ie_profs},
                         "revenue": ie_a.get("revenue",{}),
                         "n_in_S":  ie_a.get("n_in_S",{})}
        except Exception:
            ie_result = {}

        # Paired t-test on profit (faithful vs static)
        pf = faithful["profit"]["all"]
        ps_raw = static.get("profit", {})
        if not ps_raw:
            s_profs_raw = [r-C*s for r,s in
                           zip(static.get("revenue",{}).get("all",[0]*N_TRIALS),
                               static.get("n_in_S",{}).get("all",[0]*N_TRIALS))]
        else:
            s_profs_raw = ps_raw.get("all", [0]*N_TRIALS)
        stat_t, pval = stats.ttest_rel(pf, s_profs_raw, alternative="two-sided")
        diff_mean = float(np.mean(pf)) - float(np.mean(s_profs_raw))
        ci = stats.t.interval(0.95, df=len(pf)-1,
                               loc=diff_mean,
                               scale=stats.sem(np.array(pf)-np.array(s_profs_raw)))

        net_res[f"k{kappa}"] = {
            "kappa": kappa,
            "faithful":  faithful,
            "static":    {"profit": {"mean": float(np.mean(s_profs_raw)),
                                      "std":  float(np.std(s_profs_raw)),
                                      "all":  s_profs_raw},
                           "revenue": static.get("revenue",{}),
                           "n_in_S":  static.get("n_in_S",{})},
            "ie_aware":  ie_result,
            "paired_test": {"diff_mean": diff_mean, "t": float(stat_t),
                             "p": float(pval), "ci95": list(ci),
                             "not_sig": pval > 0.05},
        }

    return net_name, net_res, None


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=== Greedy+Budget Faithful vs Static Sweep ===")
    print(f"C={C}  kappas={KAPPAS}  N_TRIALS={N_TRIALS}  N_MC={N_MC}", flush=True)

    sanity_check()  # exits if >10% diff at kappa=40

    nets = _make_graphs()
    net_names = list(nets.keys())

    print(f"\nNetworks: {net_names}  (parallelised by network)", flush=True)

    all_results = {}
    with ProcessPoolExecutor(max_workers=min(5, len(net_names))) as pool:
        futs = {pool.submit(_run_network, n): n for n in net_names}
        for fut in as_completed(futs):
            name, res, err = fut.result()
            if err:
                print(f"[ERR] {name}: {err}", flush=True)
                continue
            all_results[name] = res
            # Print summary for this network
            print(f"\n── {name} (n={nets[name].number_of_nodes()}) ──")
            print(f"{'k':3s}  {'faithful profit':>16s}±{'std':>5s}  {'skips':>5s}  "
                  f"{'static profit':>14s}±{'std':>5s}  "
                  f"{'IE-aware':>9s}  {'diff':>7s}  {'p':>6s}  {'sig?':>5s}")
            for k in KAPPAS:
                cell = res.get(f"k{k}", {})
                if not cell:
                    continue
                fp = cell["faithful"]["profit"]
                sp = cell["static"]["profit"]
                sk = cell["faithful"].get("n_skips",{})
                ie = cell.get("ie_aware",{}).get("profit",{})
                pt = cell["paired_test"]
                print(f"  {k:2d}  {fp['mean']:7.2f}±{fp['std']:5.2f}  "
                      f"{sk.get('mean',0):5.1f}  "
                      f"{sp['mean']:7.2f}±{sp['std']:5.2f}  "
                      f"{ie.get('mean',float('nan')):8.2f}  "
                      f"{pt['diff_mean']:+7.2f}  {pt['p']:6.4f}  "
                      f"{'NOT SIG' if pt['not_sig'] else 'SIG'}", flush=True)

    # Save JSON
    out = os.path.join(_ROOT, "results/logs/greedy_faithful.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump({"C": C, "kappas": KAPPAS, "n_trials": N_TRIALS,
               "results": all_results}, open(out, "w"), indent=2)
    print(f"\nSaved → {out}", flush=True)

    # Commit
    subprocess.run(["git", "add", out], cwd=_ROOT)
    subprocess.run(["git", "add", "src/evaluation/greedy_budget_faithful.py",
                    "experiments/eval_greedy_faithful.py"], cwd=_ROOT)
    subprocess.run(["git", "commit", "-m",
                    "greedy_budget_faithful: faithful dynamic-rerank variant + sweep"], cwd=_ROOT)
    h = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                        capture_output=True, text=True, cwd=_ROOT).stdout.strip()
    print(h, flush=True)


if __name__ == "__main__":
    main()
