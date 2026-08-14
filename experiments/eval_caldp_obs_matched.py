"""
experiments/eval_caldp_obs_matched.py
Full matched-seed Cal-DP obs sweep:
  k = [5,10,15,20,30,40], seeds [42,123,7], frozen 5x5 calibration.
  Composite = per-k max(v2_obs, v3_obs).
  Saves results/logs/caldp_obs_matched_seeds.json.
  Prints 5x6 table.

Accounting-checker fix note:
  The env's stored "accounting_err" uses env.t (total offers), but the
  correct identity is B_T = B_0 - c * n_accepted + R. This script uses
  n_acc (returned by _execute_v2/_execute_v3) for the correct check.
"""
import json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.budget_baselines import _make_env
from src.evaluation.dp_calibrated import _deg_class
from src.evaluation.dp_calibrated_v2 import _plan_dp_v2, _execute_v2, _N_CLASSES, _N_BUCKETS
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table
from src.evaluation.dp_calibrated_v3 import _plan_dp_v3, _execute_v3, _N_S_BUCKETS
from src.evaluation.dp_calibrated_v3_obs import calibrate_v3_obs_table

C      = 0.3
SEEDS  = [42, 123, 7]
K_ALL  = [5, 10, 15, 20, 30, 40]
TIERS  = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA  = 0.05
N_SIMS = 5
OUT    = "results/logs/caldp_obs_matched_seeds.json"

GRAPHS = {
    "polblogs":   load_polblogs(),
    "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook(),
    "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
}
cfg = BudgetEnvConfig(production_cost=C, weight_high=1.0)


def eval_net(net, graph, k_list, n_sims=N_SIMS):
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    V2, A2, P2, cb2, ib2 = calibrate_v2_obs_table(graph, cfg, n_sims=n_sims, seed=0)
    V3, A3, T3, cb3, sb3 = calibrate_v3_obs_table(graph, cfg, n_sims=n_sims, seed=0)
    cpos = np.array([_deg_class(int(all_deg[i]), cb2) for i in range(n)], dtype=np.int32)

    out = {}
    for k in k_list:
        B = k * C
        b_steps = max(1, int(B / DELTA) + 1)
        plan2 = _plan_dp_v2(n_total=n, V=V2, A=A2, P=P2, class_of_pos=cpos,
                            B=B, c=C, tiers=TIERS, delta=DELTA)
        dp3, tier3 = _plan_dp_v3(n_total=n, V3=V3, A3=A3, T=T3,
                                  class_of_pos=cpos, B=B, c=C, sb_size=sb3,
                                  n_s_buckets=_N_S_BUCKETS, tiers=TIERS, delta=DELTA)
        revs2, revs3 = [], []
        acct_ok_all, bk_count = True, 0
        for seed in SEEDS:
            env2 = _make_env(graph, B=B, c=C, seed=seed, weight_high=1.0)
            env2.reset()
            rev2, nacc2, _ = _execute_v2(
                env=env2, ordering=ordering, plan=plan2,
                V=V2, A=A2, class_boundaries=cb2, infl_boundaries=ib2, c=C,
                class_of_pos=cpos,
                dp_table=[[0.0]*(n+1) for _ in range(b_steps+1)],
                b_steps=b_steps, delta=DELTA, tiers=TIERS)
            # correct accounting: B_T = B_0 - c * n_accepted + R
            acct2 = abs(env2.B - (B - C * nacc2 + rev2))
            bk2   = env2._check_bankrupt()

            env3 = _make_env(graph, B=B, c=C, seed=seed, weight_high=1.0)
            env3.reset()
            rev3, nacc3, _, _ = _execute_v3(
                env=env3, ordering=ordering, dp3=dp3, tier3=tier3,
                V3=V3, A3=A3, class_boundaries=cb3, sb_size=sb3, c=C,
                class_of_pos=cpos, n_s_buckets=_N_S_BUCKETS,
                b_steps=b_steps, delta=DELTA, tiers=TIERS, log_steps=10)
            acct3 = abs(env3.B - (B - C * nacc3 + rev3))
            bk3   = env3._check_bankrupt()

            revs2.append(rev2); revs3.append(rev3)
            if acct2 > 1e-6 or acct3 > 1e-6:
                acct_ok_all = False
            bk_count += int(bk2) + int(bk3)

        comp = max(float(np.mean(revs2)), float(np.mean(revs3)))
        out[k] = {
            "v2_obs":    round(float(np.mean(revs2)), 4),
            "v3_obs":    round(float(np.mean(revs3)), 4),
            "composite": round(comp, 4),
            "acct_ok":   acct_ok_all,
            "bankrupts": bk_count,
        }
        winner = "v2" if float(np.mean(revs2)) >= float(np.mean(revs3)) else "v3"
        print(f"  k={k:2d}  v2={out[k]['v2_obs']:>7.2f}  v3={out[k]['v3_obs']:>7.2f}"
              f"  comp={comp:>7.2f} [{winner}]  acct={'OK' if acct_ok_all else 'FAIL'}  bk={bk_count}",
              flush=True)
    return out


# ── run ───────────────────────────────────────────────────────────────────────
all_results = {}
for net, graph in GRAPHS.items():
    print(f"\n=== {net} (n={graph.number_of_nodes()}) ===", flush=True)
    t0 = time.time()
    all_results[net] = eval_net(net, graph, K_ALL)
    print(f"  [{time.time()-t0:.0f}s]", flush=True)

# ── save ─────────────────────────────────────────────────────────────────────
os.makedirs("results/logs", exist_ok=True)
record = {"seeds": SEEDS, "n_sims": N_SIMS, "calib_budget": "5x5=25k_per_1k_nodes",
          "results": all_results}
with open(OUT, "w") as f:
    json.dump(record, f, indent=2)
print(f"\nSaved: {OUT}")

# ── print 5x6 table ───────────────────────────────────────────────────────────
print("\n=== Full 5x6 table: Cal-DP obs composite, seeds [42,123,7] ===")
header = f"{'network':12s}" + "".join(f"  k={k:2d}" for k in K_ALL)
print(header)
print("-" * len(header))
for net in GRAPHS:
    row = f"{net:12s}"
    for k in K_ALL:
        row += f"  {all_results[net][k]['composite']:6.1f}"
    print(row)

# ── accounting summary ────────────────────────────────────────────────────────
print("\nAccounting identity (correct: B_T = B_0 - c*n_acc + R):")
for net in GRAPHS:
    fails = [k for k in K_ALL if not all_results[net][k]["acct_ok"]]
    bks   = sum(all_results[net][k]["bankrupts"] for k in K_ALL)
    print(f"  {net:12s}: {'ALL OK' if not fails else f'FAIL at k={fails}'}  total_bankrupts={bks}")

# ── accounting fix note ───────────────────────────────────────────────────────
print("\nFix note: the env stores accounting_err as abs(env.B - (B - c*env.t + rev))")
print("  env.t counts ALL offers (accepted+rejected); identity uses n_accepted.")
print("  This script uses n_acc from _execute_v2/_v3 for the correct check.")
