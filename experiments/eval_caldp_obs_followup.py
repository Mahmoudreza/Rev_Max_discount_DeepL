"""
experiments/eval_caldp_obs_followup.py
Three follow-up checks for the obs Cal-DP sweep.

F1: Reseed evaluation to [42,123,7] — all 5 networks, k=[5,10,20,40].
F2: Accounting identity (bankrupt count + max accounting_err) per network.
F3: Rice_FB (n_sims=12, ~26.6k) and Modular_FF (n_sims=10, ~25k) rescaled.
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
TIERS  = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA  = 0.05

GRAPHS = {
    "polblogs":   load_polblogs(),
    "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook(),
    "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
}
cfg = BudgetEnvConfig(production_cost=C, weight_high=1.0)

# OLD values (seeds 0,1,2, n_sims=5) for comparison
OLD = {
    "polblogs":   {5:94.3,  10:179.8, 20:521.2, 40:660.9},
    "FF_1000":    {5:214.5, 10:419.4, 20:438.2, 40:438.2},
    "Rice_FB":    {5:0.8,   10:4.3,   20:224.3, 40:224.4},
    "Modular_FF": {5:1.3,   10:29.3,  20:139.2, 40:227.5},
    "FF_2000":    {5:503.0, 10:766.9, 20:911.0, 40:911.0},
}


def eval_net_custom(net, graph, k_list, n_sims=5):
    """Run both v2+v3 with seeds [42,123,7]. Return per-k composite + accounting."""
    n = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    # --- calibrate (reuses cache if n_sims matches) ---
    V2, A2, P2, cb2, ib2 = calibrate_v2_obs_table(
        graph, cfg, n_sims=n_sims, seed=0)
    V3, A3, T3, cb3, sb3 = calibrate_v3_obs_table(
        graph, cfg, n_sims=n_sims, seed=0)
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
        revs2, revs3, acct_errs, bankrupts = [], [], [], []
        for seed in SEEDS:
            # v2
            env = _make_env(graph, B=B, c=C, seed=seed, weight_high=1.0)
            env.reset()
            rev2, _, _ = _execute_v2(env=env, ordering=ordering, plan=plan2,
                                     V=V2, A=A2, class_boundaries=cb2,
                                     infl_boundaries=ib2, c=C,
                                     class_of_pos=cpos,
                                     dp_table=[[0.0]*(n+1) for _ in range(b_steps+1)],
                                     b_steps=b_steps, delta=DELTA, tiers=TIERS)
            acct2 = abs(env.B - (B - C * env.t + rev2))
            bk2 = env._check_bankrupt()

            # v3
            env3 = _make_env(graph, B=B, c=C, seed=seed, weight_high=1.0)
            env3.reset()
            rev3, _, _, _ = _execute_v3(env=env3, ordering=ordering,
                                         dp3=dp3, tier3=tier3, V3=V3, A3=A3,
                                         class_boundaries=cb3, sb_size=sb3, c=C,
                                         class_of_pos=cpos, n_s_buckets=_N_S_BUCKETS,
                                         b_steps=b_steps, delta=DELTA, tiers=TIERS,
                                         log_steps=10)
            acct3 = abs(env3.B - (B - C * env3.t + rev3))
            bk3 = env3._check_bankrupt()

            revs2.append(rev2); revs3.append(rev3)
            acct_errs.append(max(acct2, acct3))
            bankrupts.append(int(bk2) + int(bk3))

        comp = max(np.mean(revs2), np.mean(revs3))
        out[k] = {
            "v2": round(float(np.mean(revs2)), 2),
            "v3": round(float(np.mean(revs3)), 2),
            "comp": round(comp, 2),
            "max_acct_err": round(max(acct_errs), 6),
            "bankrupts": sum(bankrupts),
        }
    return out


# ── F1 + F2: all networks, seeds [42,123,7], n_sims=5 ────────────────────────
print("=== F1+F2: evaluation seeds [42,123,7], n_sims=5 ===")
print(f"{'net':12s}  {'k':>3}  {'old':>7}  {'new':>7}  {'delta':>7}  acct_err  bk")
F1_results = {}
K4 = [5, 10, 20, 40]
for net, graph in GRAPHS.items():
    res = eval_net_custom(net, graph, K4, n_sims=5)
    F1_results[net] = res
    for k in K4:
        old = OLD[net][k]
        new = res[k]["comp"]
        d = new - old
        ae = res[k]["max_acct_err"]
        bk = res[k]["bankrupts"]
        ae_str = f"{ae:.2e}"
        ok = "OK" if ae < 1e-3 else "VIOLATED"
        print(f"  {net:12s}  {k:>3}  {old:>7.1f}  {new:>7.1f}  {d:>+7.1f}  {ae_str}({ok})  bk={bk}")

# ── F3: Rice_FB scaled to ~26.6k, Modular_FF to 25k ─────────────────────────
print("\n=== F3: Rice_FB n_sims=12 (~26.6k), Modular_FF n_sims=10 (~25k) ===")
print(f"{'net':12s}  {'k':>3}  {'n_sims5':>8}  {'scaled':>8}  {'delta':>7}  acct_err")
F3_NETS = {"Rice_FB": (GRAPHS["Rice_FB"], 12), "Modular_FF": (GRAPHS["Modular_FF"], 10)}
for net, (graph, ns) in F3_NETS.items():
    res_scaled = eval_net_custom(net, graph, K4, n_sims=ns)
    for k in K4:
        old5  = F1_results[net][k]["comp"]   # seeds [42,123,7] n_sims=5
        new_s = res_scaled[k]["comp"]
        d = new_s - old5
        ae = res_scaled[k]["max_acct_err"]
        ae_str = f"{ae:.2e}"
        ok = "OK" if ae < 1e-3 else "VIOLATED"
        n = graph.number_of_nodes()
        actual_offers = 5 * ns * n
        print(f"  {net:12s}  {k:>3}  {old5:>8.1f}  {new_s:>8.1f}  {d:>+7.1f}  {ae_str}({ok})  (offers={actual_offers})")
