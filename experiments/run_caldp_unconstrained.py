"""
experiments/run_caldp_unconstrained.py
Cal-DP v2_obs on the UNCONSTRAINED revenue problem.

Approach: BudgetRevenueEnv with c=0, B0=0. Wallet never decreases
(no production cost), always affordable, never bankrupt. The DP table
degenerates to b_steps=1 (single dummy wallet state). All 5x5=25k
calibration tables are reused from cache.

This makes the DP degenerate to per-step calibrated tier selection:
for each buyer (degree-descending), pick the tier that maximises
expected immediate revenue given the observed influence bucket.

Protocol: seeds [0,1,2,3,4]; report 5-seed means.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs
from src.env.graph_generators import (
    generate_forest_fire, generate_modular_forest_fire, load_rice_facebook,
)
from src.evaluation.budget_baselines import _make_env
from src.evaluation.dp_calibrated import _deg_class
from src.evaluation.dp_calibrated_v2 import _plan_dp_v2, _execute_v2
from src.evaluation.dp_calibrated_v2_obs import calibrate_v2_obs_table

NETWORKS_ALL = ["polblogs", "FF_1000", "Rice_FB", "Modular_FF", "FF_2000"]
SEEDS        = [0, 1, 2, 3, 4]
TIERS        = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA        = 0.05
N_SIMS       = 5   # frozen 5x5 budget (reuses cache)
C_UNCONSTRAINED = 0.0
B_UNCONSTRAINED = 0.0

# cfg for calibration — cache key is graph-hash + n_sims only, not c
CFG_CALIB = BudgetEnvConfig(production_cost=0.3, weight_high=1.0)

REFS = {   # frozen reference values (5-seed means)
    "FF_1000":   dict(greedy=417.7, ie=146.2, gnn=448.6),
    "FF_2000":   dict(greedy=839.2, ie=191.8, gnn=915.0),
    "Modular_FF":dict(greedy=356.6, ie=125.8, gnn=414.4),
    "Rice_FB":   dict(greedy=159.7, ie=114.9, gnn=214.1),
    "polblogs":  dict(greedy=530.4, ie=481.7, gnn=374.2),
}

GRAPH_LOADERS = {
    "polblogs":   load_polblogs,
    "FF_1000":    lambda: generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook,
    "Modular_FF": lambda: generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    lambda: generate_forest_fire(2000, 0.37, 0.32, seed=1),
}


def run_net(net):
    graph   = GRAPH_LOADERS[net]()
    n       = graph.number_of_nodes()
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
    all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

    V2, A2, P2, cb2, ib2 = calibrate_v2_obs_table(
        graph, CFG_CALIB, n_sims=N_SIMS, seed=0)
    cpos = np.array([_deg_class(int(all_deg[i]), cb2) for i in range(n)], dtype=np.int32)

    # c=0, B=0 → b_steps=1, single dummy wallet state
    b_steps = max(1, int(B_UNCONSTRAINED / DELTA) + 1)  # = 1
    plan2   = _plan_dp_v2(n_total=n, V=V2, A=A2, P=P2, class_of_pos=cpos,
                          B=B_UNCONSTRAINED, c=C_UNCONSTRAINED,
                          tiers=TIERS, delta=DELTA)

    revenues, tier_counts = [], {t: 0 for t in TIERS}

    for seed in SEEDS:
        env = _make_env(graph, B=B_UNCONSTRAINED, c=C_UNCONSTRAINED,
                        seed=seed, weight_high=1.0)
        env.reset()

        # track tier usage via monkey-patch
        _orig = env.step
        def _step(node_idx, discount, _o=_orig):
            tier_counts[discount] = tier_counts.get(discount, 0) + 1
            return _o(node_idx, discount)
        env.step = _step

        rev, nacc, _ = _execute_v2(
            env=env, ordering=ordering, plan=plan2,
            V=V2, A=A2, class_boundaries=cb2,
            infl_boundaries=ib2, c=C_UNCONSTRAINED,
            class_of_pos=cpos,
            dp_table=[[0.0]*(n+1) for _ in range(b_steps+1)],
            b_steps=b_steps, delta=DELTA, tiers=TIERS)
        revenues.append(rev)

    mean_rev = float(np.mean(revenues))
    print(f"  {net:12s}: mean={mean_rev:.1f}  per_seed={[round(r,1) for r in revenues]}",
          flush=True)
    # Normalise tier counts across seeds
    total = sum(tier_counts.values())
    tier_pct = {str(t): round(100.0 * c / total, 1) for t, c in tier_counts.items()}
    return {"mean_rev": round(mean_rev, 2), "per_seed": [round(r, 2) for r in revenues],
            "tier_pct": tier_pct, "b_steps": b_steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--networks", nargs="+", default=NETWORKS_ALL)
    args = ap.parse_args()

    results = {}
    for net in args.networks:
        t0 = time.time()
        results[net] = run_net(net)
        print(f"    [{time.time()-t0:.0f}s]", flush=True)

    os.makedirs("results/logs", exist_ok=True)
    with open("results/logs/caldp_unconstrained.json", "w") as f:
        json.dump(results, f, indent=2)

    # ── table ────────────────────────────────────────────────────────────────
    print(f"\n{'network':12s}  {'Greedy-Disc':>11}  {'IE-Strat':>8}  {'Rev-GNN':>8}  {'Cal-DP-Unc':>11}")
    print("-" * 60)
    beats, deltas = 0, []
    for net in ["FF_1000","FF_2000","Modular_FF","Rice_FB","polblogs"]:
        if net not in results:
            continue
        r   = results[net]
        ref = REFS[net]
        v   = r["mean_rev"]
        d   = v - ref["greedy"]
        if d > 0:
            beats += 1
            deltas.append(d)
        flag = "+" if d > 0 else ""
        print(f"  {net:12s}  {ref['greedy']:>11.1f}  {ref['ie']:>8.1f}  {ref['gnn']:>8.1f}"
              f"  {v:>9.1f}{flag}")

    computed = sum(1 for net in NETWORKS_ALL if net in results)
    if computed == len(NETWORKS_ALL):
        print(f"\na) Cal-DP(unconstrained) beats Greedy-Discount on {beats}/5 networks"
              f"; avg delta = {np.mean(deltas) if deltas else 0:+.1f}")
        # tier distribution
        tc = {t: 0.0 for t in TIERS}
        for net, r in results.items():
            for t in TIERS:
                tc[t] += r["tier_pct"].get(str(t), 0.0)
        n_nets = len(results)
        print("b) Tier distribution (avg % across networks):", end="")
        for t in TIERS:
            print(f"  tier={t}: {tc[t]/n_nets:.1f}%", end="")
        print()


if __name__ == "__main__":
    main()
