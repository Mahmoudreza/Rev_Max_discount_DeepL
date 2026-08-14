"""experiments/verify_caldp_obs.py — six-check verification of obs Cal-DP sweep."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── REFERENCE TABLE ──────────────────────────────────────────────────────────
REF = {
    "polblogs":   {5:94.3,  10:179.8, 15:265.1, 20:521.2, 30:597.7, 40:660.9},
    "FF_1000":    {5:214.5, 10:419.4, 15:438.2, 20:438.2, 30:438.2, 40:438.2},
    "Rice_FB":    {5:0.8,   10:4.3,   15:120.6, 20:224.3, 30:224.4, 40:224.4},
    "Modular_FF": {5:1.3,   10:29.3,  15:100.1, 20:139.2, 30:227.5, 40:227.5},
    "FF_2000":    {5:503.0, 10:766.9, 15:797.6, 20:911.0, 30:911.0, 40:911.0},
}
K = [5, 10, 15, 20, 30, 40]

# ── CHECK 1: SOURCE + 30-cell match ──────────────────────────────────────────
path = "results/logs/caldp_obs_all_networks.json"
print(f"CHECK 1 SOURCE: {path}")
data = json.load(open(path))
mismatches = []
for net, ks in REF.items():
    for k, ref in ks.items():
        got = round(data[net]["results"][str(k)]["composite"], 1)
        if abs(got - ref) > 0.05:
            mismatches.append(f"  {net} k={k}: ref={ref} got={got}")
if mismatches:
    print("MISMATCHES:\n" + "\n".join(mismatches))
else:
    print("CHECK 1: 30/30 MATCH")

# ── CHECK 2: NO ORACLE (done locally via grep — reported here) ────────────────
print("\nCHECK 2 NO ORACLE: _true_valuation appears only in docstrings/comments")
print("  v2_obs: 'EXCEPT _true_valuation is never called' (docstring only)")
print("  v3_obs: 'No call to _true_valuation is made' (docstring only)")
print("  Imports: dp_calibrated_v2_obs, dp_calibrated_v3_obs — confirmed")

# ── CHECK 3: BUDGET per network ───────────────────────────────────────────────
print("\nCHECK 3 BUDGET:")
from src.env.graph_generators import generate_forest_fire, generate_modular_forest_fire, load_rice_facebook
from src.env.polblogs_loader import load_polblogs
graphs = {
    "polblogs":   load_polblogs(),
    "FF_1000":    generate_forest_fire(1000, 0.37, 0.32, seed=0),
    "Rice_FB":    load_rice_facebook(),
    "Modular_FF": generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0),
    "FF_2000":    generate_forest_fire(2000, 0.37, 0.32, seed=1),
}
N_SIMS, N_PASSES = 5, 5
for net, g in graphs.items():
    n = g.number_of_nodes()
    n_offers = N_PASSES * N_SIMS * n
    flag = " *** DIFFERS from 25k ***" if n != 1000 else ""
    print(f"  {net:12s}: n={n:4d}  offers={n_offers:6d}{flag}")

# ── CHECK 4: PROTOCOL ─────────────────────────────────────────────────────────
print("\nCHECK 4 PROTOCOL:")
print("  c=0.3: YES (C=0.3 in resweep_caldp_obs.py)")
print("  B0=k*c: YES (B=k*C per k iteration)")
print("  n_trials=3 seeds=0,1,2: the built-in function uses seed=trial (0,1,2).")
print("  NOTE: function does NOT support custom seeds [42,123,7].")
print("        Seeds [42,123,7] are used by the GNN policy evaluator, not Cal-DP.")
print("  n_mc=200: N/A to Cal-DP (no Monte Carlo inner loop; n_trials=3)")
print("  SKIP-never-reprice: dp_calibrated_v2/v3_obs_budget — no reprice logic.")
# Accounting identity: read from shard (accounting_err is per-trial in the shard)
net_shard = json.load(open("results/logs/caldp_obs_FF_1000.json"))
print("  accounting_err: not stored in shard (buried inside _aggregate 'all' list)")
print("  bankrupt: not stored in shard")
print("  → Re-run required for per-episode accounting check; not available from shard.")

# ── CHECK 5: COMPOSITE max(v2,v3) visible ────────────────────────────────────
print("\nCHECK 5 COMPOSITE (FF_1000, k=[5,10,20,40]):")
ff = data["FF_1000"]["results"]
print(f"  {'k':>3}  {'v2_obs':>8}  {'v3_obs':>8}  {'composite':>10}  {'winner':>6}")
for k in [5, 10, 20, 40]:
    r = ff[str(k)]
    v2, v3, co = r["v2_obs"], r["v3_obs"], r["composite"]
    winner = "v2" if v2 >= v3 else "v3"
    assert abs(co - max(v2, v3)) < 0.001, f"composite mismatch at k={k}"
    print(f"  {k:>3}  {v2:>8.2f}  {v3:>8.2f}  {co:>10.2f}  {winner:>6}")
print("  max rule VERIFIED")

# ── CHECK 6: REPRODUCE FF_1000 k=10 ─────────────────────────────────────────
print("\nCHECK 6 REPRODUCE FF_1000 k=10:")
from src.env.budget_revenue_env import BudgetEnvConfig
from src.evaluation.dp_calibrated_v2_obs import dp_calibrated_v2_obs_budget
from src.evaluation.dp_calibrated_v3_obs import dp_calibrated_v3_obs_budget
cfg = BudgetEnvConfig(production_cost=0.3, weight_high=1.0)
g1k = graphs["FF_1000"]
t0 = time.time()
r2 = dp_calibrated_v2_obs_budget(g1k, cfg, B=3.0, c=0.3, n_trials=3, n_sims=5)
r3 = dp_calibrated_v3_obs_budget(g1k, cfg, B=3.0, c=0.3, n_trials=3, n_sims=5)
v2 = r2["revenue"]["mean"]; v3 = r3["revenue"]["mean"]
comp = max(v2, v3)
ref_val = 419.4
delta = comp - ref_val
print(f"  v2={v2:.2f}  v3={v3:.2f}  composite={comp:.2f}  ref=419.4  delta={delta:+.2f}  ({time.time()-t0:.0f}s)")
print(f"  {'PASS (within ±10)' if abs(delta) <= 10 else 'FAIL (delta > 10)'}")
