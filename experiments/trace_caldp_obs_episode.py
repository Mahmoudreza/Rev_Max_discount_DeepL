"""
experiments/trace_caldp_obs_episode.py
One-episode accounting trace: FF_1000, k=10, seed=42, Cal-DP v3_obs.
Prints B_0, c, B_T, |S_T|, R, identity LHS/RHS/diff.
Prints first 10 accepted steps: step, node, price, wallet_before, wallet_after.
"""
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.graph_generators import generate_forest_fire
from src.evaluation.budget_baselines import _make_env
from src.evaluation.dp_calibrated import _deg_class
from src.evaluation.dp_calibrated_v3 import _plan_dp_v3, _execute_v3, _N_S_BUCKETS
from src.evaluation.dp_calibrated_v3_obs import calibrate_v3_obs_table

C     = 0.3
K     = 10
B0    = K * C          # 3.0
SEED  = 42
TIERS = (1.0, 0.8, 0.5, 0.2, 0.0)
DELTA = 0.05

cfg   = BudgetEnvConfig(production_cost=C, weight_high=1.0)
graph = generate_forest_fire(1000, 0.37, 0.32, seed=0)
n     = graph.number_of_nodes()
ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)
all_deg  = np.array([graph.degree(v) for v in ordering], dtype=float)

# calibrate (reuses cache)
V3, A3, T3, cb3, sb3 = calibrate_v3_obs_table(graph, cfg, n_sims=5, seed=0)
cpos = np.array([_deg_class(int(all_deg[i]), cb3) for i in range(n)], dtype=np.int32)

b_steps = max(1, int(B0 / DELTA) + 1)
dp3, tier3 = _plan_dp_v3(n_total=n, V3=V3, A3=A3, T=T3,
                          class_of_pos=cpos, B=B0, c=C, sb_size=sb3,
                          n_s_buckets=_N_S_BUCKETS, tiers=TIERS, delta=DELTA)

# ── instrument env.step ──────────────────────────────────────────────────────
env = _make_env(graph, B=B0, c=C, seed=SEED, weight_high=1.0)
env.reset()

step_log = []   # list of (step_t, node_idx, price, w_before, w_after)
_orig_step = env.step

def _traced_step(node_idx, discount):
    w_before = env.B
    obs, rew, done, info = _orig_step(node_idx, discount)
    w_after = env.B
    if info["accepted"]:
        step_log.append((env.t, node_idx, info["offered_price"], w_before, w_after))
    return obs, rew, done, info

env.step = _traced_step

# ── run one episode ──────────────────────────────────────────────────────────
rev, n_acc, n_sub, _ = _execute_v3(
    env=env, ordering=ordering,
    dp3=dp3, tier3=tier3, V3=V3, A3=A3,
    class_boundaries=cb3, sb_size=sb3, c=C,
    class_of_pos=cpos, n_s_buckets=_N_S_BUCKETS,
    b_steps=b_steps, delta=DELTA, tiers=TIERS, log_steps=10,
)

# ── episode summary ──────────────────────────────────────────────────────────
B_T    = env.B
n_S_T  = n_acc
R      = rev
lhs    = B_T
rhs    = B0 - C * n_S_T + R
diff   = lhs - rhs

print(f"B_0 = {B0:.4f}")
print(f"c   = {C:.4f}")
print(f"B_T = {B_T:.6f}")
print(f"|S_T| (accepted) = {n_S_T}")
print(f"R (sum accepted prices) = {R:.6f}")
print(f"lhs = B_T            = {lhs:.6f}")
print(f"rhs = B_0 - c*|S_T| + R = {B0} - {C}*{n_S_T} + {R:.6f} = {rhs:.6f}")
print(f"diff = lhs - rhs = {diff:.2e}")
print(f"IDENTITY: {'OK' if abs(diff) < 1e-6 else 'VIOLATED'}")

# ── first 10 accepted steps ──────────────────────────────────────────────────
print(f"\n{'step':>5}  {'node':>5}  {'price':>8}  {'w_before':>10}  {'w_after':>10}  check")
for (t, node, price, wb, wa) in step_log[:10]:
    expected_wa = wb - C + price
    ok = "OK" if abs(wa - expected_wa) < 1e-9 else f"ERR(exp={expected_wa:.6f})"
    print(f"  {t:>3}  {node:>5}  {price:>8.4f}  {wb:>10.6f}  {wa:>10.6f}  {ok}")
