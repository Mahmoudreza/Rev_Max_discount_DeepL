#!/usr/bin/env python3
"""diag_whigh.py — Confirm W_HIGH reaches env.reset() by printing realized edge weights.

For polblogs seed=0: print mean/std/max of link weights under W_HIGH=1.0 and W_HIGH=2.0.
If identical → W_HIGH is not propagated to env.reset().
Also checks where RevenueEnv stores link weights after reset.

Usage:
  venv/bin/python3 -u experiments/diag_whigh.py
"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.env.polblogs_loader import load_polblogs

G = load_polblogs()
n = G.number_of_nodes()
print(f"polblogs n={n}  m={G.number_of_edges()}", flush=True)

for wh in [1.0, 2.0]:
    cfg = BudgetEnvConfig(budget_B=5*0.3, production_cost=0.3,
                          seed=0, weight_high=wh, n_mc_samples=200)
    env = BudgetRevenueEnv(G, cfg)
    obs = env.reset()

    # Inspect env for link weight storage
    # RevenueEnv should have self.link_weights or similar after reset
    w_arr = None
    for attr in ("link_weights", "_link_weights", "weights", "w", "_weights"):
        if hasattr(env, attr):
            w_arr = getattr(env, attr)
            print(f"  W_HIGH={wh}: found weights at env.{attr}  type={type(w_arr)}", flush=True)
            break
    if w_arr is None:
        # Try parent
        for attr in dir(env):
            if "weight" in attr.lower() and not attr.startswith("__"):
                v = getattr(env, attr)
                if hasattr(v, "__len__") and not callable(v):
                    print(f"  W_HIGH={wh}: env.{attr} len={len(v)}", flush=True)

    # Also try env.cfg or env.config
    print(f"  W_HIGH={wh}: cfg.weight_high={env.cfg.weight_high}", flush=True)

    # Try to get weights from env._link_weight_matrix or similar
    # Look for a numpy array attribute
    arrays = {}
    for attr in dir(env):
        if attr.startswith("_") and not attr.startswith("__"):
            try:
                v = getattr(env, attr)
                if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) > 1:
                    arrays[attr] = v
            except: pass
    for attr in dir(env):
        if not attr.startswith("_"):
            try:
                v = getattr(env, attr)
                if isinstance(v, np.ndarray) and v.ndim >= 1 and len(v) > 1:
                    arrays[attr] = v
            except: pass

    for attr, v in sorted(arrays.items()):
        if v.max() > 0:
            print(f"  W_HIGH={wh}: env.{attr}  shape={v.shape}  mean={v.mean():.4f}  std={v.std():.4f}  max={v.max():.4f}", flush=True)

    # Force: call _estimate_valuation on node 0 and print result
    nodes = list(G.nodes())
    v_hat_1 = env._estimate_valuation(nodes[0])
    v_hat_2 = env._estimate_valuation(nodes[0])
    print(f"  W_HIGH={wh}: _estimate_valuation(node0) call1={v_hat_1:.4f}  call2={v_hat_2:.4f}", flush=True)
    print()

# Also check what happens if W_HIGH is set but BudgetEnvConfig is created differently
print("=== Direct RevenueEnvConfig check ===")
from src.env.revenue_env import RevenueEnv, RevenueEnvConfig
for wh in [1.0, 2.0]:
    cfg2 = RevenueEnvConfig(weight_high=wh, seed=0)
    env2 = RevenueEnv(G, cfg2)
    env2.reset()
    # Find weights
    for attr in dir(env2):
        if "weight" in attr.lower() and not attr.startswith("__"):
            try:
                v = getattr(env2, attr)
                if isinstance(v, (list, np.ndarray)) and hasattr(v, "__len__") and len(v) > 1:
                    arr = np.array(v)
                    if arr.max() > 0:
                        print(f"  W_HIGH={wh}: env2.{attr}  mean={arr.mean():.4f}  max={arr.max():.4f}")
            except: pass
    v_hat_r = env2._estimate_valuation(nodes[0])
    print(f"  W_HIGH={wh}: RevenueEnv _estimate_valuation(node0)={v_hat_r:.4f}")
