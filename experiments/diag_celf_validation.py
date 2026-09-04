#!/usr/bin/env python3
"""Validate CELF gives identical seeds to naive on Modular_FF, 30 trials."""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from src.env.budget_revenue_env import BudgetEnvConfig
from src.env.graph_generators import generate_modular_forest_fire
from src.evaluation.budget_baselines import _make_env
from src.evaluation.baselines import _greedy_seed_selection
from src.evaluation.ie_budget import _greedy_seed_selection_celf, IE_K_SEEDS

G = generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
W_HIGH = 2.0; C = 0.3; N_TRIALS = 30

match = 0; t_naive = 0.0; t_celf = 0.0
for trial in range(N_TRIALS):
    # Naive
    env_n = _make_env(G, B=float("inf"), c=C, seed=trial, weight_high=W_HIGH); env_n.reset()
    t0 = time.time()
    s_naive = set(_greedy_seed_selection(G, env_n, IE_K_SEEDS))
    t_naive += time.time() - t0
    # CELF
    env_c = _make_env(G, B=float("inf"), c=C, seed=trial, weight_high=W_HIGH); env_c.reset()
    t0 = time.time()
    s_celf = set(_greedy_seed_selection_celf(G, env_c, IE_K_SEEDS))
    t_celf += time.time() - t0
    if s_naive == s_celf:
        match += 1
    else:
        print(f"  trial={trial}: DIFF naive={sorted(s_naive)[:5]} celf={sorted(s_celf)[:5]}")

print(f"Match: {match}/{N_TRIALS}  naive={t_naive:.2f}s  celf={t_celf:.2f}s  "
      f"speedup={t_naive/max(t_celf,0.001):.1f}x")
