#!/usr/bin/env python3
"""Validate CELF N_MC=20 vs N_MC=200 reference on Modular_FF, 30 trials.

Item 3: Rank candidates with N_MC=20; validate ≥27/30 seed overlap per trial.
"27 of 30 seeds" = at least 27 out of 30 individual seed nodes in common
between the N_MC=20 and N_MC=200 orderings for each trial.
Items 1+2 (exact CELF) are pure caching — no result change, not validated here.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.env.graph_generators import generate_modular_forest_fire
from src.evaluation.budget_baselines import _make_env
from src.evaluation.ie_budget import _greedy_seed_selection_celf, IE_K_SEEDS

G = generate_modular_forest_fire([250,250], 0.37, 0.32, 0.05, seed=0)
W_HIGH = 2.0; C = 0.3; N_TRIALS = 30

trials_pass = 0; min_overlap = IE_K_SEEDS; t_ref = 0.0; t_mc20 = 0.0
for trial in range(N_TRIALS):
    # Reference: N_MC=200
    env_r = _make_env(G, B=float("inf"), c=C, seed=trial, weight_high=W_HIGH)
    env_r.reset()
    t0 = time.time()
    s_ref = set(_greedy_seed_selection_celf(G, env_r, IE_K_SEEDS, n_mc_rank=200))
    t_ref += time.time() - t0
    # N_MC=20
    env_m = _make_env(G, B=float("inf"), c=C, seed=trial, weight_high=W_HIGH)
    env_m.reset()
    t0 = time.time()
    s_mc20 = set(_greedy_seed_selection_celf(G, env_m, IE_K_SEEDS, n_mc_rank=20))
    t_mc20 += time.time() - t0

    overlap = len(s_ref & s_mc20)  # seeds in common
    min_overlap = min(min_overlap, overlap)
    if overlap >= 27:
        trials_pass += 1
    else:
        print(f"  trial={trial}: overlap={overlap}/30 (FAIL)")

print(f"Trials with overlap>=27: {trials_pass}/{N_TRIALS}  "
      f"min_overlap={min_overlap}/30  "
      f"ref(200)={t_ref:.1f}s  mc20={t_mc20:.1f}s  "
      f"speedup={t_ref/max(t_mc20,0.001):.1f}x")
assert trials_pass == N_TRIALS, f"FAIL: {N_TRIALS - trials_pass} trials below 27/30 overlap"
print("PASS")
