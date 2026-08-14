"""src/training/mixed_expert_trajectories.py — Mixed-expert trajectory generation.

Generates per-step (node_idx, discount, accepted, price, B_after) trajectories
from the REGIME-APPROPRIATE champion method for a given k:

  k <=  5 : DP-Calibrated-v3 executor
  6 <= k <= 15 : DP-Calibrated-v2 executor
  k >= 16 : greedy_discount_budget

Each executor is run via a class-level monkeypatch on BudgetRevenueEnv.step
(with n_trials=1 to capture exactly one episode), so the executors are NOT
modified.  The recorder is unpatched in a finally block.

Trajectory cache: results/logs/expert_traj_cache/{graph_hash}_k{k}_s{seed}.pkl
  Avoids re-running calibration on every Phase-1 epoch.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import threading
from typing import List, Tuple

import numpy as np
import networkx as nx

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
import src.env.budget_revenue_env as _bre_mod

# ── Thread-local lock so monkeypatching is safe in single-threaded training ──
_PATCH_LOCK = threading.Lock()

CACHE_DIR = "results/logs/expert_traj_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _graph_hash(graph: nx.Graph) -> str:
    """Stable 8-char hash from graph topology (edge count + node count + degree seq)."""
    deg_seq = sorted(dict(graph.degree()).values())
    sig = f"{graph.number_of_nodes()}_{graph.number_of_edges()}_{deg_seq[:20]}"
    return hashlib.md5(sig.encode()).hexdigest()[:8]


def _cache_path(graph: nx.Graph, k: int, seed: int) -> str:
    return os.path.join(CACHE_DIR, f"{_graph_hash(graph)}_k{k}_s{seed}.pkl")


# ── Core: run executor and record trajectory via class-level monkeypatch ──────

def _run_and_record(executor_fn, **kwargs) -> List[dict]:
    """Call executor_fn(**kwargs) and capture every env.step call.

    Uses a class-level monkeypatch so we don't need internal env access.
    Thread-unsafe by design — training is single-threaded.

    Args:
        executor_fn: Callable that internally creates/uses BudgetRevenueEnv.
        **kwargs:    Forwarded to executor_fn.

    Returns:
        trajectory: List of step dicts with keys
            node_idx, discount, accepted, price, B_after.
    """
    trajectory: List[dict] = []
    orig_step = _bre_mod.BudgetRevenueEnv.step

    def _patched(self: BudgetRevenueEnv, node_idx: int, discount: float):
        obs, reward, done, info = orig_step(self, node_idx, discount)
        trajectory.append({
            "node_idx":  int(node_idx),
            "discount":  float(discount),
            "accepted":  bool(info["accepted"]),
            "price":     float(info["offered_price"]),
            "B_after":   float(self.B),
        })
        return obs, reward, done, info

    with _PATCH_LOCK:
        _bre_mod.BudgetRevenueEnv.step = _patched
        try:
            executor_fn(**kwargs)
        finally:
            _bre_mod.BudgetRevenueEnv.step = orig_step

    return trajectory


# ── Executor wrappers ─────────────────────────────────────────────────────────

def _run_greedy(graph, B, c, seed):
    from src.evaluation.budget_baselines import greedy_discount_budget
    greedy_discount_budget(graph, B=B, c=c, n_trials=1)


def _run_dp_v2(graph, B, c, seed):
    from src.evaluation.dp_calibrated_v2 import dp_calibrated_v2_budget
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=seed)
    dp_calibrated_v2_budget(graph, cfg=cfg, B=B, c=c, n_trials=1)


def _run_dp_v3(graph, B, c, seed):
    from src.evaluation.dp_calibrated_v3 import dp_calibrated_v3_budget
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=seed)
    dp_calibrated_v3_budget(graph, cfg=cfg, B=B, c=c, n_trials=1)


# ── Public API ────────────────────────────────────────────────────────────────

def generate_budget_expert_trajectory(
    graph: nx.Graph,
    k: int,
    c: float = 0.3,
    seed: int = 0,
    force_rebuild: bool = False,
) -> List[dict]:
    """Return a per-step trajectory from the champion method for this k.

    Champion routing:
      k <=  5  → DP-Calibrated-v3
      6..15    → DP-Calibrated-v2
      k >= 16  → Greedy+Budget

    Each step dict:  node_idx, discount, accepted, price, B_after.
    Skip steps (budget-exceeded) are NOT included (executor doesn't call env.step).

    Results are cached to disk; set force_rebuild=True to regenerate.

    Args:
        graph:         NetworkX graph.
        k:             Number of seed buyers (determines B = k*c and champion).
        c:             Production cost per item.
        seed:          RNG seed for the episode.
        force_rebuild: Ignore cache and regenerate.

    Returns:
        List of per-step dicts (may be shorter than k if bankrupt early).
    """
    cp = _cache_path(graph, k, seed)
    if not force_rebuild and os.path.exists(cp):
        with open(cp, "rb") as f:
            return pickle.load(f)

    B = k * c

    if k <= 5:
        traj = _run_and_record(_run_dp_v3, graph=graph, B=B, c=c, seed=seed)
    elif k <= 15:
        traj = _run_and_record(_run_dp_v2, graph=graph, B=B, c=c, seed=seed)
    else:
        traj = _run_and_record(_run_greedy, graph=graph, B=B, c=c, seed=seed)

    with open(cp, "wb") as f:
        pickle.dump(traj, f, protocol=4)

    return traj


# ── Cache builder (called once before training) ───────────────────────────────

def build_trajectory_cache(
    graphs: List[nx.Graph],
    k_list: Tuple[int, ...] = (1, 3, 5, 10, 15, 25, 40),
    n_seeds: int = 10,
    c: float = 0.3,
    verbose: bool = True,
) -> dict:
    """Pre-generate expert trajectories for all (graph, k, seed) combinations.

    Args:
        graphs:   List of training graphs.
        k_list:   k values to cache.
        n_seeds:  Seeds 0 .. n_seeds-1 per (graph, k).
        c:        Production cost.
        verbose:  Print progress and revenue sanity table.

    Returns:
        summary: {k: avg_revenue_across_graphs_seed0}
    """
    summary: dict = {}
    total = len(graphs) * len(k_list) * n_seeds
    done = 0

    for k in k_list:
        k_revs = []
        for g_idx, graph in enumerate(graphs):
            for seed in range(n_seeds):
                traj = generate_budget_expert_trajectory(graph, k, c=c, seed=seed)
                if seed == 0:
                    rev = sum(s["price"] for s in traj if s["accepted"])
                    k_revs.append(rev)
                done += 1
                if verbose and done % 10 == 0:
                    print(f"  cache {done}/{total}  k={k} g={g_idx} s={seed}")
        summary[k] = float(np.mean(k_revs)) if k_revs else 0.0

    if verbose:
        print("\n── Expert trajectory sanity table (seed=0, mean over graphs) ──")
        print(f"{'k':>5} | {'champion':>18} | {'avg_rev (seed=0)':>16}")
        print("-" * 46)
        for k in k_list:
            champ = ("DP-v3" if k <= 5 else "DP-v2" if k <= 15 else "Greedy+B")
            print(f"{k:>5} | {champ:>18} | {summary[k]:>16.3f}")

    return summary
