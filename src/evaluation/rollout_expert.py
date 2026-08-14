"""src/evaluation/rollout_expert.py — Bertsekas rollout expert for budget-constrained revenue.

One-step lookahead + truncated rollout expert (Bertsekas 2020, §6.3).

CRITICAL CONSTRAINT — seller-observable information only
  * _estimate_valuation(node): seller's MC estimate (fresh independent samples;
    stored in clone's _est_val_cache).
  * get_current_influence(node): uses clone's _link_weights (freshly resampled,
    not copied from the real episode env).
  * NEVER: real_env._true_valuation(), real_env._link_weights, any field that
    encodes the ground-truth weight realisation of the live episode.

The clone env gets its own Uniform(0, w_high) weight draw at reset() — this is the
seller's hypothetical simulation of what might happen, not a privileged oracle.

Bertsekas guarantee: if the base policy is fixed (time-invariant), the rollout
policy is >= the base policy in expected revenue, with equality when the base is
already optimal.

Public API
----------
build_j_table(graph, cfg, max_budget, delta, tiers, force_rebuild)
    → np.ndarray (n_budget_bins+1, n+1)  — terminal value J[b_idx, k_pos]

rollout_expert_step(env, J_table, cfg)
    → (node_idx: int, discount: float)

generate_rollout_expert_trajectory(graph, cfg, k, c, seed)
    → List[dict]  (same format as mixed_expert_trajectories.py)
"""

from __future__ import annotations

import hashlib
import os
import pickle
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import networkx as nx

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.budget_baselines import _make_env, _aggregate
from src.evaluation.baselines import _rayleigh_price
from src.evaluation.dp_calibrated import calibrate_valuation_curves, _graph_hash

# ── Directories ────────────────────────────────────────────────────────────────
_ROLLOUT_CACHE_DIR = "results/logs/rollout_expert_cache"
_JTABLE_CACHE_DIR  = "results/logs"
os.makedirs(_ROLLOUT_CACHE_DIR, exist_ok=True)
os.makedirs(_JTABLE_CACHE_DIR,  exist_ok=True)


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class RolloutExpertConfig:
    """Hyper-parameters for rollout_expert_step.

    Args:
        c:                Production cost per offer (default 0.3).
        rollout_H:        Max rollout steps with base policy (default 20).
                          Bertsekas guarantee holds for any fixed H >= 1.
        n_mc_rollout:     MC samples for valuation estimates inside clone (default 30).
        discount_grid:    Discrete discount values tried per candidate node.
                          3 values × 8 candidates = 24 pairs (fast & sufficient).
        n_cand_degree:    Candidate nodes taken by degree rank (default 5).
        n_cand_val:       Candidate nodes taken by current MC est_val (default 3).
        j_delta:          Budget discretisation step for J-table (default 0.1).
        j_tiers:          Discount tiers used when building the J-table.
        b:                Rayleigh scale parameter (default 1.0).
        weight_high:      Upper bound of link-weight distribution (default 2.0).

    Performance note (n=200):
        The inner rollout uses degree-ordering (O(1) per step) not MC-greedy
        (O(n×n_mc) per step), so each candidate pair costs O(H × degree) instead
        of O(H × n × n_mc).  This reduces per-episode wall-clock from ~5 min to
        ~5-15 sec on n=200.  Bertsekas guarantee is preserved because the
        degree-ordering policy is a valid fixed policy.
    """
    c: float = 0.3
    rollout_H: int = 20
    n_mc_rollout: int = 30
    discount_grid: Tuple[float, ...] = (0.0, 0.5, 1.0)
    n_cand_degree: int = 5
    n_cand_val:    int = 3
    j_delta: float = 0.1
    j_tiers: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.0)
    b: float = 1.0
    weight_high: float = 2.0


# ── J-table: terminal value ────────────────────────────────────────────────────

def build_j_table(
    graph: nx.Graph,
    cfg: BudgetEnvConfig,
    max_budget: float,
    delta: float = 0.1,
    tiers: Tuple[float, ...] = (1.0, 0.8, 0.5, 0.2, 0.0),
    force_rebuild: bool = False,
) -> np.ndarray:
    """Build the terminal-value table J[b_idx, k_pos] via backward DP.

    J[b_idx, k_pos] = expected revenue achievable from graph position ``k_pos``
    (in degree-descending order) to the end of the episode, given a remaining
    budget of ``b_idx * delta``, under the calibrated tier policy.

    The table is computed once per (graph, max_budget, delta) triple and cached
    to ``results/logs/j_table_{hash}.npy``.

    Args:
        graph:         NetworkX graph.
        cfg:           BudgetEnvConfig — only production_cost and weight_high matter.
        max_budget:    Maximum budget to consider (e.g. initial B_0 = k * c).
        delta:         Budget discretisation step (default 0.1).
        tiers:         Discount fractions for J-table DP.
        force_rebuild: If True, ignore cached table.

    Returns:
        J: np.ndarray of shape (b_steps+1, n+1), dtype float64.
    """
    n       = graph.number_of_nodes()
    c       = cfg.production_cost
    b_steps = max(1, int(max_budget / delta) + 1)

    gh          = _graph_hash(graph)
    cache_name  = f"j_table_{gh}_B{max_budget:.1f}_d{delta:.3f}.npy"
    cache_path  = os.path.join(_JTABLE_CACHE_DIR, cache_name)

    if os.path.exists(cache_path) and not force_rebuild:
        return np.load(cache_path)

    # Load (or compute) calibrated valuation curves — always cached after 1st run.
    v_curve, class_of_position, _ = calibrate_valuation_curves(graph, cfg)

    tiers_list = list(tiers)

    # Backward DP.  J[b_idx, n] = 0 (no positions left → zero revenue).
    J = np.zeros((b_steps + 1, n + 1), dtype=np.float64)

    for k_pos in range(n - 1, -1, -1):
        cls     = int(class_of_position[k_pos])
        avg_val = float(v_curve[cls, k_pos])

        for b_idx in range(b_steps + 1):
            b_curr = b_idx * delta
            best   = J[b_idx, k_pos + 1]          # option: skip this position

            for t_disc in tiers_list:
                price = avg_val * (1.0 - t_disc)
                if b_curr - c + price < -1e-9:
                    continue                        # unaffordable at this budget
                new_b_raw = b_curr - c + price
                new_b_idx = min(int(new_b_raw / delta), b_steps)
                val = price + J[new_b_idx, k_pos + 1]
                if val > best:
                    best = val

            J[b_idx, k_pos] = best

    np.save(cache_path, J)
    return J


def _j_lookup(J: np.ndarray, budget: float, k_pos: int, delta: float) -> float:
    """Look up J[b_idx, k_pos] for a continuous budget value.

    Clamps both indices to valid table ranges.

    Args:
        J:      J-table from build_j_table.
        budget: Remaining budget (float).
        k_pos:  Current position in degree ordering (int).
        delta:  Budget discretisation step.

    Returns:
        Estimated future revenue from (budget, k_pos) onward.
    """
    n_b = J.shape[0] - 1
    n_k = J.shape[1] - 1
    b_idx  = min(int(max(0.0, budget) / delta), n_b)
    k_idx  = min(max(0, k_pos), n_k)
    return float(J[b_idx, k_idx])


# ── Environment state snapshot / clone ────────────────────────────────────────

def _snapshot_env_state(env: BudgetRevenueEnv) -> dict:
    """Capture the minimal seller-observable episode state.

    Does NOT copy link weights (those are ground truth, not observable).

    Args:
        env: Live BudgetRevenueEnv mid-episode.

    Returns:
        dict with keys: S, offered, B, t, total_revenue.
    """
    return {
        "S":             frozenset(env.S),
        "offered":       frozenset(env.offered),
        "B":             env.B,
        "t":             env.t,
        "total_revenue": env.total_revenue,
    }


def _make_clone(
    real_env: BudgetRevenueEnv,
    snapshot: dict,
    clone_seed: int,
    n_mc_rollout: int = 50,
) -> BudgetRevenueEnv:
    """Build a simulated env whose 'true' weights are fresh MC samples.

    The clone has the same graph TOPOLOGY as real_env but draws its own
    independent link-weight realisation from Uniform(0, w_high).  This
    strictly enforces the seller-observable constraint:
      - acceptance inside the clone is decided by fresh simulated weights,
      - NOT by the real episode's ground-truth w_ij.

    Args:
        real_env:      Live BudgetRevenueEnv (state source).
        snapshot:      Result of _snapshot_env_state(real_env).
        clone_seed:    RNG seed for the clone's fresh link-weight draw.
        n_mc_rollout:  MC samples for valuation estimates in clone (speed knob).

    Returns:
        Initialised BudgetRevenueEnv with state matching snapshot.
    """
    clone_cfg = BudgetEnvConfig(
        budget_B        = real_env.initial_budget,
        production_cost = real_env.production_cost,
        seed            = clone_seed,
        weight_high     = real_env.cfg.weight_high,
        n_mc_samples    = n_mc_rollout,          # fewer samples → faster rollout
        influence_model = real_env.cfg.influence_model,
        b               = real_env.cfg.b,
    )
    clone = BudgetRevenueEnv(real_env.graph, clone_cfg)
    clone.reset()                                # draws fresh link weights

    # Overwrite episode state with snapshot (graph structure already shared).
    clone.S             = set(snapshot["S"])
    clone.offered       = set(snapshot["offered"])
    clone.B             = snapshot["B"]
    clone.t             = snapshot["t"]
    clone.total_revenue = snapshot["total_revenue"]

    # Invalidate ALL caches: S changed, so all influence/valuation estimates
    # need to be recomputed against the clone's fresh weights.
    clone._influence_cache = {}
    clone._true_val_cache  = {}
    clone._est_val_cache   = {}

    return clone


# ── Greedy base policy (step-wise, operates on clone) ─────────────────────────

def _tier_price(infl: float, b: float = 1.0) -> float:
    """Babaei 2013 tier price from influence level (seller-observable via clone weights)."""
    if infl < 2.0 / 6.0:
        return 0.0
    elif infl < 4.0 / 6.0:
        return _rayleigh_price(2.0 / 6.0, b)   # ≈ 0.534
    else:
        return _rayleigh_price(4.0 / 6.0, b)   # ≈ 0.548


def _degree_base_step(
    clone: BudgetRevenueEnv,
    degree_order: List,
    c: float,
    b: float = 1.0,
) -> Tuple[Optional[int], float]:
    """One step of degree-ordered tier-pricing policy on a clone env.

    Scans ``degree_order`` (pre-sorted descending by degree, computed once
    per episode) for the next unvisited node.  Uses tier pricing based on
    the clone's current influence (O(degree) per node, not O(n × n_mc)).

    This replaces the MC-greedy scan (O(n × n_mc) per step) used in the
    original ``_greedy_base_step``.  Bertsekas guarantee is preserved because
    degree-ordering is a valid fixed base policy.

    Args:
        clone:        Simulated BudgetRevenueEnv.
        degree_order: List of nodes, sorted descending by degree (pre-computed).
        c:            Production cost.
        b:            Rayleigh scale.

    Returns:
        (node_idx, discount) or (None, 0.0) if no feasible move.
    """
    for node in degree_order:
        if node in clone.offered:
            continue

        infl  = clone.get_current_influence(node)   # O(degree(node))
        price = _tier_price(infl, b)

        # Budget check
        if clone.B - c + price < -1e-9:
            continue   # skip unaffordable node, try next

        ev_node  = clone._estimate_valuation(node)  # ONE MC call (not n×mc)
        discount = max(0.0, 1.0 - price / ev_node) if ev_node > 1e-9 else 1.0
        node_idx = clone.node_to_idx[node]
        return node_idx, discount

    return None, 0.0   # all nodes offered or unaffordable


def _run_rollout(
    clone: BudgetRevenueEnv,
    H: int,
    c: float,
    b: float = 1.0,
    degree_order: Optional[List] = None,
) -> Tuple[float, float, int]:
    """Roll forward at most H steps on clone using degree-ordered tier policy.

    Uses ``_degree_base_step`` (O(degree) per step) instead of
    ``_greedy_base_step`` (O(n × n_mc) per step).

    Args:
        clone:        Simulated BudgetRevenueEnv at start of rollout.
        H:            Maximum rollout steps.
        c:            Production cost.
        b:            Rayleigh scale.
        degree_order: Pre-sorted node list (computed by caller once per episode).
                      If None, computed locally (slower, for backward compat).

    Returns:
        (rollout_revenue, final_B, final_t)
    """
    if degree_order is None:
        degree_order = sorted(
            clone.nodes, key=lambda v: clone.graph.degree(v), reverse=True
        )

    rollout_rev = 0.0
    for _ in range(H):
        if clone._check_bankrupt() or len(clone.offered) >= clone.n:
            break
        node_idx, discount = _degree_base_step(clone, degree_order, c, b)
        if node_idx is None:
            break
        _, reward, done, _ = clone.step(node_idx, discount)
        rollout_rev += reward
        if done:
            break
    return rollout_rev, clone.B, clone.t


# ── Core: rollout_expert_step ─────────────────────────────────────────────────

def rollout_expert_step(
    env: BudgetRevenueEnv,
    J_table: np.ndarray,
    cfg: RolloutExpertConfig,
) -> Tuple[int, float]:
    """One expert decision via 1-step lookahead + truncated rollout + J-table.

    Candidate generation:
      - Top-``cfg.n_cand_degree`` unoffered nodes by graph degree.
      - Top-``cfg.n_cand_val`` unoffered nodes by current MC est_val.
      - Deduplicated; crossed with ``cfg.discount_grid`` → at most
        (n_cand_degree + n_cand_val) × len(discount_grid) ≤ 75 pairs.
      - Infeasible pairs (B - c + price < 0) dropped before rollout.

    Scoring (all on fresh clone, never reading real_env._link_weights):
      score = immediate_revenue + rollout_revenue + J_terminal

    Fallback: if all candidates are infeasible, select the unoffered node
      with the highest est_val and offer at discount=1.0 (free / zero cost).

    Args:
        env:     Live BudgetRevenueEnv mid-episode.
        J_table: From build_j_table(); shape (b_steps+1, n+1).
        cfg:     RolloutExpertConfig.

    Returns:
        (node_idx, discount) — best candidate according to rollout score.
    """
    c        = cfg.c
    b        = cfg.b
    H        = cfg.rollout_H
    n_mc     = cfg.n_mc_rollout
    delta    = cfg.j_delta
    n        = env.n

    # ── Collect available nodes ──────────────────────────────────────────────
    available = [node for node in env.nodes if node not in env.offered]
    if not available:
        # Edge case: no unoffered nodes (shouldn't normally happen).
        return 0, 0.0

    # ── Candidate set ────────────────────────────────────────────────────────
    # Top-k by degree
    by_degree = sorted(available, key=lambda v: env.graph.degree(v), reverse=True)
    cand_deg  = by_degree[:cfg.n_cand_degree]

    # Top-j by MC est_val (seller-observable valuation estimate)
    by_val   = sorted(available, key=lambda v: env._estimate_valuation(v), reverse=True)
    cand_val = by_val[:cfg.n_cand_val]

    # Deduplicate preserving order (degree-first)
    seen: set = set()
    candidates = []
    for node in cand_deg + cand_val:
        if node not in seen:
            candidates.append(node)
            seen.add(node)

    # Build (node, discount) pairs
    pairs: List[Tuple] = []
    for node in candidates:
        ev = env._estimate_valuation(node)
        for d in cfg.discount_grid:
            price      = ev * (1.0 - d)
            affordable = (env.B - c + price) >= -1e-9
            if affordable:
                pairs.append((node, d, price))

    # ── Snapshot current state (no ground-truth info copied) ─────────────────
    snap = _snapshot_env_state(env)

    # ── Pre-compute degree ordering for rollout base policy ───────────────────
    # Computed once per expert_step call, shared across all candidate clones.
    degree_order = sorted(env.nodes, key=lambda v: env.graph.degree(v), reverse=True)

    # ── Score each candidate pair ─────────────────────────────────────────────
    best_score    = -float("inf")
    best_node_idx = env.node_to_idx[available[0]]
    best_discount = 1.0                     # safe fallback: free offer

    for pair_idx, (node, d, price) in enumerate(pairs):
        clone_seed = env.cfg.seed * 10000 + env.t * 100 + pair_idx

        # Build clone with fresh simulated link weights
        clone = _make_clone(env, snap, clone_seed=clone_seed, n_mc_rollout=n_mc)

        # Step 1: apply this candidate action on the clone
        node_idx = clone.node_to_idx[node]
        _, imm_reward, done, _ = clone.step(node_idx, d)

        if done:
            # Episode ended immediately (bankrupt / all offered)
            k_terminal = clone.t
            score = imm_reward + _j_lookup(J_table, clone.B, k_terminal, delta)
        else:
            # Step 2: rollout H more steps with degree-order base policy
            # Pass pre-computed degree_order (same topology, no re-sort needed).
            rollout_rev, final_B, final_t = _run_rollout(
                clone, H, c, b, degree_order=degree_order
            )

            # Step 3: terminal value from J-table
            j_term = _j_lookup(J_table, final_B, final_t, delta)
            score  = imm_reward + rollout_rev + j_term

        if score > best_score:
            best_score    = score
            best_node_idx = node_idx
            best_discount = d

    return best_node_idx, best_discount


# ── Full trajectory generation ─────────────────────────────────────────────────

def generate_rollout_expert_trajectory(
    graph: nx.Graph,
    cfg: BudgetEnvConfig,
    k: int,
    c: float = 0.3,
    seed: int = 0,
    expert_cfg: Optional[RolloutExpertConfig] = None,
    force_rebuild: bool = False,
) -> List[dict]:
    """Generate a full-episode rollout-expert trajectory on the REAL env.

    The expert decides each step using rollout_expert_step.  The trajectory
    records exactly the same fields as mixed_expert_trajectories.py so that
    both can be fed to the same imitation-learning trainer.

    Caching: trajectories are pickled to
        results/logs/rollout_expert_cache/{graph_hash}_k{k}_s{seed}.pkl
    The cache is keyed on (graph topology, k, seed) and is rebuilt if
    force_rebuild=True or if the cached file is absent.

    Args:
        graph:        NetworkX graph.
        cfg:          BudgetEnvConfig (seed used for env; weight_high controls weights).
        k:            Initial budget parameter (B_0 = k * c).
        c:            Production cost per offer.
        seed:         RNG seed for the live episode env.
        expert_cfg:   RolloutExpertConfig (default: RolloutExpertConfig()).
        force_rebuild: If True, skip cache and regenerate.

    Returns:
        trajectory: List[dict] with keys
            node_idx, discount, accepted, price, B_after.
            (Identical format to mixed_expert_trajectories.py.)
    """
    if expert_cfg is None:
        expert_cfg = RolloutExpertConfig(c=c)

    # Cache lookup
    gh         = _graph_hash(graph)
    cache_path = os.path.join(_ROLLOUT_CACHE_DIR, f"{gh}_k{k}_s{seed}.pkl")
    if os.path.exists(cache_path) and not force_rebuild:
        with open(cache_path, "rb") as fh:
            return pickle.load(fh)

    # Build J-table (or load from cache)
    max_budget = float(k * c) * 3.0 + 1.0    # enough headroom; table caps at this
    env_cfg_for_j = BudgetEnvConfig(
        budget_B=max_budget, production_cost=c, seed=seed, weight_high=cfg.weight_high,
        n_mc_samples=200, influence_model=cfg.influence_model, b=cfg.b,
    )
    J_table = build_j_table(
        graph, env_cfg_for_j, max_budget=max_budget,
        delta=expert_cfg.j_delta, tiers=expert_cfg.j_tiers,
    )

    # Create the REAL env (true link weights are drawn here)
    real_cfg = BudgetEnvConfig(
        budget_B=float(k * c),
        production_cost=c,
        seed=seed,
        weight_high=cfg.weight_high,
        n_mc_samples=cfg.n_mc_samples,
        influence_model=cfg.influence_model,
        b=cfg.b,
    )
    env = BudgetRevenueEnv(graph, real_cfg)
    env.reset()

    trajectory: List[dict] = []

    while len(env.offered) < env.n and not env._check_bankrupt():
        node_idx, discount = rollout_expert_step(env, J_table, expert_cfg)

        _, reward, done, info = env.step(node_idx, discount)
        trajectory.append({
            "node_idx": int(node_idx),
            "discount": float(discount),
            "accepted": bool(info.get("accepted", False)),
            "price":    float(info.get("offered_price", 0.0)),
            "B_after":  float(env.B),
        })
        if done:
            break

    # Save to cache
    with open(cache_path, "wb") as fh:
        pickle.dump(trajectory, fh)

    return trajectory


# ── Stage A evaluation wrapper ─────────────────────────────────────────────────

def evaluate_rollout_expert(
    graph: nx.Graph,
    k: int,
    c: float = 0.3,
    n_trials: int = 3,
    expert_cfg: Optional[RolloutExpertConfig] = None,
    base_cfg: Optional[BudgetEnvConfig] = None,
    verbose: bool = False,
) -> dict:
    """Run rollout expert as a POLICY and record revenue (Stage A evaluation).

    Generates ``n_trials`` independent trajectories (different seeds) and
    returns aggregated revenue statistics.  Measures wall-clock time per episode
    for the feasibility check (n=200 episode > 10 min → flag scale-up infeasible).

    Args:
        graph:      NetworkX graph.
        k:          Initial budget parameter.
        c:          Production cost.
        n_trials:   Number of independent seeds.
        expert_cfg: RolloutExpertConfig (default: RolloutExpertConfig(c=c)).
        base_cfg:   BudgetEnvConfig template (seed/weight_high).
        verbose:    If True, print per-trial progress.

    Returns:
        dict with keys: revenue (mean/std/all), wall_clock_sec (per episode),
        n_accepted, bankrupt_count.
    """
    if expert_cfg is None:
        expert_cfg = RolloutExpertConfig(c=c)
    if base_cfg is None:
        base_cfg = BudgetEnvConfig(
            budget_B=float(k * c), production_cost=c, seed=0, weight_high=2.0
        )

    revenues:       List[float] = []
    n_accepted_all: List[int]   = []
    bankrupt_count: int         = 0
    wall_clocks:    List[float] = []

    for trial in range(n_trials):
        seed = trial
        t0   = time.time()

        traj = generate_rollout_expert_trajectory(
            graph, base_cfg, k=k, c=c, seed=seed,
            expert_cfg=expert_cfg, force_rebuild=False,
        )
        elapsed = time.time() - t0

        total_rev  = sum(s["price"] for s in traj if s["accepted"])
        n_acc      = sum(1 for s in traj if s["accepted"])
        is_bankrupt = (len(traj) < graph.number_of_nodes())

        revenues.append(total_rev)
        n_accepted_all.append(n_acc)
        if is_bankrupt:
            bankrupt_count += 1
        wall_clocks.append(elapsed)

        if verbose:
            print(f"  trial {trial}: rev={total_rev:.1f}  n_acc={n_acc}"
                  f"  bankrupt={is_bankrupt}  wall={elapsed:.1f}s")

    return {
        "revenue":          {"mean": float(np.mean(revenues)),
                             "std":  float(np.std(revenues)),
                             "all":  revenues},
        "n_accepted":       {"mean": float(np.mean(n_accepted_all)),
                             "std":  float(np.std(n_accepted_all)),
                             "all":  n_accepted_all},
        "bankrupt_count":   bankrupt_count,
        "wall_clock_sec":   float(np.mean(wall_clocks)),
        "wall_clock_all":   wall_clocks,
    }
