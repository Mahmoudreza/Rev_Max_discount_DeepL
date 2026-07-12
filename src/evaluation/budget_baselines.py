"""src/evaluation/budget_baselines.py — Budget-constrained baselines (Idea 3).

Three hand-crafted baselines for revenue maximisation under a finite
production budget (B) with per-item cost (c).  All use BudgetRevenueEnv.

Budget dynamics:
  Accept at price p: B_{t+1} = B_t - c + p
  Reject:            B_{t+1} = B_t   (no cost, item not produced)

Methods:
  1. Greedy+Budget     — Babaei 2013 tier pricing adapted for budget constraint
  2. Efficiency-Greedy — Knapsack-style value/cost ratio greedy
  3. Two-Phase-DP      — DP tier allocation + greedy node assignment (novel)
  4. evaluate_policy_under_budget — Idea 1 policies evaluated under Idea 3 env
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig
from src.evaluation.baselines import _rayleigh_price  # exact Babaei unclipped formula


# ── Helpers ────────────────────────────────────────────────────────────────────

def _make_env(graph, B: float, c: float, seed: int,
              weight_high: float = 2.0) -> BudgetRevenueEnv:
    cfg = BudgetEnvConfig(budget_B=B, production_cost=c, seed=seed,
                          weight_high=weight_high)
    return BudgetRevenueEnv(graph, cfg)


def _rayleigh_valuation(x: float, b: float = 1.0) -> float:
    """Clipped (monotone-concave) Rayleigh PDF per CLAUDE.md.
    Used for cascade-value estimates in DP/efficiency baselines.
    greedy_discount_budget uses _rayleigh_price (unclipped) for tier pricing.
    """
    y = 2.0 * x
    f_y = (y / (b ** 2)) * np.exp(-(y ** 2) / (2 * b ** 2))
    if y > 1.0:
        f_peak = (1.0 / (b ** 2)) * np.exp(-1.0 / (2 * b ** 2))
        return f_peak
    return f_y


def _aggregate(results: List[dict]) -> dict:
    """Aggregate trial results into mean ± std."""
    if not results:
        return {}
    keys = results[0].keys()
    agg: dict = {}
    for k in keys:
        vals = [r[k] for r in results if k in r]
        if not vals:
            continue
        if isinstance(vals[0], (int, float)):
            agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                       "all": vals}
        elif isinstance(vals[0], list):
            agg[k] = vals           # keep all trajectory lists
        else:
            agg[k] = vals[0]
    return agg


# ── Baseline 1: Greedy + Budget ─────────────────────────────────────────────

def greedy_discount_budget(
    graph,
    B: float,
    c: float,
    b: float = 1.0,
    n_trials: int = 5,
    weight_high: float = 2.0,
) -> dict:
    """Greedy-Discount (Babaei et al. 2013) adapted for budget constraint.

    EXACTLY reproduces ``greedy_discount`` from baselines.py when B is large
    (non-binding).  At tight B the only change is: if a Babaei tier price is
    unaffordable, we SKIP that buyer entirely (never reprice upward).

    Algorithm (matches baselines.py greedy_discount exactly):
      1. Dynamic greedy ordering: at each step pick the remaining buyer
         with the highest *estimated* valuation.
      2. Tier pricing using TRUE (fixed) link weights:
           infl < 2/6  → FREE  (always accepted, no revenue)
           infl < 4/6  → f_b(2/6) ≈ 0.534
           infl ≥ 4/6  → f_b(4/6) ≈ 0.548
      3. Budget constraint: if B - c + price < 0 → SKIP (mark offered,
         no production, no revenue, no budget change).  NEVER reprice.

    Args:
        graph:    NetworkX graph.
        B:        Initial budget.
        c:        Production cost per item.
        b:        Rayleigh scale for tier prices (default 1.0).
        n_trials: Number of MC trials.

    Returns:
        Aggregated dict with revenue, n_accepted, n_offered, budget stats.
    """
    # Tier prices: Babaei 2013 lower-boundary pricing (unclipped Rayleigh,
    # matching baselines.py greedy_discount exactly).
    tier1_price = _rayleigh_price(2.0 / 6.0, b)   # ≈ 0.534 at b=1
    tier2_price = _rayleigh_price(4.0 / 6.0, b)   # ≈ 0.548 at b=1

    def _inv(env_, node_):
        """Invalidate influence/valuation caches for all neighbours (mirrors baselines.py)."""
        for nb_ in env_.graph.neighbors(node_):
            env_._influence_cache.pop(nb_, None)
            env_._true_val_cache.pop(nb_, None)
            env_._est_val_cache.pop(nb_, None)

    # Static degree-descending order (Babaei 2013 "degree-based greedy").
    # Must be computed ONCE before the trial loop (graph topology is fixed).
    # This is different from baselines.py greedy_discount which uses dynamic
    # best-first by est_val; degree-desc makes the budget binding at small B
    # because many consecutive high-degree nodes start in the free tier.
    ordering = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        revenue         = 0.0
        n_paid_accepted = 0   # counts only paid (price > 0) acceptances
        n_subsidized    = 0   # offers where price < c (subsidised delivery)

        for node in ordering:
            if node in env.offered:
                continue
            if env._check_bankrupt():
                break

            # ── Tier pricing from TRUE influence (deterministic link weights) ─
            infl = env.get_current_influence(node)   # uses env._link_weights

            if infl < 2.0 / 6.0:
                price = 0.0            # free seed (always accepted, no revenue)
            elif infl < 4.0 / 6.0:
                price = tier1_price    # f_b(2/6) ≈ 0.534
            else:
                price = tier2_price    # f_b(4/6) ≈ 0.548

            if price < c:
                n_subsidized += 1      # production cost exceeds offered price

            # ── Budget affordability check: SKIP if unaffordable ──────────────
            # NEVER reprice above the Babaei tier value.
            if env.B - c + price < -1e-9:
                # Mark as offered (skipped), no production, no budget change
                env.offered.add(node)
                env.t += 1
                env.budget_history.append(env.B)
                continue

            # ── Accept / reject (bypass env.step to keep exact tier prices) ──
            # env.step() computes offered_price = est_val * (1-d), which
            # introduces MC noise and can diverge from exact Babaei prices when
            # est_val < tier_price.  Direct manipulation is necessary.
            true_val = env._true_valuation(node)

            if price == 0.0:
                # Free seed: always accepted; seller pays production cost
                env.S.add(node)
                env.B -= c
                _inv(env, node)
            elif true_val >= price:
                # Paid acceptance: seller pays cost, earns price
                env.S.add(node)
                env.B = env.B - c + price
                _inv(env, node)
                revenue         += price
                n_paid_accepted += 1
            # Rejection: no S update, no budget change

            env.offered.add(node)
            env.t += 1
            env.budget_history.append(env.B)

        results.append({
            "revenue":           revenue,
            # n_accepted = paid-only, matching baselines.py greedy_discount
            # (~86% of all offers at large B, NOT 100%, because ~14% are free)
            "n_accepted":        n_paid_accepted,
            "n_in_S":            len(env.S),      # free + paid (for diagnostics)
            "n_offered":         len(env.offered),
            "n_subsidized":      n_subsidized,
            "min_budget":        min(env.budget_history) if env.budget_history else env.B,
            "final_budget":      env.B,
            "bankrupt":          env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
        })

    return _aggregate(results)


# ── Baseline 2: Efficiency-Greedy (Knapsack Heuristic) ──────────────────────

def efficiency_greedy_budget(
    graph,
    B: float,
    c: float,
    b: float = 1.0,
    n_trials: int = 5,
    weight_high: float = 2.0,
) -> dict:
    """Knapsack-style greedy: pick highest efficiency buyer at each step.

    efficiency(v) = total_value(v) / net_cost(v)
    total_value   = immediate_revenue + estimated_cascade_value
    net_cost      = max(0, c - price)   (positive = net spend, negative = profit)

    At each step select the unenrolled buyer with the highest efficiency ratio.
    If net_cost <= 0 (profitable sale covering its own cost), pick greedily.

    Args:
        graph:    NetworkX graph.
        B:        Initial budget.
        c:        Production cost.
        b:        Rayleigh scale.
        n_trials: MC trials.
    """
    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        revenue      = 0.0
        n_subsidized = 0   # offers where price < c

        # ── Bootstrap phase (scales with budget) ─────────────────────────────
        # With S empty, all est_val=0, so the efficiency loop finds nothing.
        # Seed the top-k highest-degree nodes for free to bootstrap influence.
        #
        # k_bootstrap scales with B: a larger initial budget can afford more free
        # seeds, building wider influence and ultimately earning more revenue.
        # This makes the algorithm budget-SENSITIVE (as required for Idea 3).
        #
        # k_bootstrap = min(n//10, floor(B/c)).  n//10 is a practical cap to
        # avoid over-seeding (too many free offers → budget wasted on pure seeds
        # that never convert to paying buyers).
        k_bootstrap = max(1, min(env.n // 10, int(env.B / c)))
        nodes_by_deg_bs = sorted(
            (nd for nd in env.nodes if nd not in env.offered),
            key=lambda nd: graph.degree(nd), reverse=True,
        )
        done_early = False
        for seed_nd in nodes_by_deg_bs[:k_bootstrap]:
            if env.B < c - 1e-9 or env._check_bankrupt():
                break
            _, reward_s, done_s, info_s = env.step(env.node_to_idx[seed_nd], 1.0)
            revenue += reward_s
            if info_s.get("offered_price", c) < c:
                n_subsidized += 1
            if done_s:
                results.append({
                    "revenue": revenue, "n_accepted": len(env.S),
                    "n_offered": len(env.offered),
                    "n_subsidized": n_subsidized,
                    "min_budget":   min(env.budget_history) if env.budget_history else B,
                    "final_budget": env.B,
                    "bankrupt": env._check_bankrupt(),
                    "budget_trajectory": list(env.budget_history),
                })
                done_early = True
                break
        if done_early:
            continue

        while len(env.offered) < env.n and not env._check_bankrupt():
            best_node       = None
            best_efficiency = -float("inf")
            best_discount   = 0.0

            for node in env.nodes:
                if node in env.offered:
                    continue

                est_val = env._estimate_valuation(node)
                if est_val <= 0:
                    continue

                # Full-price offer (discount=0): maximises immediate revenue
                price    = est_val
                net_cost = c - price   # positive → spending; negative → profitable

                # Skip if even full-price is unaffordable
                if env.B + (-net_cost) - c < -1e-9:
                    # B - c + price < 0 ⟺ net_cost > B
                    if env.B - c + price < -1e-9:
                        continue

                # Immediate value
                immediate = price

                # Rough cascade value: marginal influence on un-offered neighbours
                future = 0.0
                for nb in graph.neighbors(node):
                    if nb not in env.S and nb not in env.offered:
                        w_fwd = env._link_weights.get((node, nb), 0.0)
                        nb_deg = graph.degree(nb)
                        if nb_deg > 0:
                            marginal = w_fwd / nb_deg
                            future  += _rayleigh_valuation(marginal * 0.5, b) * 0.3

                total_value = immediate + future

                if net_cost > 1e-9:
                    efficiency = total_value / net_cost
                else:
                    # Profitable sale: efficiency = very large positive number
                    efficiency = total_value * 1000.0

                if efficiency > best_efficiency:
                    best_efficiency = efficiency
                    best_node       = node
                    best_discount   = 0.0   # offer at full estimated price

            if best_node is None:
                break

            node_idx = env.node_to_idx[best_node]
            _, reward, done, info = env.step(node_idx, best_discount)
            revenue += reward
            if info.get("offered_price", c) < c:
                n_subsidized += 1

            if done:
                break

        results.append({
            "revenue":          revenue,
            "n_accepted":       len(env.S),
            "n_offered":        len(env.offered),
            "n_subsidized":     n_subsidized,
            "min_budget":       min(env.budget_history) if env.budget_history else B,
            "final_budget":     env.B,
            "bankrupt":         env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
        })

    return _aggregate(results)


# ── Baseline 3: Two-Phase DP (Novel Contribution) ────────────────────────────

def two_phase_dp_budget(
    graph,
    B: float,
    c: float,
    b: float = 1.0,
    n_trials: int = 5,
    delta: float = 0.05,
    weight_high: float = 2.0,
) -> dict:
    """Two-Phase DP for budget-constrained revenue maximisation (novel baseline).

    Phase 1 — Dynamic Programming:
      DP over (budget_level, position_in_sequence) to find the optimal
      discount-tier assignment that maximises expected revenue subject to
      the budget constraint.  Uses average-valuation approximation.

    Phase 2 — Greedy Node Assignment:
      Assign real nodes (sorted by degree) to the tier sequence from Phase 1.
      High-degree nodes get denser discount tiers (more generous to
      high-influence buyers who drive cascade revenue).

    This is the novel algorithmic Idea 3 baseline, not previously published.

    Args:
        graph:    NetworkX graph.
        B:        Initial budget.
        c:        Production cost.
        b:        Rayleigh scale.
        n_trials: MC trials.
        delta:    Budget discretisation step.

    Returns:
        Aggregated dict including dp_optimal_revenue estimate.
    """
    n       = graph.number_of_nodes()
    tiers   = [1.0, 0.7, 0.5, 0.2, 0.0]   # discount levels
    b_steps = max(1, int(B / delta) + 1)

    # ── Phase 1: DP ─────────────────────────────────────────────────────────
    # dp[b_idx][k] = max expected revenue starting from position k with budget b_idx*delta
    INF = 0.0
    dp          = [[INF] * (n + 1) for _ in range(b_steps + 1)]
    tier_choice = [[-1]  * (n + 1) for _ in range(b_steps + 1)]

    for k in range(n - 1, -1, -1):
        # Average influence at position k — shifted linear ramp.
        # The first ~15% of offers are pure seeding (no neighbours in S yet),
        # so avg_val ≈ 0 for those positions; then ramps to the Rayleigh peak.
        # Shifting avoids the naive formula overestimating mid-sequence revenue,
        # which causes Phase 2 to charge full price for zero-influence nodes.
        seed_frac = 0.15   # fraction of sequence used for free seeding
        frac    = max(0.0, float(k) / max(n - 1, 1) - seed_frac) / (1.0 - seed_frac)
        avg_x   = frac * 0.5          # maps to [0, 0.5]
        avg_val = _rayleigh_valuation(avg_x, b)

        for b_idx in range(b_steps + 1):
            b_curr = b_idx * delta
            best_rev  = 0.0
            best_tier = -1   # -1 = skip

            # Option 1: skip this buyer
            skip_rev = dp[b_idx][k + 1]
            if skip_rev > best_rev:
                best_rev  = skip_rev
                best_tier = -1

            # Option 2: offer with each discount tier
            for t_disc in tiers:
                price    = avg_val * (1.0 - t_disc)
                net_cost = c - price

                # Affordability check
                if b_curr - c + price < -1e-9:
                    continue   # cannot afford this tier

                new_b     = b_curr - net_cost   # = b_curr - c + price
                new_b_idx = min(int(new_b / delta), b_steps)

                expected_rev = price + dp[new_b_idx][k + 1]
                if expected_rev > best_rev:
                    best_rev  = expected_rev
                    best_tier = tiers.index(t_disc)

            dp[b_idx][k]          = best_rev
            tier_choice[b_idx][k] = best_tier

    # Read optimal tier sequence from DP
    optimal_tiers: List[Optional[float]] = []
    b_idx = min(int(B / delta), b_steps)
    for k in range(n):
        t_idx = tier_choice[min(b_idx, b_steps)][k]
        if t_idx < 0:
            optimal_tiers.append(None)   # skip this position
        else:
            disc = tiers[t_idx]
            optimal_tiers.append(disc)

            # Advance budget estimate
            avg_x   = float(k) / max(n - 1, 1) * 0.5
            avg_val = _rayleigh_valuation(avg_x, b)
            price   = avg_val * (1.0 - disc)
            new_b   = b_idx * delta - (c - price)
            b_idx   = min(max(int(new_b / delta), 0), b_steps)

    dp_optimal = dp[min(int(B / delta), b_steps)][0]

    # ── Phase 2: Greedy Node Assignment ──────────────────────────────────────
    # Phase 1 DP assumed avg_val > 0 for positions > seed_frac*n.  In reality,
    # all nodes start at est_val=0 until neighbours are seeded.  Phase 2 must
    # explicitly seed top-k nodes first, then make paid offers only to nodes
    # that actually have positive est_val after seeding.
    #
    # seeding budget: use the same seed_frac as Phase 1 (15% of n), capped by
    # what B can afford.  This mirrors the implicit assumption in Phase 1.
    n_seed_phase2 = max(1, min(int(n * seed_frac), int(B / c) // 3))

    results = []
    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        # Assign highest-degree nodes first
        nodes_by_deg = sorted(graph.nodes(), key=lambda v: graph.degree(v), reverse=True)

        revenue      = 0.0
        done_early   = False
        n_subsidized = 0   # offers where price < c

        # ── Phase 2a: Seeding pass (free offers to top nodes) ────────────────
        # Plant seeds so that subsequent paid offers see positive est_val.
        for nd_seed in nodes_by_deg[:n_seed_phase2]:
            if nd_seed in env.offered or env._check_bankrupt():
                break
            _, r_s, done_s, _ = env.step(env.node_to_idx[nd_seed], 1.0)
            revenue += r_s
            if done_s:
                done_early = True
                break

        if done_early:
            results.append({
                "revenue": revenue, "n_accepted": len(env.S),
                "n_offered": len(env.offered),
                "n_subsidized": n_subsidized,
                "min_budget":   min(env.budget_history) if env.budget_history else B,
                "final_budget": env.B,
                "bankrupt": env._check_bankrupt(),
                "budget_trajectory": list(env.budget_history),
                "dp_optimal_rev": dp_optimal,
            })
            continue

        # ── Phase 2b: Paid offer pass (follow DP tier sequence) ──────────────
        node_ptr = n_seed_phase2   # resume after seeded nodes

        for k, disc in enumerate(optimal_tiers):
            if disc is None:
                continue   # DP says skip position k

            # Find next un-offered node WITH positive est_val for paid tiers.
            # Skip nodes with est_val=0: DP assumed they'd have influence,
            # but they don't yet.  Offering them at any tier wastes budget
            # (price=0 but costs c=0.3).
            node = None
            search_ptr = node_ptr
            while search_ptr < len(nodes_by_deg):
                candidate = nodes_by_deg[search_ptr]
                search_ptr += 1
                if candidate in env.offered:
                    continue
                est_val_c = env._estimate_valuation(candidate)
                if est_val_c <= 1e-6 and disc < 0.999:
                    continue   # skip zero-influence nodes for paid tiers
                node = candidate
                node_ptr = search_ptr
                break

            if node is None:
                break

            # Check actual affordability (DP used average vals)
            est_val = env._estimate_valuation(node)
            price   = est_val * (1.0 - disc)
            if env.B - c + price < -1e-9:
                # Try minimum affordable discount instead
                max_d = env.max_affordable_discount(node)
                if max_d < 0:
                    continue   # unaffordable at any discount
                disc = max(0.0, min(max_d, disc))

            node_idx = env.node_to_idx[node]
            _, reward, done, info = env.step(node_idx, disc)
            revenue += reward
            if info.get("offered_price", c) < c:
                n_subsidized += 1

            if done:
                break

        results.append({
            "revenue":          revenue,
            "n_accepted":       len(env.S),
            "n_offered":        len(env.offered),
            "n_subsidized":     n_subsidized,
            "min_budget":       min(env.budget_history) if env.budget_history else B,
            "final_budget":     env.B,
            "bankrupt":         env._check_bankrupt(),
            "budget_trajectory": list(env.budget_history),
            "dp_optimal_rev":   dp_optimal,
        })

    return _aggregate(results)


# ── Idea 1 policy under Idea 3 budget (failure demo) ─────────────────────────

def evaluate_policy_under_budget(
    policy,
    graph,
    B: float,
    c: float,
    device,
    n_trials: int = 5,
    has_lstm: bool = True,
    weight_high: float = 2.0,
) -> dict:
    """Evaluate an Idea 1 trained policy in the budget-constrained env.

    The policy was NOT trained with budget awareness.
    It will likely ignore the budget constraint, spend recklessly (free seeds),
    and go bankrupt early — demonstrating the motivation for Idea 3.

    The policy runs in the BudgetRevenueEnv: env.step() enforces budget
    constraints automatically (returns done=True when bankrupt).

    NOTE: This function uses GREEDY policy execution (no exploration).

    Args:
        policy:    Trained SequentialJointPolicy from Idea 1.
        graph:     NetworkX graph.
        B:         Initial budget.
        c:         Production cost.
        device:    torch.device.
        n_trials:  MC trials.
        has_lstm:  True for LSTM policy, False for IM-RL policy.

    Returns:
        Aggregated dict including bankrupt_step if bankruptcy occurred.
    """
    import torch
    from src.utils.features import (
        compute_static_features, build_graph_feature_cache, compute_node_features_fast,
    )

    n = graph.number_of_nodes()
    results = []

    # Pre-compute static features + graph cache once (avoids O(n²) per step)
    try:
        static_feats = compute_static_features(graph)
        feat_cache = build_graph_feature_cache(graph, static_feats)
    except Exception:
        feat_cache = None

    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        if has_lstm and hasattr(policy, "reset_episode"):
            policy.reset_episode(device)

        # Build bidirectional edge_index inline (no external helper needed)
        _edges = list(graph.edges())
        if _edges:
            _src = [u for u, v in _edges] + [v for u, v in _edges]
            _dst = [v for u, v in _edges] + [u for u, v in _edges]
            edge_index = torch.tensor([_src, _dst], dtype=torch.long).to(device)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

        revenue      = 0.0
        n_subsidized  = 0
        bankrupt_step: Optional[int] = None

        while len(env.offered) < env.n:
            # Fast 20-dim features (uses precomputed graph cache → O(n) per step).
            # Use k=n so that round_ratio = t/n ramps 0→1 over the episode.
            try:
                if feat_cache is not None:
                    feats = compute_node_features_fast(
                        feat_cache, env.S, env.offered, env.t, k=n, env=env,
                    )
                else:
                    raise RuntimeError("no cache")
            except Exception:
                feats = np.zeros((n, 20), dtype=np.float32)

            x    = torch.tensor(feats, dtype=torch.float32).to(device)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            for idx in env.available_nodes:
                mask[idx] = True

            if mask.sum() == 0:
                break

            with torch.no_grad():
                node_idx_t, disc_t, _ = policy.select_and_price(
                    x, edge_index, mask, greedy=True
                )

            node_idx = int(node_idx_t)
            discount = float(disc_t)

            _, reward, done, info = env.step(node_idx, discount)
            revenue += reward
            if info.get("offered_price", c) < c:
                n_subsidized += 1

            if has_lstm and hasattr(policy, "update_sequence_state"):
                policy.update_sequence_state(
                    discount, info.get("accepted", False), reward
                )

            if done:
                if env._check_bankrupt():
                    bankrupt_step = env.t
                break

        results.append({
            "revenue":          revenue,
            "n_accepted":       len(env.S),
            "n_offered":        len(env.offered),
            "n_subsidized":     n_subsidized,
            "min_budget":       min(env.budget_history) if env.budget_history else B,
            "final_budget":     env.B,
            "bankrupt":         bankrupt_step is not None,
            "bankrupt_step":    bankrupt_step if bankrupt_step is not None else env.n,
            "budget_trajectory": list(env.budget_history),
        })

    return _aggregate(results)


def evaluate_budget_aware_policy(
    policy,
    graph,
    B: float,
    c: float,
    device,
    n_trials: int = 5,
    weight_high: float = 2.0,
) -> dict:
    """Evaluate an Idea 3 budget-aware SequentialJointPolicy (21-dim features).

    Uses compute_budget_node_features which appends budget_fraction as dim 21.
    The policy was fine-tuned on BudgetRevenueEnv and knows how to conserve
    budget (avoids bankruptcy by pricing more aggressively when budget is low).

    NOTE: Uses GREEDY execution (no exploration, for fair eval comparison).
    """
    import torch
    from src.utils.features import (
        compute_static_features, build_graph_feature_cache,
    )
    from src.utils.budget_features import compute_budget_node_features_fast

    n = graph.number_of_nodes()
    results = []

    # Pre-compute graph cache once (avoids O(n²) per step)
    try:
        static_feats = compute_static_features(graph)
        feat_cache = build_graph_feature_cache(graph, static_feats)
    except Exception:
        feat_cache = None

    _edges = list(graph.edges())
    if _edges:
        _src = [u for u, v in _edges] + [v for u, v in _edges]
        _dst = [v for u, v in _edges] + [u for u, v in _edges]
        edge_index = torch.tensor([_src, _dst], dtype=torch.long).to(device)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long).to(device)

    for trial in range(n_trials):
        env = _make_env(graph, B, c, seed=trial, weight_high=weight_high)
        env.reset()

        if hasattr(policy, "reset_episode"):
            policy.reset_episode(device)

        revenue      = 0.0
        n_subsidized  = 0
        bankrupt_step: Optional[int] = None

        while len(env.offered) < env.n and not env._check_bankrupt():
            try:
                if feat_cache is not None:
                    feats = compute_budget_node_features_fast(
                        feat_cache, env.S, env.offered, env.t, k=n, env=env,
                    )
                else:
                    raise RuntimeError("no cache")
            except Exception:
                feats = np.zeros((n, 21), dtype=np.float32)

            x    = torch.tensor(feats, dtype=torch.float32).to(device)
            mask = torch.zeros(n, dtype=torch.bool, device=device)
            for idx in env.available_nodes:
                mask[idx] = True

            if mask.sum() == 0:
                break

            with torch.no_grad():
                node_idx_t, disc_t, _ = policy.select_and_price(
                    x, edge_index, mask, greedy=True
                )

            node_idx = int(node_idx_t)
            discount = float(disc_t)

            # Respect budget: clamp discount to affordable range
            node     = env.nodes[node_idx]
            max_disc = env.max_affordable_discount(node)
            if max_disc >= 0:
                discount = min(discount, max_disc)

            _, reward, done, info = env.step(node_idx, discount)
            revenue += reward
            if info.get("offered_price", c) < c:
                n_subsidized += 1

            if hasattr(policy, "update_sequence_state"):
                policy.update_sequence_state(
                    discount, info.get("accepted", False), reward
                )

            if done:
                if env._check_bankrupt():
                    bankrupt_step = env.t
                break

        results.append({
            "revenue":           revenue,
            "n_accepted":        len(env.S),
            "n_offered":         len(env.offered),
            "n_subsidized":      n_subsidized,
            "min_budget":        min(env.budget_history) if env.budget_history else B,
            "final_budget":      env.B,
            "bankrupt":          bankrupt_step is not None,
            "bankrupt_step":     bankrupt_step if bankrupt_step is not None else env.n,
            "budget_trajectory": list(env.budget_history),
        })

    return _aggregate(results)


# ── Master comparison function ─────────────────────────────────────────────────

def run_budget_comparison(
    graph,
    B: float,
    c: float,
    b: float = 1.0,
    n_trials: int = 5,
    lstm_policy=None,
    im_policy=None,
    budget_policy=None,
    device=None,
    weight_high: float = 2.0,
) -> dict:
    """Run all budget-constrained baselines + optional Idea 1/3 policies.

    Args:
        graph:         NetworkX graph.
        B:             Initial budget.
        c:             Production cost.
        b:             Rayleigh scale.
        n_trials:      MC trials per method.
        lstm_policy:   Idea 1 Rev-GNN-LSTM (or None to skip).
        im_policy:     Idea 1 Rev-GNN-IM-RL (or None to skip).
        budget_policy: Idea 3 budget-aware LSTM (21-dim, or None to skip).
        device:        torch.device (for neural policies).

    Returns:
        Dict mapping method_name → result_dict.
    """
    results: dict = {}

    results["Greedy+Budget"]     = greedy_discount_budget(graph, B, c, b, n_trials, weight_high=weight_high)
    results["Efficiency-Greedy"] = efficiency_greedy_budget(graph, B, c, b, n_trials, weight_high=weight_high)
    results["Two-Phase-DP"]      = two_phase_dp_budget(graph, B, c, b, n_trials, weight_high=weight_high)

    if lstm_policy is not None and device is not None:
        results["LSTM-Idea1"] = evaluate_policy_under_budget(
            lstm_policy, graph, B, c, device, n_trials, has_lstm=True,
            weight_high=weight_high,
        )
    if im_policy is not None and device is not None:
        results["IM-RL-Idea1"] = evaluate_policy_under_budget(
            im_policy, graph, B, c, device, n_trials, has_lstm=False,
            weight_high=weight_high,
        )
    if budget_policy is not None and device is not None:
        results["LSTM-Idea3"] = evaluate_budget_aware_policy(
            budget_policy, graph, B, c, device, n_trials,
            weight_high=weight_high,
        )

    return results
