"""src/env/budget_revenue_env.py — Budget-constrained revenue MDP (Idea 3).

BudgetRevenueEnv INHERITS from RevenueEnv and adds a finite production budget.
All valuation logic (both true and estimated) is delegated to the parent class,
guaranteeing identical results with the original Idea 1 environment.

Budget dynamics (Idea 3 spec):
  Accept at price p: B_{t+1} = B_t - c + p
  Reject:            B_{t+1} = B_t    (no cost — item NOT produced)

Bankruptcy condition:
  B_t < c AND max_est_val(remaining) < c - B_t
  i.e. no remaining buyer can pay enough to cover the production cost.

The LSTM and IM-RL policies trained in Idea 1 are budget-UNAWARE.
They give free/cheap offers to seed influence → drain budget → bankruptcy.
This motivates Idea 3: training a budget-aware policy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.env.revenue_env import RevenueEnv, RevenueEnvConfig


# ── Config ─────────────────────────────────────────────────────────────────────

@dataclass
class BudgetEnvConfig:
    """Configuration for the budget-constrained MDP.

    Extends RevenueEnvConfig with budget parameters.
    All RevenueEnvConfig fields are replicated here so that BudgetRevenueEnv
    can pass this directly to the parent RevenueEnv.__init__.
    """
    # ── Budget parameters (new for Idea 3) ──
    budget_B: float = 10.0           # initial production budget
    production_cost: float = 0.3     # per-item production cost c

    # ── Valuation model (mirrored from RevenueEnvConfig) ──
    influence_model: str = "monotone"
    acceptance_mode: str = "monotone"   # "monotone" (default) or "rayleigh_nm"
    b: float = 1.0
    weight_low: float = 0.0
    weight_high: float = 2.0
    n_mc_samples: int = 200
    reward_type: str = "flat"
    gamma: float = 1.0
    seed: int = 42

    def to_revenue_cfg(self) -> RevenueEnvConfig:
        """Build a RevenueEnvConfig from the shared fields."""
        return RevenueEnvConfig(
            influence_model=self.influence_model,
            b=self.b,
            weight_low=self.weight_low,
            weight_high=self.weight_high,
            n_mc_samples=self.n_mc_samples,
            reward_type=self.reward_type,
            gamma=self.gamma,
            seed=self.seed,
        )


# ── Environment ────────────────────────────────────────────────────────────────

class BudgetRevenueEnv(RevenueEnv):
    """Budget-constrained Revenue MDP.

    Inherits ALL valuation logic from RevenueEnv (identical results guaranteed).
    Only ``reset()`` and ``step()`` are overridden to add budget mechanics.

    Args:
        graph: NetworkX graph.
        cfg:   BudgetEnvConfig (or any duck-typed object with the same fields).
    """

    def __init__(self, graph, cfg: BudgetEnvConfig) -> None:
        # Delegate to parent using the shared valuation fields.
        # Python duck-typing: RevenueEnv only reads cfg.seed, cfg.weight_low,
        # cfg.weight_high, cfg.n_mc_samples, cfg.b, cfg.influence_model,
        # cfg.reward_type, cfg.gamma — all present on BudgetEnvConfig.
        super().__init__(graph, cfg)

        # Budget parameters (Idea 3 only)
        self.initial_budget: float = cfg.budget_B
        self.production_cost: float = cfg.production_cost
        self.B: float = self.initial_budget
        self.budget_history: List[float] = [self.B]

    # ── Reset ──────────────────────────────────────────────────────────────────

    @property
    def budget_fraction(self) -> float:
        """Remaining budget as fraction of initial budget B_t / B_0 ∈ [0, 1].

        Used as feature dim 20 in budget-aware LSTM training (Idea 3).
        Clipped to [0, 1]: profitable sales can push B above B_0 (frac > 1
        would give misleading signal), bankruptcy pins it at 0.
        """
        if self.initial_budget <= 0:
            return 1.0
        return float(max(0.0, min(1.0, self.B / self.initial_budget)))

    def reset(self) -> Dict:
        """Reset episode, including budget state.

        Returns:
            Initial observation from parent reset().
        """
        obs = super().reset()          # resets S, offered, t, caches, link_weights
        self.B = self.initial_budget
        self.budget_history = [self.B]
        return obs

    # ── Step ───────────────────────────────────────────────────────────────────

    def step(self, node_idx: int, discount: float) -> Tuple[Dict, float, bool, Dict]:
        """Execute one budget-constrained offer.

        Checks affordability BEFORE calling parent step.
        If unaffordable (B - c + price < 0): marks node as offered but skips
        production (no reward, no cost, no S update).
        If affordable: delegates to parent, then updates budget.

        Args:
            node_idx: Index of buyer node.
            discount: Discount fraction in [0,1].

        Returns:
            (obs, reward, done, info) — info includes "budget", "affordable".
        """
        node = self.nodes[node_idx]

        # Pre-compute price (same formula as parent) to check affordability.
        est_val    = self._estimate_valuation(node)
        price      = est_val * (1.0 - discount)
        affordable = (self.B - self.production_cost + price) >= -1e-9

        if not affordable:
            # Cannot produce: mark as offered but skip production entirely.
            self.offered.add(node)
            self.t += 1
            self.budget_history.append(self.B)
            done = self._check_bankrupt() or (len(self.offered) == self.n)
            obs  = self._get_observation()
            return obs, 0.0, done, {
                "node":       node,
                "node_idx":   node_idx,
                "accepted":   False,
                "affordable": False,
                "price":      0.0,
                "offered_price": 0.0,
                "true_val":   self._true_valuation(node),
                "discount":   discount,
                "revenue_step": 0.0,
                "total_revenue": self.total_revenue,
                "budget":     self.B,
            }

        # Affordable: delegate to parent for valuation, acceptance, S update.
        obs, reward, done, info = super().step(node_idx, discount)

        # Update budget on acceptance.
        if info["accepted"]:
            self.B = self.B - self.production_cost + info["offered_price"]

        self.budget_history.append(self.B)
        info["budget"]     = self.B
        info["affordable"] = True
        info["price"]      = info["offered_price"]   # convenience alias

        # Check bankruptcy in addition to parent's done condition.
        done = done or self._check_bankrupt()
        return obs, reward, done, info

    # ── Budget helpers ─────────────────────────────────────────────────────────

    def _check_bankrupt(self) -> bool:
        """Return True iff NO remaining buyer can pay enough to cover cost.

        Fast path: if B >= c, any price >= 0 is affordable → not bankrupt.
        Otherwise: bankrupt only if every remaining buyer has est_val < (c - B).
        Uses the cached MC estimates for speed.

        Returns:
            True if company is bankrupt (cannot make any viable offer).
        """
        if self.B >= self.production_cost - 1e-9:
            return False            # can always afford free offers

        min_price = self.production_cost - self.B   # minimum price needed
        for node in self.nodes:
            if node in self.offered:
                continue
            est_val = self._estimate_valuation(node)
            if est_val >= min_price - 1e-9:
                return False        # at least one buyer can cover cost
        return True

    def max_affordable_discount(self, node) -> float:
        """Maximum discount fraction still covering production cost.

        Returns:
            Discount in [0, 1] such that est_val*(1-d) >= (c - B), or -1 if
            no positive price can cover cost at current budget.
        """
        est_val = self._estimate_valuation(node)
        if est_val <= 0:
            return -1.0
        min_price = max(0.0, self.production_cost - self.B)
        if est_val < min_price - 1e-9:
            return -1.0
        return max(0.0, 1.0 - min_price / est_val)

    # get_current_influence: inherited from RevenueEnv (uses true _link_weights).
    # DO NOT override with _estimate_valuation here: the LSTM policy was trained
    # in RevenueEnv with TRUE influence features (dims 16-17). Overriding with
    # the MC estimate causes a 75-point revenue gap even at B=100 where budget
    # is not binding, because the policy sees different feature distributions
    # than it was trained on.
    # Baselines that want the sellers' noisy estimate should call
    # _estimate_valuation() directly.

    # ── Available nodes property ───────────────────────────────────────────────

    @property
    def available_nodes(self) -> List[int]:
        """Indices (not node ids) of buyers not yet offered."""
        return [i for i, n in enumerate(self.nodes) if n not in self.offered]
