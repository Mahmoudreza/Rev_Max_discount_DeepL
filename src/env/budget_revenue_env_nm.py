"""src/env/budget_revenue_env_nm.py — Non-monotone acceptance variant.

BudgetRevenueEnvNM overrides step() to use Rayleigh/Gaussian acceptance:
  P(accept | price, w_true) = exp(-(price - w_true)^2 / (2*(w_true/2)^2))
  peaked at price = w_true; decreasing for both lower AND higher prices.

Default (monotone) behaviour unchanged when acceptance_mode="monotone".
Use with BudgetEnvConfig(acceptance_mode="rayleigh_nm").
"""
from __future__ import annotations
import math
from typing import Dict, List, Tuple

import numpy as np

from src.env.budget_revenue_env import BudgetRevenueEnv, BudgetEnvConfig


def _rayleigh_nm_accept_prob(price: float, w: float) -> float:
    """P(accept) ~ Gaussian peaked at w with sigma = w/2.
    Ensures non-monotone: both price>w and price<<w reduce acceptance.
    Falls back to 0.5 if w <= 0.
    """
    if w <= 0:
        return 0.5
    sigma = w / 2.0
    return math.exp(-((price - w) ** 2) / (2 * sigma ** 2))


class BudgetRevenueEnvNM(BudgetRevenueEnv):
    """Budget env with non-monotone (Rayleigh/Gaussian) acceptance.

    Acceptance: P(accept) = exp(-(price-w)^2 / (2*(w/2)^2))
    — peaks at price=w, falls for higher OR lower prices.
    All budget mechanics identical to parent.
    """

    def __init__(self, graph, cfg: BudgetEnvConfig) -> None:
        super().__init__(graph, cfg)
        self._rng = np.random.default_rng(cfg.seed + 9999)

    def step(self, node_idx: int, discount: float) -> Tuple[Dict, float, bool, Dict]:
        node = self.nodes[node_idx]
        est_val = self._estimate_valuation(node)
        price   = est_val * (1.0 - discount)
        affordable = (self.B - self.production_cost + price) >= -1e-9

        if not affordable:
            self.offered.add(node); self.t += 1
            self.budget_history.append(self.B)
            done = self._check_bankrupt() or (len(self.offered) == self.n)
            obs  = self._get_observation()
            return obs, 0.0, done, {"node": node, "node_idx": node_idx,
                "accepted": False, "affordable": False, "price": 0.0,
                "offered_price": 0.0, "true_val": self._true_valuation(node),
                "discount": discount, "revenue_step": 0.0,
                "total_revenue": self.total_revenue, "budget": self.B}

        # Non-monotone acceptance
        w_true = self._true_valuation(node)
        p_acc  = _rayleigh_nm_accept_prob(price, w_true)
        accepted = bool(self._rng.random() < p_acc)

        # Update state manually (bypass parent's acceptance logic)
        self.offered.add(node)
        self.t += 1
        reward = price if accepted else 0.0
        if accepted:
            self.total_revenue = getattr(self, "total_revenue", 0.0) + reward
            self.B = self.B - self.production_cost + price
            # Trigger influence spread in parent (if available)
            try:
                self._update_influence(node)
            except AttributeError:
                pass

        self.budget_history.append(self.B)
        done = self._check_bankrupt() or (len(self.offered) == self.n)
        obs  = self._get_observation()
        return obs, reward, done, {
            "node": node, "node_idx": node_idx,
            "accepted": accepted, "affordable": True,
            "price": price, "offered_price": price,
            "true_val": w_true, "discount": discount,
            "revenue_step": reward, "total_revenue": self.total_revenue,
            "budget": self.B, "p_accept": p_acc,
        }
