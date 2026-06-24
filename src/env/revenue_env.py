"""
src/env/revenue_env.py

Revenue Maximization MDP Environment.

This is the core new component extending WSDM 2027 to the revenue maximization
setting of Babaei et al. (2013).

State:  GNN node features (20-dim) for all nodes
Action: (node_idx, discount) — joint discrete + continuous
Reward: actual price paid by buyer if they accept the offer, else 0

Idea 2 (NPV mode): reward is gamma^t * price_paid, where gamma < 1.

## Pricing model (Babaei et al. 2013, Section 5.3)

The seller knows only the DISTRIBUTION of link weights F_ij ~ Uniform(0,2),
NOT the exact realised values w_ij. Two distinct valuations therefore exist:

  _estimate_valuation(node)  — seller's noisy MC estimate, used to SET prices.
                                Fresh Uniform(0,2) samples per call (independent
                                of the true weights drawn at reset()).

  _true_valuation(node)      — buyer's actual willingness-to-pay, computed from
                                the single weight realisation drawn at reset().
                                Unknown to the seller; determines acceptance.

These are kept in separate caches (_est_val_cache, _true_val_cache) that are
invalidated for neighbours whenever a buyer is accepted.

_compute_valuation() is kept as an alias of _true_valuation() for backward
compatibility with existing tests.
"""

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RevenueEnvConfig:
    """Configuration for the revenue MDP environment."""
    influence_model: str = "monotone"   # "monotone" | "non_monotone"
    b: float = 1.0                       # Rayleigh parameter
    weight_low: float = 0.0             # link weight distribution lower bound
    weight_high: float = 2.0            # link weight distribution upper bound
    n_mc_samples: int = 200             # MC samples for seller's estimate
    reward_type: str = "flat"           # "flat" | "npv"
    gamma: float = 1.0                  # discount factor for NPV mode
    seed: int = 42


class RevenueEnv:
    """Revenue Maximization MDP.

    Wraps a social network graph as a sequential decision-making environment.
    At each step, the agent picks a buyer and sets a discount, then observes
    whether the buyer accepts and collects the resulting revenue.

    The key modelling distinction (Babaei et al. 2013, §5.3):
      - The SELLER estimates valuation via fresh MC samples → sets offered price.
      - The BUYER has a fixed true valuation (from reset()-time weights) → accepts
        iff true_valuation >= offered_price.

    Args:
        graph: NetworkX graph representing the social network.
        cfg: RevenueEnvConfig with environment hyperparameters.
    """

    def __init__(self, graph: nx.Graph, cfg: RevenueEnvConfig) -> None:
        self.graph = graph
        self.cfg = cfg
        self.n = graph.number_of_nodes()
        self.nodes = list(graph.nodes())
        self.node_to_idx = {v: i for i, v in enumerate(self.nodes)}
        self.rng = np.random.default_rng(cfg.seed)

        # True link weights (sampled once per episode at reset())
        self._link_weights: Dict[Tuple, float] = {}

        # Separate caches for the two valuation functions
        self._influence_cache: Dict[int, float] = {}   # legacy (kept for compat)
        self._true_val_cache: Dict = {}                # buyer's truth
        self._est_val_cache: Dict = {}                 # seller's MC estimate

        # Episode state
        self.S: set = set()             # buyers who purchased
        self.offered: set = set()       # buyers who were offered (accepted or not)
        self.t: int = 0                 # current step
        self.total_revenue: float = 0.0
        self.revenue_history: list = []

    def reset(self) -> Dict:
        """Reset environment for a new episode.

        Samples new link weights from Uniform(w_low, w_high).
        Seller knows this DISTRIBUTION but not the exact sampled values.

        Returns:
            Initial observation dict with graph and node features.
        """
        self.S = set()
        self.offered = set()
        self.t = 0
        self.total_revenue = 0.0
        self.revenue_history = []

        # Clear all caches
        self._influence_cache = {}
        self._true_val_cache = {}
        self._est_val_cache = {}

        # Sample new TRUE link weights for this episode
        self._link_weights = {}
        for u, v in self.graph.edges():
            w = self.rng.uniform(self.cfg.weight_low, self.cfg.weight_high)
            self._link_weights[(u, v)] = w
            if not self.graph.is_directed():
                self._link_weights[(v, u)] = w

        return self._get_observation()

    def step(self, node_idx: int, discount: float) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step: offer item to buyer node_idx at discounted price.

        Pricing is based on the SELLER'S noisy MC estimate of valuation.
        Acceptance is based on the BUYER'S TRUE valuation (unknown to seller).
        This correctly models Babaei et al. (2013) Section 5.3.

        Args:
            node_idx: Index of the buyer to offer (must be in 0..n-1, not yet offered).
            discount: Discount fraction in [0, 1]. Offered price = est_val * (1 - discount).

        Returns:
            observation: Updated state observation.
            reward: Revenue collected this step.
            done: Whether episode is finished.
            info: Diagnostic information dict.
        """
        node = self.nodes[node_idx]

        assert node not in self.offered, f"Node {node} already offered at step {self.t}"
        assert 0.0 <= discount <= 1.0, f"Discount must be in [0,1], got {discount}"

        # ── Seller estimates valuation (noisy) to set offered price ───────────
        estimated_val = self._estimate_valuation(node)
        offered_price = estimated_val * (1.0 - discount)

        # ── Buyer decision: accept iff TRUE valuation >= offered price ─────────
        # true_val uses the fixed link weights drawn at reset() — unknown to seller.
        # A buyer never refuses a free item (offered_price = 0 → always accepted).
        true_val = self._true_valuation(node)
        accepted = (true_val >= offered_price)

        if accepted:
            self.S.add(node)
            revenue_step = offered_price
            # Invalidate all caches for neighbours (their influence changed)
            for neighbor in self.graph.neighbors(node):
                self._influence_cache.pop(neighbor, None)
                self._true_val_cache.pop(neighbor, None)
                self._est_val_cache.pop(neighbor, None)
        else:
            revenue_step = 0.0

        self.offered.add(node)

        # ── Apply time discount (Idea 2 / NPV mode) ──────────────────────────
        if self.cfg.reward_type == "npv":
            reward = (self.cfg.gamma ** self.t) * revenue_step
        else:
            reward = revenue_step

        self.total_revenue += revenue_step  # always track flat revenue for metrics
        self.revenue_history.append({
            "step": self.t,
            "node": node,
            "estimated_val": estimated_val,
            "true_val": true_val,
            "discount": discount,
            "offered_price": offered_price,
            "accepted": accepted,
            "revenue": revenue_step,
        })

        self.t += 1
        done = (len(self.offered) == self.n)

        return self._get_observation(), reward, done, {
            "node": node,
            "estimated_val": estimated_val,
            "true_val": true_val,
            "valuation": true_val,          # legacy key — equals true_val
            "discount": discount,
            "offered_price": offered_price,
            "accepted": accepted,
            "revenue_step": revenue_step,
            "total_revenue": self.total_revenue,
        }

    # ── Valuation functions ────────────────────────────────────────────────────

    def _true_valuation(self, node) -> float:
        """Buyer's actual willingness-to-pay using true link weights from reset().

        Computed deterministically from the single weight realisation drawn at
        reset().  This value is UNKNOWN to the seller; it determines whether
        the buyer accepts the offered price.

        Cached per (node, S-state): invalidated for neighbours on acceptance.

        Args:
            node: The buyer node.

        Returns:
            True valuation in [0, f_peak].
        """
        if node in self._true_val_cache:
            return self._true_val_cache[node]

        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0

        total_weight = sum(
            self._link_weights.get((node, nb), 0.0) for nb in neighbors
        )
        if total_weight == 0:
            return 0.0

        influence_from_S = sum(
            self._link_weights.get((node, j), 0.0)
            for j in self.S if j in set(neighbors)
        )
        x = influence_from_S / total_weight
        val = self._apply_influence_model(x)
        self._true_val_cache[node] = val
        return val

    def _estimate_valuation(self, node, n_mc: int = None) -> float:
        """Seller's Monte Carlo estimate of buyer's valuation.

        Draws n_mc FRESH weight samples from Uniform(w_low, w_high),
        independent of the true weights (_link_weights).  This models
        Babaei et al. (2013) §5.3: the seller knows F_ij but not w_ij.

        Cached per (node, S-state): invalidated for neighbours on acceptance.

        Args:
            node: The buyer node.
            n_mc: Number of MC samples (defaults to cfg.n_mc_samples).

        Returns:
            Estimated valuation (expected willingness to pay under F_ij).
        """
        if node in self._est_val_cache:
            return self._est_val_cache[node]

        if n_mc is None:
            n_mc = self.cfg.n_mc_samples

        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0

        in_S = [j for j in neighbors if j in self.S]
        deg = len(neighbors)

        estimates = []
        for _ in range(n_mc):
            # Fresh weights independent of true _link_weights
            sampled_weights = self.rng.uniform(
                self.cfg.weight_low, self.cfg.weight_high, size=deg
            )
            total_w = sampled_weights.sum()
            if total_w == 0:
                estimates.append(0.0)
                continue
            # Influence = sum of weights for j in S / total weight
            # For efficiency: sum sampled_weights[i] for neighbor[i] in S
            influence = sum(
                sampled_weights[i]
                for i, nb in enumerate(neighbors)
                if nb in self.S
            ) / total_w
            estimates.append(self._apply_influence_model(influence))

        val = float(np.mean(estimates))
        self._est_val_cache[node] = val
        return val

    def _compute_valuation(self, node) -> float:
        """Backward-compatible alias for _true_valuation().

        Existing tests and code that call _compute_valuation() continue to work.
        New code should call _true_valuation() or _estimate_valuation() directly.

        Args:
            node: The buyer node.

        Returns:
            True valuation from fixed link weights.
        """
        return self._true_valuation(node)

    def _apply_influence_model(self, x: float) -> float:
        """Apply the Rayleigh-based valuation function f(x).

        Babaei et al. (2013) use the Rayleigh PDF with b=1, y=2x:
          f(y) = (y / b^2) * exp(-y^2 / (2*b^2))
        where y = 2*x (to peak at x=0.5 for non-monotone).

        Monotone variant: f is clipped to be non-decreasing (constant after peak).

        Args:
            x: Normalized influence in [0, 1].

        Returns:
            Buyer's valuation at this influence level.
        """
        b = self.cfg.b
        y = 2.0 * x  # scale so peak is at x=0.5 (y=1)
        f_y = (y / (b ** 2)) * np.exp(-(y ** 2) / (2 * b ** 2))

        if self.cfg.influence_model == "monotone":
            # Rayleigh peaks at y=1 (x=0.5), then decreases.
            # Monotone: clip the decreasing tail → constant at peak for y > 1.
            if y > 1.0:
                f_peak = (1.0 / (b ** 2)) * np.exp(-1.0 / (2 * b ** 2))
                return f_peak
            return f_y
        else:
            # Non-monotone: use raw Rayleigh (decreasing after peak)
            return f_y

    def get_current_influence(self, node) -> float:
        """Return current normalized influence on node from S.

        Uses true link weights (deterministic given the episode's weight sample).
        Used by feature computation in src/utils/features.py.

        Args:
            node: Target buyer node.

        Returns:
            Normalized influence in [0, 1].
        """
        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0
        total_weight = sum(self._link_weights.get((node, nb), 0.0) for nb in neighbors)
        if total_weight == 0:
            return 0.0
        influence_from_S = sum(
            self._link_weights.get((node, j), 0.0)
            for j in self.S if j in set(neighbors)
        )
        return influence_from_S / total_weight

    def _get_observation(self) -> Dict:
        """Construct observation dict for the current state.

        Returns:
            Dict with keys: graph, S, offered, t, n, node_list
        """
        return {
            "graph": self.graph,
            "S": frozenset(self.S),
            "offered": frozenset(self.offered),
            "t": self.t,
            "n": self.n,
            "nodes": self.nodes,
            "link_weights": self._link_weights,
        }

    @property
    def available_nodes(self) -> list:
        """Return list of buyer node indices not yet offered."""
        return [self.node_to_idx[v] for v in self.nodes if v not in self.offered]
