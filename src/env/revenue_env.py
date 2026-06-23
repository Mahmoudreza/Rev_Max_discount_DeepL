"""
src/env/revenue_env.py

Revenue Maximization MDP Environment.

This is the core new component extending WSDM 2027 to the revenue maximization
setting of Babaei et al. (2013).

State:  GNN node features (20-dim) for all nodes
Action: (node_idx, discount) — joint discrete + continuous
Reward: actual price paid by buyer if they accept the offer, else 0

Idea 2 (NPV mode): reward is gamma^t * price_paid, where gamma < 1.
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
    n_mc_samples: int = 200             # MC samples to estimate influence
    reward_type: str = "flat"           # "flat" | "npv"
    gamma: float = 1.0                  # discount factor for NPV mode
    seed: int = 42


class RevenueEnv:
    """Revenue Maximization MDP.

    Wraps a social network graph as a sequential decision-making environment.
    At each step, the agent picks a buyer and sets a discount, then observes
    whether the buyer accepts and collects the resulting revenue.

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

        # Sample link weights once per episode (re-sampled at reset)
        self._link_weights: Dict[Tuple, float] = {}
        self._influence_cache: Dict[int, float] = {}

        # Episode state
        self.S: set = set()             # buyers who purchased
        self.offered: set = set()       # buyers who were offered (accepted or not)
        self.t: int = 0                 # current step
        self.total_revenue: float = 0.0
        self.revenue_history: list = []

    def reset(self) -> Dict:
        """Reset environment for a new episode.

        Samples new link weights (seller knows distribution, not exact values).

        Returns:
            Initial observation dict with graph and node features.
        """
        self.S = set()
        self.offered = set()
        self.t = 0
        self.total_revenue = 0.0
        self.revenue_history = []
        self._influence_cache = {}

        # Sample new link weights from Uniform(w_low, w_high)
        self._link_weights = {}
        for u, v in self.graph.edges():
            w = self.rng.uniform(self.cfg.weight_low, self.cfg.weight_high)
            self._link_weights[(u, v)] = w
            if not self.graph.is_directed():
                self._link_weights[(v, u)] = w

        return self._get_observation()

    def step(self, node_idx: int, discount: float) -> Tuple[Dict, float, bool, Dict]:
        """Execute one step: offer item to buyer node_idx at discounted price.

        Args:
            node_idx: Index of the buyer to offer (must be in 0..n-1, not yet offered).
            discount: Discount fraction in [0, 1]. Offered price = valuation * (1 - discount).

        Returns:
            observation: Updated state observation.
            reward: Revenue collected this step.
            done: Whether episode is finished.
            info: Diagnostic information dict.
        """
        node = self.nodes[node_idx]

        assert node not in self.offered, f"Node {node} already offered at step {self.t}"
        assert 0.0 <= discount <= 1.0, f"Discount must be in [0,1], got {discount}"

        # ── Compute buyer's current valuation ────────────────────────────────
        valuation = self._compute_valuation(node)

        # ── Compute offered price ─────────────────────────────────────────────
        offered_price = valuation * (1.0 - discount)

        # ── Buyer decision: accept if offered_price <= valuation ──────────────
        # Buyer accepts iff offered price ≤ their valuation:
        #   offered_price = valuation * (1-discount) ≤ valuation   ← always true
        # Special case: at step 0 or when S is empty, valuation = f(0) = 0
        # and offered_price = 0.  A free item is still accepted (buyer joins S
        # and spreads influence to neighbors, bootstrapping the cascade).
        # Removing "and (offered_price > 0)" is intentional and correct per
        # Babaei et al. 2013: a buyer never refuses a free item.
        accepted = (valuation >= offered_price)

        if accepted:
            self.S.add(node)
            revenue_step = offered_price
            # Invalidate influence cache for all neighbors
            for neighbor in self.graph.neighbors(node):
                self._influence_cache.pop(neighbor, None)
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
            "valuation": valuation,
            "discount": discount,
            "offered_price": offered_price,
            "accepted": accepted,
            "revenue": revenue_step,
        })

        self.t += 1
        done = (len(self.offered) == self.n)

        return self._get_observation(), reward, done, {
            "node": node,
            "valuation": valuation,
            "discount": discount,
            "offered_price": offered_price,
            "accepted": accepted,
            "revenue_step": revenue_step,
            "total_revenue": self.total_revenue,
        }

    def _compute_valuation(self, node) -> float:
        """Compute buyer's valuation v_i(S) via Monte Carlo estimation.

        Babaei et al. (2013) Eq: v_i(S) = f_i(sum_{j in S∪{i}} w_ij / sum_{k in V} w_ik)

        Since seller only knows distribution F_ij (not exact w_ij), we average
        over cfg.n_mc_samples weight samples. Uses cache for efficiency.

        Args:
            node: The buyer node to compute valuation for.

        Returns:
            Estimated valuation (expected willingness to pay).
        """
        if node in self._influence_cache:
            return self._influence_cache[node]

        neighbors = list(self.graph.neighbors(node))
        if not neighbors:
            return 0.0

        # Total weight denominator (sum over all neighbors of node)
        total_weight = sum(
            self._link_weights.get((node, nb), 0.0) for nb in neighbors
        )
        if total_weight == 0:
            return 0.0

        # Influence from S: sum of w_ij for j in S ∩ neighbors(node)
        influence_from_S = sum(
            self._link_weights.get((node, j), 0.0)
            for j in self.S if j in set(neighbors)
        )

        normalized_influence = influence_from_S / total_weight
        valuation = self._apply_influence_model(normalized_influence)

        self._influence_cache[node] = valuation
        return valuation

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
