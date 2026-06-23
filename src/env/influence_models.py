"""
src/env/influence_models.py

Rayleigh-based influence/valuation models from Babaei et al. (2013).

Buyer i's valuation given seed set S:
    v_i(S) = f_i(normalized_influence_i(S))

Normalized influence:
    x_i(S) = sum_{j in S∪{i}} w_ij / sum_{k in V} w_ik

Valuation function f (Rayleigh PDF with b=1, y=2x):
    f(y) = (y / b^2) * exp(-y^2 / (2*b^2))

Two variants:
    MonotoneInfluenceModel   : f clipped to non-decreasing (constant after peak at y=1)
    NonMonotoneInfluenceModel: raw Rayleigh PDF (decreasing after peak)
"""

import numpy as np
from typing import Set, Dict, List, Optional


class MonotoneInfluenceModel:
    """Monotone concave valuation: Rayleigh PDF clipped at peak.

    f(y) = (y/b^2) * exp(-y^2 / (2*b^2)), y = 2*x
    For y > 1 (x > 0.5), clipped to f(1) (non-decreasing).

    Args:
        b: Rayleigh scale parameter (default 1.0).
    """

    def __init__(self, b: float = 1.0) -> None:
        self.b = b
        self._peak = self._rayleigh(1.0)   # f at y=1 (x=0.5)

    def _rayleigh(self, y: float) -> float:
        """Rayleigh PDF value at y.

        Args:
            y: Input value (y = 2*x).

        Returns:
            f(y) = (y/b^2) * exp(-y^2 / (2*b^2)).
        """
        b2 = self.b ** 2
        return (y / b2) * np.exp(-(y ** 2) / (2 * b2))

    def __call__(self, x: float) -> float:
        """Compute monotone valuation at normalized influence x.

        Rayleigh PDF increases from 0 to y=1 (x=0.5), then decreases.
        Monotone version: clip the decreasing tail at the peak value.
          x <= 0.5 (y <= 1): return Rayleigh(y)   [increasing]
          x >  0.5 (y >  1): return peak = Rayleigh(1)  [constant]

        Args:
            x: Normalized influence in [0, 1].

        Returns:
            Buyer valuation (non-decreasing in x).
        """
        y = 2.0 * x
        if y > 1.0:
            return float(self._peak)
        return float(self._rayleigh(y))

    def batch(self, x_array: np.ndarray) -> np.ndarray:
        """Batch evaluate monotone valuation.

        Args:
            x_array: Array of normalized influence values, shape (n,).

        Returns:
            Array of valuations, shape (n,).
        """
        y = 2.0 * x_array
        b2 = self.b ** 2
        f_y = (y / b2) * np.exp(-(y ** 2) / (2 * b2))
        # Clip: for y > 1.0 (x > 0.5), use peak value
        return np.where(y > 1.0, self._peak, f_y).astype(np.float32)


class NonMonotoneInfluenceModel:
    """Non-monotone concave valuation: raw Rayleigh PDF.

    f(y) = (y/b^2) * exp(-y^2 / (2*b^2)), y = 2*x
    Peaks at y=1 (x=0.5), then decreases — models saturation.

    Args:
        b: Rayleigh scale parameter (default 1.0).
    """

    def __init__(self, b: float = 1.0) -> None:
        self.b = b

    def _rayleigh(self, y: float) -> float:
        """Rayleigh PDF value at y."""
        b2 = self.b ** 2
        return (y / b2) * np.exp(-(y ** 2) / (2 * b2))

    def __call__(self, x: float) -> float:
        """Compute non-monotone valuation at normalized influence x.

        Args:
            x: Normalized influence in [0, 1].

        Returns:
            Buyer valuation (peaks at x=0.5, decreases after).
        """
        y = 2.0 * x
        return float(self._rayleigh(y))

    def batch(self, x_array: np.ndarray) -> np.ndarray:
        """Batch evaluate non-monotone valuation.

        Args:
            x_array: Array of normalized influence values, shape (n,).

        Returns:
            Array of valuations, shape (n,).
        """
        y = 2.0 * x_array
        b2 = self.b ** 2
        f_y = (y / b2) * np.exp(-(y ** 2) / (2 * b2))
        return f_y.astype(np.float32)


def get_influence_model(model_type: str, b: float = 1.0):
    """Factory: return an influence model by name.

    Args:
        model_type: "monotone" or "non_monotone".
        b: Rayleigh scale parameter.

    Returns:
        MonotoneInfluenceModel or NonMonotoneInfluenceModel instance.

    Raises:
        ValueError: If model_type is not recognized.
    """
    if model_type == "monotone":
        return MonotoneInfluenceModel(b=b)
    elif model_type == "non_monotone":
        return NonMonotoneInfluenceModel(b=b)
    else:
        raise ValueError(
            f"Unknown influence model '{model_type}'. "
            "Supported: 'monotone', 'non_monotone'."
        )


def compute_normalized_influence(
    graph,
    node,
    S: Set,
    link_weights: Dict,
) -> float:
    """Compute normalized influence on node from seed set S.

    Implements: x_i(S) = sum_{j in S ∩ neighbors(i)} w_ij / sum_{k in neighbors(i)} w_ik

    Args:
        graph: NetworkX graph.
        node: Target node.
        S: Current seed set (buyers who purchased).
        link_weights: Dict mapping (u, v) → weight.

    Returns:
        Normalized influence in [0, 1]. Returns 0 if node is isolated.
    """
    neighbors = list(graph.neighbors(node))
    if not neighbors:
        return 0.0

    total_weight = sum(link_weights.get((node, nb), 0.0) for nb in neighbors)
    if total_weight == 0.0:
        return 0.0

    influence_from_S = sum(
        link_weights.get((node, j), 0.0)
        for j in S if j in set(neighbors)
    )
    return influence_from_S / total_weight


def sample_link_weights(
    graph,
    weight_low: float = 0.0,
    weight_high: float = 2.0,
    seed: Optional[int] = None,
) -> Dict:
    """Sample link weights from Uniform(weight_low, weight_high).

    As per Babaei et al. (2013): w_ij ~ Uniform(0, 2).
    For undirected graphs, w_ij = w_ji.

    Args:
        graph: NetworkX graph.
        weight_low: Lower bound of uniform distribution.
        weight_high: Upper bound of uniform distribution.
        seed: Random seed.

    Returns:
        Dict mapping (u, v) → float weight for all edges (both directions).
    """
    rng = np.random.default_rng(seed)
    weights = {}
    for u, v in graph.edges():
        w = rng.uniform(weight_low, weight_high)
        weights[(u, v)] = w
        if not graph.is_directed():
            weights[(v, u)] = w
    return weights
