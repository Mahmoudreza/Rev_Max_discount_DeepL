"""
tests/test_influence_models.py

Tests for Rayleigh influence/valuation models (src/env/influence_models.py).
"""

import pytest
import numpy as np

from src.env.influence_models import (
    MonotoneInfluenceModel,
    NonMonotoneInfluenceModel,
    get_influence_model,
    compute_normalized_influence,
    sample_link_weights,
)
import networkx as nx


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def small_graph():
    return nx.barabasi_albert_graph(15, 2, seed=1)


# ── MonotoneInfluenceModel tests ──────────────────────────────────────────────

class TestMonotoneInfluenceModel:
    def test_zero_influence(self):
        """f(0) = 0 (no influence → no valuation)."""
        model = MonotoneInfluenceModel(b=1.0)
        assert model(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_peak_at_half(self):
        """Monotone model: f(0.5) == f(1.0) (clipped at peak)."""
        model = MonotoneInfluenceModel(b=1.0)
        val_half = model(0.5)
        val_full = model(1.0)
        assert val_full == pytest.approx(val_half, rel=1e-6), (
            f"Monotone: f(1.0)={val_full} should equal f(0.5)={val_half} (clipped)"
        )

    def test_non_decreasing(self):
        """Monotone model is non-decreasing in x."""
        model = MonotoneInfluenceModel(b=1.0)
        xs = np.linspace(0, 1, 50)
        vals = [model(float(x)) for x in xs]
        for i in range(len(vals) - 1):
            assert vals[i] <= vals[i + 1] + 1e-9, (
                f"Monotone violated at x={xs[i]:.2f}: {vals[i]:.6f} > {vals[i+1]:.6f}"
            )

    def test_batch_matches_scalar(self):
        """batch() method matches scalar __call__ for all inputs."""
        model = MonotoneInfluenceModel(b=1.0)
        xs = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 1.0])
        batch_vals = model.batch(xs)
        scalar_vals = np.array([model(float(x)) for x in xs])
        np.testing.assert_array_almost_equal(batch_vals, scalar_vals, decimal=7)

    def test_factory_returns_monotone(self):
        """get_influence_model('monotone') returns MonotoneInfluenceModel."""
        model = get_influence_model("monotone", b=1.0)
        assert isinstance(model, MonotoneInfluenceModel)


# ── NonMonotoneInfluenceModel tests ──────────────────────────────────────────

class TestNonMonotoneInfluenceModel:
    def test_zero_influence(self):
        """f(0) = 0 (no influence → no valuation)."""
        model = NonMonotoneInfluenceModel(b=1.0)
        assert model(0.0) == pytest.approx(0.0, abs=1e-9)

    def test_peak_at_half(self):
        """Non-monotone model peaks at x=0.5 (y=1.0)."""
        model = NonMonotoneInfluenceModel(b=1.0)
        xs = np.linspace(0, 1, 1000)
        vals = [model(float(x)) for x in xs]
        peak_idx = int(np.argmax(vals))
        # Peak should be close to x=0.5
        assert abs(xs[peak_idx] - 0.5) < 0.02, (
            f"Peak should be near x=0.5, found at x={xs[peak_idx]:.3f}"
        )

    def test_decreases_after_peak(self):
        """Non-monotone: f(1.0) < f(0.5) (decreasing after peak)."""
        model = NonMonotoneInfluenceModel(b=1.0)
        assert model(1.0) < model(0.5), (
            f"Non-monotone: f(1.0)={model(1.0):.4f} should be < f(0.5)={model(0.5):.4f}"
        )

    def test_batch_matches_scalar(self):
        """batch() method matches scalar __call__ for all inputs."""
        model = NonMonotoneInfluenceModel(b=1.0)
        xs = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
        batch_vals = model.batch(xs)
        scalar_vals = np.array([model(float(x)) for x in xs])
        np.testing.assert_array_almost_equal(batch_vals, scalar_vals, decimal=7)

    def test_factory_returns_non_monotone(self):
        """get_influence_model('non_monotone') returns NonMonotoneInfluenceModel."""
        model = get_influence_model("non_monotone", b=1.0)
        assert isinstance(model, NonMonotoneInfluenceModel)


# ── Utility function tests ─────────────────────────────────────────────────────

class TestInfluenceUtilities:
    def test_normalized_influence_empty_S(self, small_graph):
        """compute_normalized_influence returns 0 when S is empty."""
        weights = sample_link_weights(small_graph, seed=0)
        for node in small_graph.nodes():
            infl = compute_normalized_influence(small_graph, node, set(), weights)
            assert infl == pytest.approx(0.0, abs=1e-9)

    def test_normalized_influence_in_range(self, small_graph):
        """compute_normalized_influence always in [0, 1]."""
        weights = sample_link_weights(small_graph, seed=0)
        nodes_list = list(small_graph.nodes())
        S = set(nodes_list[:5])
        for node in small_graph.nodes():
            infl = compute_normalized_influence(small_graph, node, S, weights)
            assert 0.0 <= infl <= 1.0 + 1e-9, (
                f"Node {node}: influence={infl} out of [0,1]"
            )

    def test_sample_link_weights_shape(self, small_graph):
        """sample_link_weights returns one weight per undirected edge (both dirs)."""
        weights = sample_link_weights(small_graph, seed=0)
        n_edges = small_graph.number_of_edges()
        # Each edge → 2 entries (both directions)
        assert len(weights) == 2 * n_edges, (
            f"Expected {2*n_edges} weight entries, got {len(weights)}"
        )

    def test_sample_link_weights_range(self, small_graph):
        """All sampled weights in [weight_low, weight_high]."""
        low, high = 0.0, 2.0
        weights = sample_link_weights(small_graph, weight_low=low, weight_high=high, seed=0)
        for (u, v), w in weights.items():
            assert low <= w <= high, f"Weight ({u},{v})={w} out of [{low},{high}]"

    def test_get_influence_model_invalid(self):
        """get_influence_model raises ValueError for unknown model type."""
        with pytest.raises(ValueError):
            get_influence_model("invalid_model")
