"""Unit tests for the inferential layer (ψ_P)."""

import numpy as np
import pytest
from aicf.model.inferential import InferentialLayer


class TestInferentialLayerConstructor:
    def test_valid_constructor(self):
        layer = InferentialLayer(w_prec=0.6, w_auto=0.4, tau_0=1.0, tau_max=8.0)
        assert layer.w_prec == 0.6
        assert layer.w_auto == 0.4

    def test_tau_max_must_exceed_tau_0(self):
        with pytest.raises(ValueError):
            InferentialLayer(tau_0=5.0, tau_max=5.0)

    def test_tau_0_must_be_positive(self):
        with pytest.raises(ValueError):
            InferentialLayer(tau_0=0.0, tau_max=5.0)


class TestPolicyPosterior:
    @pytest.fixture
    def layer(self):
        return InferentialLayer()

    def test_high_gamma_concentrates_posterior(self, layer):
        """High γ_eff → posterior concentrated on lowest EFE policy."""
        G = np.array([1.0, 5.0, 10.0])  # Policy 0 is best
        pi = layer.compute_policy_posterior(G, gamma_eff=20.0)
        assert pi[0] > 0.99, f"Expected mass on policy 0; got {pi[0]:.4f}"

    def test_zero_gamma_uniform(self, layer):
        """γ_eff = 0 → uniform posterior."""
        G = np.array([1.0, 100.0, -50.0])
        pi = layer.compute_policy_posterior(G, gamma_eff=0.0)
        np.testing.assert_allclose(pi, np.ones(3) / 3, atol=1e-10)

    def test_posterior_sums_to_one(self, layer):
        """Policy posterior always sums to 1."""
        rng = np.random.default_rng(10)
        for _ in range(50):
            G = rng.standard_normal(rng.integers(2, 20))
            pi = layer.compute_policy_posterior(G, gamma_eff=rng.uniform(0.1, 10))
            assert np.isclose(pi.sum(), 1.0, atol=1e-10)

    def test_negative_gamma_raises(self, layer):
        """γ_eff < 0 raises ValueError."""
        with pytest.raises(ValueError):
            layer.compute_policy_posterior(np.array([1.0, 2.0]), gamma_eff=-1.0)


class TestEffectivePrecision:
    @pytest.fixture
    def layer(self):
        return InferentialLayer(tau_0=2.0, tau_max=10.0)

    def test_increases_from_zero(self, layer):
        """γ_eff increases from 0 for small τ."""
        g0 = layer.compute_effective_precision(1.0, 0.01)
        g1 = layer.compute_effective_precision(1.0, 1.0)
        assert g1 > g0

    def test_decreases_after_optimum(self, layer):
        """γ_eff decreases beyond τ*."""
        tau_star = layer.optimal_planning_depth()
        g_star = layer.compute_effective_precision(1.0, tau_star)
        g_deep = layer.compute_effective_precision(1.0, tau_star * 5)
        assert g_star > g_deep

    def test_invalid_gamma_raises(self, layer):
        with pytest.raises(ValueError):
            layer.compute_effective_precision(gamma=0.0, tau=2.0)

    def test_invalid_tau_raises(self, layer):
        with pytest.raises(ValueError):
            layer.compute_effective_precision(gamma=1.0, tau=0.0)
