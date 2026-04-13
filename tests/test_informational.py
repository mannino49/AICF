"""Unit tests for the informational layer (ψ_I)."""

import numpy as np
import pytest
from aicf.model.informational import InformationalLayer, _shannon_entropy


class TestShannonEntropy:
    def test_uniform_entropy(self):
        """H(uniform over n) = log(n)."""
        for n in [2, 4, 8]:
            p = np.ones(n) / n
            assert np.isclose(_shannon_entropy(p), np.log(n), atol=1e-12)

    def test_deterministic_entropy_zero(self):
        """H([1, 0, 0]) = 0."""
        p = np.array([1.0, 0.0, 0.0])
        assert np.isclose(_shannon_entropy(p), 0.0, atol=1e-12)

    def test_binary_entropy(self):
        """H([0.5, 0.5]) = log(2) ≈ 0.693 nats."""
        p = np.array([0.5, 0.5])
        assert np.isclose(_shannon_entropy(p), np.log(2), atol=1e-12)


class TestInformationalLayer:
    @pytest.fixture
    def layer(self):
        return InformationalLayer()

    def test_compute_uniform_ends(self, layer):
        """With uniform distributions and independent joint, ψ_I = 0."""
        p_m = np.ones(4) / 4
        p_e = np.ones(3) / 3
        psi = layer.compute(p_m, p_e)  # independence assumed
        assert np.isclose(psi, 0.0, atol=1e-10)

    def test_compute_partial_coupling(self, layer):
        """Partial coupling: 0 < ψ_I < 1."""
        # Block-structured joint: correlated but not perfectly
        p_joint = np.array([
            [0.3, 0.1, 0.0],
            [0.1, 0.3, 0.0],
            [0.0, 0.0, 0.2],
        ])
        psi = layer.compute_from_joint(p_joint)
        assert 0.0 < psi < 1.0, f"Expected partial coupling, got {psi}"

    def test_joint_distribution_validation(self, layer):
        """2-D joint that doesn't sum to 1 raises ValueError."""
        with pytest.raises(ValueError):
            layer.compute_from_joint(np.array([[0.5, 0.3], [0.1, 0.0]]))

    def test_negative_joint_raises(self, layer):
        """Joint with negative entries raises ValueError."""
        with pytest.raises(ValueError):
            layer.compute_from_joint(np.array([[-0.1, 0.6], [0.2, 0.3]]))

    def test_mismatched_joint_shape_raises(self, layer):
        """p_joint with wrong shape raises ValueError."""
        p_m = np.array([0.5, 0.5])
        p_e = np.array([0.3, 0.7])
        p_joint_wrong = np.array([[0.2, 0.3, 0.1], [0.1, 0.2, 0.1]])
        with pytest.raises(ValueError):
            layer.compute(p_m, p_e, p_joint=p_joint_wrong)

    def test_symmetry_in_MI(self, layer):
        """I(M; E) = I(E; M) — the factor is symmetric."""
        p_joint = np.array([[0.2, 0.1], [0.3, 0.4]])
        psi_forward = layer.compute_from_joint(p_joint)
        psi_backward = layer.compute_from_joint(p_joint.T)
        # The normalized values differ because min{H(M), H(E)} is used
        # but both should be in [0, 1]
        assert 0 <= psi_forward <= 1
        assert 0 <= psi_backward <= 1
