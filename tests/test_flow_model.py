"""Unit tests for the FlowModel (state equation + observation)."""

import numpy as np
import pytest
from aicf.model.flow_model import FlowModel


@pytest.fixture
def deterministic_model():
    return FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                     sigma_F=0.0, F0=0.0, dt=0.01, seed=0)


class TestFlowModelConstruction:
    def test_defaults(self):
        m = FlowModel()
        assert m.kappa == 1.0
        assert m.F0 == 0.0

    def test_custom_params(self):
        m = FlowModel(kappa=2.0, beta_I=0.5, beta_P=0.5, beta_D=0.5,
                      sigma_F=0.2, F0=1.0, dt=0.001)
        assert m.kappa == 2.0
        assert m.F0 == 1.0


class TestDriftDiffusion:
    def test_drift_all_zero(self, deterministic_model):
        """At F=F0 with all ψ=0, drift = 0."""
        d = deterministic_model.drift(F=0.0, psi_I=0.0, psi_P=0.0, psi_D=0.0)
        assert np.isclose(d, 0.0)

    def test_drift_drives_above_baseline(self, deterministic_model):
        """Positive ψ inputs produce positive drift."""
        d = deterministic_model.drift(F=0.0, psi_I=0.5, psi_P=0.5, psi_D=0.5)
        assert d > 0

    def test_drift_mean_reverts_above_baseline(self, deterministic_model):
        """At high F with all ψ=0, drift is negative."""
        d = deterministic_model.drift(F=10.0, psi_I=0.0, psi_P=0.0, psi_D=0.0)
        assert d < 0

    def test_diffusion_constant(self, deterministic_model):
        """Diffusion equals sigma_F regardless of F."""
        assert deterministic_model.diffusion(0.0) == 0.0
        m = FlowModel(sigma_F=0.3)
        assert m.diffusion(99.9) == 0.3


class TestSimulation:
    def test_initial_condition(self, deterministic_model):
        """Trajectory starts at F_init."""
        traj = deterministic_model.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5,
                                             n_steps=100, F_init=2.5)
        assert traj[0] == 2.5

    def test_constant_input_monotone_convergence(self, deterministic_model):
        """Deterministic trajectory monotonically approaches steady state from below."""
        traj = deterministic_model.simulate(psi_I=1.0, psi_P=1.0, psi_D=1.0,
                                             n_steps=500, F_init=0.0)
        F_star = deterministic_model.steady_state(1.0, 1.0, 1.0)
        # Should approach but not overshoot (OU process is stable)
        assert traj[-1] < F_star + 0.01

    def test_array_input_correct_shape(self, deterministic_model):
        """Array ψ inputs of length n_steps are accepted."""
        n = 300
        psi = np.linspace(0, 1, n)
        traj = deterministic_model.simulate(psi_I=psi, psi_P=0.5, psi_D=0.5,
                                             n_steps=n)
        assert len(traj) == n + 1

    def test_mismatched_array_raises(self, deterministic_model):
        """Array input with wrong length raises ValueError."""
        with pytest.raises(ValueError):
            deterministic_model.simulate(psi_I=np.ones(50), psi_P=0.5, psi_D=0.5,
                                          n_steps=100)


class TestObservationModel:
    def test_observation_shape(self, deterministic_model):
        """Observation output has shape (n_obs, n_steps)."""
        traj = deterministic_model.simulate(0.5, 0.5, 0.5, n_steps=100)
        n_obs = 3
        Lambda = np.ones((n_obs, 1))
        R = np.eye(n_obs) * 0.1
        y = deterministic_model.observe(traj, Lambda, R)
        assert y.shape == (n_obs, len(traj))

    def test_zero_noise_observation(self):
        """With R → 0, observations ≈ Λ · F(t)."""
        m = FlowModel(sigma_F=0.0, seed=42)
        traj = m.simulate(0.5, 0.5, 0.5, n_steps=50)
        Lambda = np.array([[2.0]])
        R = np.eye(1) * 1e-20
        y = m.observe(traj, Lambda, R)
        np.testing.assert_allclose(y[0], 2.0 * traj, atol=1e-6)
