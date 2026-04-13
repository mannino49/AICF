"""Unit tests for the simulation engine and parameter sweep."""

import numpy as np
import pytest
from aicf.model.flow_model import FlowModel
from aicf.model.reduced_models import FullModel, InformationOnlyModel, MODEL_REGISTRY
from aicf.simulation.engine import SimulationEngine, SimulationResult
from aicf.simulation.parameter_sweep import ParameterSweep


@pytest.fixture
def engine():
    return SimulationEngine()


@pytest.fixture
def simple_model():
    return FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                     sigma_F=0.1, seed=0)


class TestSimulationEngine:
    def test_result_trajectories_shape(self, engine, simple_model):
        """trajectories has shape (n_trials, n_steps + 1)."""
        n_steps, n_trials = 100, 20
        res = engine.run(simple_model,
                         inputs={"psi_I": 0.5, "psi_P": 0.5, "psi_D": 0.5},
                         n_steps=n_steps, n_trials=n_trials, seed=0)
        assert res.trajectories.shape == (n_trials, n_steps + 1)

    def test_mean_trajectory_shape(self, engine, simple_model):
        """mean_trajectory has length n_steps + 1."""
        res = engine.run(simple_model,
                         inputs={"psi_I": 0.5, "psi_P": 0.5, "psi_D": 0.5},
                         n_steps=50, n_trials=10, seed=1)
        assert len(res.mean_trajectory) == 51

    def test_theoretical_steady_state(self, engine, simple_model):
        """steady_state_theoretical matches FlowModel.steady_state."""
        psi_I, psi_P, psi_D = 0.6, 0.7, 0.8
        res = engine.run(simple_model,
                         inputs={"psi_I": psi_I, "psi_P": psi_P, "psi_D": psi_D},
                         n_steps=100, n_trials=5, seed=2)
        expected = simple_model.steady_state(psi_I, psi_P, psi_D)
        assert np.isclose(res.steady_state_theoretical, expected, atol=1e-10)

    def test_reproducibility(self, engine, simple_model):
        """Same seed produces identical results."""
        r1 = engine.run(simple_model,
                        inputs={"psi_I": 0.5, "psi_P": 0.5, "psi_D": 0.5},
                        n_steps=50, n_trials=10, seed=42)
        r2 = engine.run(simple_model,
                        inputs={"psi_I": 0.5, "psi_P": 0.5, "psi_D": 0.5},
                        n_steps=50, n_trials=10, seed=42)
        np.testing.assert_array_equal(r1.trajectories, r2.trajectories)

    def test_empirical_steady_state_close_to_theoretical(self, engine):
        """For long deterministic simulation, empirical ≈ theoretical F*."""
        m = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                      sigma_F=0.0, seed=0)
        psi_I, psi_P, psi_D = 0.5, 0.6, 0.7
        res = engine.run(m,
                         inputs={"psi_I": psi_I, "psi_P": psi_P, "psi_D": psi_D},
                         n_steps=5000, n_trials=1, seed=0)
        theoretical = m.steady_state(psi_I, psi_P, psi_D)
        empirical = res.empirical_steady_state(last_fraction=0.2)
        assert np.isclose(empirical, theoretical, atol=0.05)


class TestSimulationResult:
    def test_time_axis(self):
        """time axis has correct length and spacing."""
        traj = np.zeros((5, 201))
        res = SimulationResult(
            trajectories=traj,
            mean_trajectory=np.zeros(201),
            std_trajectory=np.zeros(201),
            steady_state_theoretical=1.0,
            inputs={},
            dt=0.01,
        )
        assert len(res.time) == 201
        assert np.isclose(res.time[-1], 200 * 0.01)

    def test_summary_keys(self):
        """summary() returns dict with expected keys."""
        traj = np.ones((10, 101))
        res = SimulationResult(
            trajectories=traj,
            mean_trajectory=np.ones(101),
            std_trajectory=np.zeros(101),
            steady_state_theoretical=2.5,
            inputs={"psi_I": 0.5},
            dt=0.01,
        )
        s = res.summary()
        for key in ["n_trials", "n_steps", "dt", "steady_state_theoretical",
                    "steady_state_empirical"]:
            assert key in s


class TestReducedModels:
    def test_all_models_in_registry(self):
        """All seven model types are registered."""
        expected = {
            "information_only", "inference_only", "dynamics_only",
            "information_inference", "information_dynamics",
            "inference_dynamics", "full",
        }
        assert set(MODEL_REGISTRY.keys()) == expected

    def test_information_only_has_zero_betas(self):
        m = InformationOnlyModel(beta_I=2.0)
        assert m.beta_I == 2.0
        assert m.beta_P == 0.0
        assert m.beta_D == 0.0

    def test_full_model_shares_interface(self):
        """FullModel has identical interface to FlowModel."""
        m = FullModel(beta_I=0.5, beta_P=0.5, beta_D=0.5, sigma_F=0.0)
        traj = m.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=10)
        assert len(traj) == 11

    def test_reduced_models_lower_steady_state(self):
        """Ablation models reach lower F* than the full model."""
        from aicf.model.reduced_models import InformationOnlyModel, FullModel
        psi = 0.8
        m_full = FullModel(sigma_F=0.0)
        m_info = InformationOnlyModel(sigma_F=0.0)
        F_full = m_full.steady_state(psi, psi, psi)
        F_info = m_info.steady_state(psi, psi, psi)
        assert F_full > F_info
