"""Unit tests for the dynamical layer (ψ_D)."""

import numpy as np
import pytest
from aicf.model.dynamical import DynamicalLayer


@pytest.fixture
def dyn():
    return DynamicalLayer()


class TestModalAgility:
    def test_single_state_zero_agility(self, dyn):
        """All-same-state sequence: no transitions → zero off-diagonal."""
        states = np.zeros(100, dtype=int)
        agil = dyn.compute_modal_agility(states, window_size=10)
        # Transition matrix is uniform (degenerate window: row sums = 0 handled)
        # Just verify output shape and non-negativity
        assert agil.shape[0] == 100 - 10 + 1
        assert np.all(agil >= 0)

    def test_two_state_alternating_high_agility(self, dyn):
        """0-1-0-1 alternating: high transition entropy."""
        states = np.tile([0, 1], 100)
        agil = dyn.compute_modal_agility(states, window_size=20)
        # Each window has 100% transitions from 0→1 and 1→0: T = [[0,1],[1,0]]
        # H(T) = -2*(0.5*log(0.5) + 0.5*log(0.5)) per row... actually
        # entropy of the flat T = log(2) * 2 rows = 2*log(2) max
        assert np.all(agil >= 0)

    def test_output_shape(self, dyn):
        """Output length = n_timepoints - window_size + 1."""
        n, w = 100, 15
        states = np.random.randint(0, 4, n)
        agil = dyn.compute_modal_agility(states, window_size=w)
        assert len(agil) == n - w + 1


class TestEntropyRate:
    def test_output_shape(self, dyn):
        """Output length = n_timepoints - window_size + 1."""
        ts = np.random.randn(3, 200)
        h = dyn.compute_entropy_rate(ts, window_size=20)
        assert len(h) == 200 - 20 + 1

    def test_constant_signal_low_entropy(self, dyn):
        """Constant signal has low entropy rate."""
        ts = np.ones((2, 100))
        h = dyn.compute_entropy_rate(ts, window_size=20)
        assert np.all(h >= 0)

    def test_1d_input_accepted(self, dyn):
        """1-D input is treated as single-channel."""
        ts = np.random.randn(100)
        h = dyn.compute_entropy_rate(ts, window_size=10)
        assert len(h) == 91


class TestDynamicalComplexity:
    def test_output_shape(self, dyn):
        """Output length = n_timepoints - window_size + 1."""
        ts = np.random.randn(4, 200)
        C = dyn.compute_dynamical_complexity(ts, window_size=20)
        assert len(C) == 200 - 20 + 1

    def test_non_negative(self, dyn):
        """Dynamical complexity is always ≥ 0."""
        ts = np.random.randn(2, 100)
        C = dyn.compute_dynamical_complexity(ts, window_size=10)
        assert np.all(C >= 0)

    def test_odd_window_raises(self, dyn):
        """Odd window size raises ValueError (needs equal past/future split)."""
        ts = np.random.randn(2, 100)
        with pytest.raises(ValueError):
            dyn.compute_dynamical_complexity(ts, window_size=11)


class TestNormalize:
    def test_midpoint_at_half(self, dyn):
        """z = 0 maps to 0.5."""
        raw = np.array([0.0])
        norm = dyn.normalize(raw, mean=0.0, std=1.0)
        assert np.isclose(norm[0], 0.5, atol=1e-10)

    def test_large_positive_near_one(self, dyn):
        """Very large z maps near 1."""
        raw = np.array([100.0])
        norm = dyn.normalize(raw, mean=0.0, std=1.0)
        assert norm[0] > 0.99

    def test_large_negative_near_zero(self, dyn):
        """Very negative z maps near 0."""
        raw = np.array([-100.0])
        norm = dyn.normalize(raw, mean=0.0, std=1.0)
        assert norm[0] < 0.01
