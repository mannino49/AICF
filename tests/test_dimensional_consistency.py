"""
Dimensional Consistency Tests
================================

The most critical test file — verifies the mathematical correctness and
dimensional consistency of the AICF model.

All tests in this file correspond directly to the verification checklist
in codebase_build_instructions.md and the invariants stated in
formalism_repair_v1.md.
"""

import numpy as np
import pytest

from aicf.model.informational import InformationalLayer
from aicf.model.inferential import InferentialLayer
from aicf.model.dynamical import DynamicalLayer
from aicf.model.flow_model import FlowModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def info():
    return InformationalLayer()


@pytest.fixture
def infer():
    return InferentialLayer(w_prec=0.5, w_auto=0.5, tau_0=2.0, tau_max=10.0)


@pytest.fixture
def dyn():
    return DynamicalLayer()


@pytest.fixture
def model():
    return FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                     sigma_F=0.0, F0=0.0, dt=0.01, seed=42)


# ---------------------------------------------------------------------------
# ψ_I tests (§2)
# ---------------------------------------------------------------------------

class TestPsiI:
    def test_psi_I_range(self, info):
        """ψ_I always in [0, 1] for valid probability distributions."""
        rng = np.random.default_rng(0)
        for _ in range(100):
            n_m, n_e = rng.integers(2, 10), rng.integers(2, 10)
            p_m = rng.dirichlet(np.ones(n_m))
            p_e = rng.dirichlet(np.ones(n_e))
            psi = info.compute(p_m, p_e)
            assert 0.0 <= psi <= 1.0, f"ψ_I = {psi} out of [0, 1]"

    def test_psi_I_boundary_independent(self, info):
        """ψ_I = 0 when M and E are independent (joint = outer product)."""
        p_m = np.array([0.5, 0.5])
        p_e = np.array([0.3, 0.7])
        # With independence assumption (default), I(M; E) = 0
        psi = info.compute(p_m, p_e)
        assert np.isclose(psi, 0.0, atol=1e-10), f"Expected 0, got {psi}"

    def test_psi_I_boundary_perfect_coupling(self, info):
        """ψ_I = 1 when M perfectly determines E (and vice versa)."""
        # Perfect coupling: P(M=0, E=0) = 0.4, P(M=1, E=1) = 0.6
        p_joint = np.array([[0.4, 0.0],
                             [0.0, 0.6]])
        psi = info.compute_from_joint(p_joint)
        assert np.isclose(psi, 1.0, atol=1e-10), f"Expected 1, got {psi}"

    def test_psi_I_boundary_degenerate_deterministic_M(self, info):
        """ψ_I = 0 when H(M) = 0 (M is deterministic)."""
        p_m = np.array([1.0, 0.0])
        p_e = np.array([0.5, 0.5])
        psi = info.compute(p_m, p_e)
        assert np.isclose(psi, 0.0, atol=1e-10), f"Expected 0, got {psi}"

    def test_psi_I_boundary_degenerate_deterministic_E(self, info):
        """ψ_I = 0 when H(E) = 0 (E is deterministic)."""
        p_m = np.array([0.5, 0.5])
        p_e = np.array([0.0, 1.0])
        psi = info.compute(p_m, p_e)
        assert np.isclose(psi, 0.0, atol=1e-10), f"Expected 0, got {psi}"

    def test_psi_I_from_joint_equals_compute(self, info):
        """compute_from_joint and compute agree when using independence."""
        p_m = np.array([0.3, 0.4, 0.3])
        p_e = np.array([0.6, 0.4])
        p_joint = np.outer(p_m, p_e)

        psi_a = info.compute(p_m, p_e)
        psi_b = info.compute_from_joint(p_joint)
        assert np.isclose(psi_a, psi_b, atol=1e-12)

    def test_psi_I_invalid_distribution_raises(self, info):
        """ValueError for invalid distributions."""
        with pytest.raises(ValueError):
            info.compute(np.array([0.3, 0.3]), np.array([0.5, 0.5]))  # doesn't sum to 1
        with pytest.raises(ValueError):
            info.compute(np.array([-0.1, 1.1]), np.array([0.5, 0.5]))  # negative


# ---------------------------------------------------------------------------
# Policy negentropy tests (§3.2)
# ---------------------------------------------------------------------------

class TestNegentropy:
    def test_negentropy_range(self, infer):
        """n_π always in [0, 1]."""
        rng = np.random.default_rng(1)
        for _ in range(100):
            n = rng.integers(2, 20)
            pi = rng.dirichlet(np.ones(n))
            n_pi = infer.compute_negentropy(pi)
            assert 0.0 <= n_pi <= 1.0, f"n_π = {n_pi} out of [0, 1]"

    def test_negentropy_uniform(self, infer):
        """n_π = 0 for uniform policy posterior (maximum entropy)."""
        pi = np.ones(8) / 8
        n_pi = infer.compute_negentropy(pi)
        assert np.isclose(n_pi, 0.0, atol=1e-10)

    def test_negentropy_point_mass(self, infer):
        """n_π = 1 for deterministic policy (zero entropy)."""
        pi = np.array([1.0, 0.0, 0.0, 0.0])
        n_pi = infer.compute_negentropy(pi)
        assert np.isclose(n_pi, 1.0, atol=1e-10)

    def test_negentropy_single_policy_boundary(self, infer):
        """n_π = 1 when N_π = 1 (boundary convention §3.2)."""
        pi = np.array([1.0])
        n_pi = infer.compute_negentropy(pi)
        assert n_pi == 1.0

    def test_negentropy_monotone_in_entropy(self, infer):
        """n_π decreases as policy entropy increases."""
        rng = np.random.default_rng(2)
        n = 10
        alphas = [0.1, 0.5, 1.0, 5.0, 20.0]  # increasing concentration → decreasing entropy
        n_pis = []
        for alpha in alphas:
            pi = rng.dirichlet(alpha * np.ones(n))
            # For consistency, set concentration directly
            pi_fixed = np.full(n, 1.0 / n)  # start uniform
            # Use deterministic concentration
        # Use analytically controlled distributions
        vals = [
            infer.compute_negentropy(np.array([0.9, 0.05, 0.05])),
            infer.compute_negentropy(np.array([0.5, 0.3, 0.2])),
            infer.compute_negentropy(np.array([1/3, 1/3, 1/3])),
        ]
        assert vals[0] > vals[1] > vals[2], "n_π should decrease as distribution spreads"


# ---------------------------------------------------------------------------
# Automaticity tests (§3.2)
# ---------------------------------------------------------------------------

class TestAutomaticity:
    def test_automaticity_range(self, infer):
        """A(t) always in [0, 1]."""
        rng = np.random.default_rng(3)
        for _ in range(100):
            n = rng.integers(2, 15)
            pi_post = rng.dirichlet(np.ones(n))
            pi_habit = rng.dirichlet(np.ones(n))
            A = infer.compute_automaticity(pi_post, pi_habit)
            assert 0.0 <= A <= 1.0, f"A = {A} out of [0, 1]"

    def test_automaticity_identical(self, infer):
        """A = 1 when posterior equals habitual prior (KL = 0)."""
        pi = np.array([0.2, 0.5, 0.3])
        A = infer.compute_automaticity(pi, pi.copy())
        assert np.isclose(A, 1.0, atol=1e-10)

    def test_automaticity_maximally_different(self, infer):
        """A = 0 when posterior is maximally divergent from prior."""
        # Put all mass on different policies: max KL ~ log(N)
        n = 5
        pi_post = np.array([1.0] + [0.0] * (n - 1))
        # π_habit puts equal probability on all but the first policy
        eps = 1e-9  # avoid infinite KL
        pi_habit = np.array([eps] + [(1.0 - eps) / (n - 1)] * (n - 1))
        pi_habit /= pi_habit.sum()
        A = infer.compute_automaticity(pi_post, pi_habit)
        assert A < 0.1, f"Expected A near 0 for maximally divergent distributions, got {A}"

    def test_automaticity_single_policy_boundary(self, infer):
        """A = 1 when N_π = 1 (boundary convention §3.2)."""
        A = infer.compute_automaticity(np.array([1.0]), np.array([1.0]))
        assert A == 1.0

    def test_automaticity_length_mismatch_raises(self, infer):
        """ValueError when posterior and prior have different lengths."""
        with pytest.raises(ValueError):
            infer.compute_automaticity(np.array([0.5, 0.5]), np.array([1/3, 1/3, 1/3]))


# ---------------------------------------------------------------------------
# ψ_P composite tests (§3.2)
# ---------------------------------------------------------------------------

class TestPsiP:
    def test_psi_P_range(self, infer):
        """ψ_P always in [0, 1]."""
        rng = np.random.default_rng(4)
        for _ in range(100):
            n = rng.integers(2, 10)
            pi_post = rng.dirichlet(np.ones(n))
            pi_habit = rng.dirichlet(np.ones(n))
            psi_P = infer.compute(pi_post, pi_habit)
            assert 0.0 <= psi_P <= 1.0

    def test_psi_P_max_at_both_high(self, infer):
        """ψ_P is highest when both n_π and A are high (flow condition)."""
        n = 5
        pi_flow = np.array([0.95, 0.025, 0.025 / 3, 0.025 / 3, 0.025 / 3])
        pi_flow /= pi_flow.sum()

        # High n_π + high A: posterior ≈ habit (flow)
        psi_flow = infer.compute(pi_flow, pi_flow.copy())

        # High n_π + low A: posterior very different from uniform habit
        pi_habit_uniform = np.ones(n) / n
        psi_deliberate = infer.compute(pi_flow, pi_habit_uniform)

        assert psi_flow > psi_deliberate

    def test_psi_P_weights_sum_to_1_constructor(self):
        """Constructor raises ValueError if weights don't sum to 1."""
        with pytest.raises(ValueError):
            InferentialLayer(w_prec=0.6, w_auto=0.6)

    def test_psi_P_negative_weight_raises(self):
        """Constructor raises ValueError for negative weights."""
        with pytest.raises(ValueError):
            InferentialLayer(w_prec=-0.1, w_auto=1.1)


# ---------------------------------------------------------------------------
# Planning depth and γ_eff tests (§5)
# ---------------------------------------------------------------------------

class TestPlanningDepth:
    def test_gamma_eff_has_interior_maximum(self, infer):
        """γ_eff(τ) has exactly one maximum at τ* for τ > 0."""
        taus = np.linspace(0.01, 30.0, 1000)
        gamma_eff = np.array([infer.compute_effective_precision(1.0, t) for t in taus])
        peak_idx = np.argmax(gamma_eff)
        # Peak should not be at the boundary
        assert 0 < peak_idx < len(taus) - 1, "γ_eff peak is at a boundary (no interior maximum)"

    def test_optimal_planning_depth_formula(self, infer):
        """τ* from formula matches numerical argmax of γ_eff."""
        tau_star_analytic = infer.optimal_planning_depth()
        taus = np.linspace(0.01, 50.0, 10000)
        gamma_eff = np.array([infer.compute_effective_precision(5.0, t) for t in taus])
        tau_star_numeric = taus[np.argmax(gamma_eff)]
        assert np.isclose(tau_star_analytic, tau_star_numeric, atol=0.1), (
            f"Analytic τ* = {tau_star_analytic:.4f}, numeric τ* = {tau_star_numeric:.4f}"
        )

    def test_gamma_eff_positive(self, infer):
        """γ_eff is always positive for valid inputs."""
        for gamma in [0.1, 1.0, 5.0, 10.0]:
            for tau in [0.5, 2.0, 5.0, 10.0, 20.0]:
                g_eff = infer.compute_effective_precision(gamma, tau)
                assert g_eff > 0, f"γ_eff = {g_eff} for γ={gamma}, τ={tau}"

    def test_gamma_eff_scales_with_gamma(self, infer):
        """γ_eff is proportional to base precision γ."""
        tau = infer.optimal_planning_depth()
        g1 = infer.compute_effective_precision(1.0, tau)
        g2 = infer.compute_effective_precision(2.0, tau)
        assert np.isclose(g2 / g1, 2.0, atol=1e-10)

    def test_tau_max_greater_than_tau_0_required(self):
        """Constructor raises ValueError if τ_max ≤ τ₀."""
        with pytest.raises(ValueError):
            InferentialLayer(tau_0=5.0, tau_max=3.0)

    def test_softmax_numerical_stability(self, infer):
        """Softmax doesn't overflow/underflow with extreme inputs."""
        # Very large values
        G_large = np.array([1e10, 1e10 + 1, 1e10 - 1])
        pi = infer.compute_policy_posterior(G_large, gamma_eff=1.0)
        assert np.all(np.isfinite(pi)), "Softmax produced non-finite values"
        assert np.isclose(pi.sum(), 1.0, atol=1e-10)

        # Very small (negative) values
        G_small = np.array([-1e10, -1e10 + 1, -1e10 - 1])
        pi = infer.compute_policy_posterior(G_small, gamma_eff=1.0)
        assert np.all(np.isfinite(pi))
        assert np.isclose(pi.sum(), 1.0, atol=1e-10)

    def test_softmax_zero_gamma_gives_uniform(self, infer):
        """γ_eff = 0 → uniform policy posterior."""
        G = np.array([1.0, 2.0, 3.0, 4.0])
        pi = infer.compute_policy_posterior(G, gamma_eff=0.0)
        expected = np.ones(4) / 4
        np.testing.assert_allclose(pi, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# ψ_D tests (§4)
# ---------------------------------------------------------------------------

class TestPsiD:
    def test_psi_D_range_scalar(self, dyn):
        """ψ_D always in [0, 1] for scalar inputs."""
        rng = np.random.default_rng(5)
        for _ in range(50):
            h = rng.normal(0, 1)
            C = rng.normal(0, 1)
            a = rng.normal(0, 1)
            psi_D = dyn.compute(
                np.array([h]), np.array([C]), np.array([a])
            )
            assert 0.0 <= psi_D <= 1.0, f"ψ_D = {psi_D} out of [0, 1]"

    def test_psi_D_range_array(self, dyn):
        """ψ_D always in [0, 1] for array inputs."""
        rng = np.random.default_rng(6)
        h = rng.normal(0, 2, 50)
        C = rng.normal(0, 2, 50)
        a = rng.normal(0, 2, 50)
        psi_D = dyn.compute(h, C, a)
        assert np.all(psi_D >= 0.0) and np.all(psi_D <= 1.0)

    def test_psi_D_weights_sum_to_1_constructor(self):
        """Constructor raises ValueError if weights don't sum to 1."""
        with pytest.raises(ValueError):
            DynamicalLayer(w_h=0.5, w_C=0.5, w_agil=0.5)

    def test_psi_D_normalize_output_range(self, dyn):
        """normalize() maps arbitrary floats to (0, 1) via z-score + sigmoid."""
        raw = np.array([-10.0, -1.0, 0.0, 1.0, 10.0])
        norm = dyn.normalize(raw)
        assert np.all(norm > 0.0) and np.all(norm < 1.0)

    def test_psi_D_normalize_constant_gives_half(self, dyn):
        """Constant input (zero std) maps to 0.5 (sigmoid(0))."""
        raw = np.ones(20) * 3.5
        norm = dyn.normalize(raw)
        np.testing.assert_allclose(norm, 0.5, atol=1e-10)

    def test_modal_agility_range(self, dyn):
        """Modal agility is non-negative."""
        rng = np.random.default_rng(7)
        states = rng.integers(0, 4, size=200)
        agility = dyn.compute_modal_agility(states, window_size=20)
        assert np.all(agility >= 0.0)

    def test_entropy_rate_positive(self, dyn):
        """Entropy rate is non-negative."""
        rng = np.random.default_rng(8)
        ts = rng.standard_normal((4, 200))
        h = dyn.compute_entropy_rate(ts, window_size=20)
        assert np.all(h >= 0.0)


# ---------------------------------------------------------------------------
# Flow model (state equation) tests (§1.2, §6)
# ---------------------------------------------------------------------------

class TestFlowModel:
    def test_drift_units_scale_correctly(self, model):
        """Drift scales correctly with κ, β, ψ — dimensional check."""
        # With κ=1, β=1, F=0, F0=0:  drift = 0 + ψ_I + ψ_P + ψ_D
        d = model.drift(F=0.0, psi_I=0.3, psi_P=0.4, psi_D=0.5)
        assert np.isclose(d, 0.3 + 0.4 + 0.5, atol=1e-12)

    def test_drift_mean_reversion_term(self, model):
        """Drift mean-reversion term: −κ(F − F₀)."""
        # kappa=1, F0=0, all psi=0: drift = −F
        d = model.drift(F=2.0, psi_I=0.0, psi_P=0.0, psi_D=0.0)
        assert np.isclose(d, -2.0, atol=1e-12)

    def test_steady_state_formula(self, model):
        """F* = F₀ + (β_I·ψ_I + β_P·ψ_P + β_D·ψ_D) / κ."""
        psi_I, psi_P, psi_D = 0.6, 0.7, 0.8
        F_star = model.steady_state(psi_I, psi_P, psi_D)
        expected = 0.0 + (1.0 * 0.6 + 1.0 * 0.7 + 1.0 * 0.8) / 1.0
        assert np.isclose(F_star, expected, atol=1e-12)

    def test_mean_reversion(self):
        """With all ψ = 0, F(t) → F₀ over time (deterministic)."""
        m = FlowModel(kappa=2.0, sigma_F=0.0, F0=0.0, dt=0.01, seed=0)
        traj = m.simulate(psi_I=0.0, psi_P=0.0, psi_D=0.0, n_steps=500,
                          F_init=3.0)
        # After 5 time units with κ=2, F should decay to near 0
        assert abs(traj[-1]) < 0.05, f"F did not decay to baseline; F_final = {traj[-1]:.4f}"

    def test_flow_requires_input(self):
        """Without sustained ψ inputs, flow decays to baseline (F₀)."""
        m = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                      sigma_F=0.0, F0=0.0, dt=0.01, seed=0)
        # First drive to high F, then remove input
        traj_on = m.simulate(psi_I=0.8, psi_P=0.8, psi_D=0.8, n_steps=1000)
        F_high = traj_on[-1]
        traj_off = m.simulate(psi_I=0.0, psi_P=0.0, psi_D=0.0,
                              n_steps=1000, F_init=F_high)
        assert abs(traj_off[-1]) < 0.1, (
            f"Flow did not decay; F_final = {traj_off[-1]:.4f}"
        )

    def test_steady_state_matches_simulation(self):
        """Long deterministic simulation converges to analytic steady state."""
        m = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                      sigma_F=0.0, F0=0.0, dt=0.005, seed=0)
        psi_I, psi_P, psi_D = 0.5, 0.6, 0.7
        traj = m.simulate(psi_I=psi_I, psi_P=psi_P, psi_D=psi_D,
                          n_steps=5000, F_init=0.0)
        F_star_analytic = m.steady_state(psi_I, psi_P, psi_D)
        F_empirical = float(traj[-1])
        assert np.isclose(F_empirical, F_star_analytic, atol=0.01), (
            f"Simulation F* = {F_empirical:.4f}, analytic F* = {F_star_analytic:.4f}"
        )

    def test_trajectory_length(self, model):
        """simulate() returns array of length n_steps + 1."""
        n = 200
        traj = model.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=n)
        assert len(traj) == n + 1

    def test_invalid_kappa_raises(self):
        """Constructor raises ValueError for kappa ≤ 0."""
        with pytest.raises(ValueError):
            FlowModel(kappa=0.0)
        with pytest.raises(ValueError):
            FlowModel(kappa=-1.0)

    def test_invalid_beta_raises(self):
        """Constructor raises ValueError for negative coupling strengths."""
        with pytest.raises(ValueError):
            FlowModel(beta_I=-0.1)

    def test_invalid_sigma_raises(self):
        """Constructor raises ValueError for negative sigma_F."""
        with pytest.raises(ValueError):
            FlowModel(sigma_F=-0.1)

    def test_deterministic_mode_reproducible(self):
        """σ_F = 0 produces identical trajectories every call."""
        m = FlowModel(sigma_F=0.0, seed=0)
        t1 = m.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=100)
        m2 = FlowModel(sigma_F=0.0, seed=0)
        t2 = m2.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=100)
        np.testing.assert_array_equal(t1, t2)

    def test_stochastic_mode_seeded_reproducible(self):
        """Same seed → identical stochastic trajectory."""
        m1 = FlowModel(sigma_F=0.5, seed=99)
        m2 = FlowModel(sigma_F=0.5, seed=99)
        t1 = m1.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=100)
        t2 = m2.simulate(psi_I=0.5, psi_P=0.5, psi_D=0.5, n_steps=100)
        np.testing.assert_array_equal(t1, t2)
