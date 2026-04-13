"""
AICF Flow Model — Full Latent State-Space Model
================================================

Implements the central state equation and observation model for the
Active Inference–Complexity Model of Flow, as specified in
formalism_repair_v1.md §1.2 and §7.

State equation (Itô SDE):
    dF/dt = −κ[F(t) − F₀] + β_I·ψ_I(t) + β_P·ψ_P(t) + β_D·ψ_D(t) + σ_F·dW(t)

Observation equation:
    y(t) = Λ · F(t) + ε(t),  ε ~ N(0, R)

F(t) is a scalar latent variable (dimensionless by convention).  The three
factor inputs ψ_I, ψ_P, ψ_D ∈ [0, 1] drive flow above the baseline F₀.

Numerical integration uses the Euler-Maruyama scheme, which is first-order
strong and sufficient for this additive-noise SDE.

Reference
---------
formalism_repair_v1.md §1, §6, §7.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
from numpy.typing import NDArray


class FlowModel:
    """Ornstein-Uhlenbeck state-space model of psychological flow.

    The latent flow state F(t) evolves as a mean-reverting SDE driven by
    three normalized factor inputs.  The model supports deterministic
    (σ_F = 0) and stochastic (σ_F > 0) integration.

    Parameters
    ----------
    kappa : float, optional
        Mean-reversion rate κ > 0.  Units: 1/time.  Controls how fast
        flow decays to baseline in the absence of driving inputs.
        Default 1.0.
    beta_I : float, optional
        Coupling strength for the informational layer β_I ≥ 0.
        Units: flow_units / time.  Default 1.0.
    beta_P : float, optional
        Coupling strength for the inferential layer β_P ≥ 0.
        Units: flow_units / time.  Default 1.0.
    beta_D : float, optional
        Coupling strength for the dynamical layer β_D ≥ 0.
        Units: flow_units / time.  Default 1.0.
    sigma_F : float, optional
        Process noise scale σ_F ≥ 0.  Units: flow_units / √time.
        Set to 0.0 for deterministic integration.  Default 0.1.
    F0 : float, optional
        Baseline flow level (convention: 0 = no flow).  Default 0.0.
    dt : float, optional
        Time step for Euler-Maruyama integration.  Units: time.  Default 0.01.
    seed : int or numpy.random.Generator, optional
        Seed for the random number generator (reproducibility).

    Notes
    -----
    Dimensional analysis (from §6.1):
        dF/dt   [F/t] = −κ[F/t] + β_I·ψ_I [F/t] + ... + σ_F·dW/dt [F/√t · 1/√t]
    All factors ψ are dimensionless ∈ [0, 1].
    The model is completely self-contained — it can be used without the
    layer classes by supplying pre-computed ψ values.
    """

    def __init__(
        self,
        kappa: float = 1.0,
        beta_I: float = 1.0,
        beta_P: float = 1.0,
        beta_D: float = 1.0,
        sigma_F: float = 0.1,
        F0: float = 0.0,
        dt: float = 0.01,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if kappa <= 0:
            raise ValueError(f"kappa must be positive; got {kappa}")
        if any(b < 0 for b in (beta_I, beta_P, beta_D)):
            raise ValueError("Coupling strengths beta_* must be non-negative")
        if sigma_F < 0:
            raise ValueError(f"sigma_F must be non-negative; got {sigma_F}")
        if dt <= 0:
            raise ValueError(f"dt must be positive; got {dt}")

        self.kappa = float(kappa)
        self.beta_I = float(beta_I)
        self.beta_P = float(beta_P)
        self.beta_D = float(beta_D)
        self.sigma_F = float(sigma_F)
        self.F0 = float(F0)
        self.dt = float(dt)

        if isinstance(seed, np.random.Generator):
            self.rng = seed
        else:
            self.rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core SDE components
    # ------------------------------------------------------------------

    def drift(
        self,
        F: float,
        psi_I: float,
        psi_P: float,
        psi_D: float,
    ) -> float:
        """Compute the drift term of the SDE.

        f(F, ψ) = −κ(F − F₀) + β_I·ψ_I + β_P·ψ_P + β_D·ψ_D

        Units: [flow_units / time]

        Parameters
        ----------
        F : float
            Current latent flow state.
        psi_I, psi_P, psi_D : float
            Normalized factor inputs ∈ [0, 1].

        Returns
        -------
        float
            Drift value in units [F / time].
        """
        mean_reversion = -self.kappa * (F - self.F0)
        driving = (
            self.beta_I * psi_I
            + self.beta_P * psi_P
            + self.beta_D * psi_D
        )
        return mean_reversion + driving

    def diffusion(self, F: float = 0.0) -> float:
        """Compute the diffusion coefficient (constant, additive noise).

        g(F) = σ_F

        Units: [flow_units / √time]

        Parameters
        ----------
        F : float
            Current latent flow state (unused for additive noise; included
            for API consistency with state-dependent diffusion extensions).

        Returns
        -------
        float
            σ_F.
        """
        return self.sigma_F

    def step(
        self,
        F: float,
        psi_I: float,
        psi_P: float,
        psi_D: float,
    ) -> float:
        """Advance the SDE by one Euler-Maruyama step.

        F_{t+Δt} = F_t + f(F_t, ψ)·Δt + g(F_t)·√Δt · ξ_t

        where ξ_t ~ N(0, 1) is a standard normal increment.

        Parameters
        ----------
        F : float
            Current latent flow state.
        psi_I, psi_P, psi_D : float
            Factor inputs ∈ [0, 1] at the current time step.

        Returns
        -------
        float
            Updated latent flow state at t + dt.
        """
        noise = self.rng.standard_normal()
        dF = (
            self.drift(F, psi_I, psi_P, psi_D) * self.dt
            + self.diffusion(F) * np.sqrt(self.dt) * noise
        )
        return F + dF

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------

    def simulate(
        self,
        psi_I: float | Sequence[float] | NDArray[np.float64],
        psi_P: float | Sequence[float] | NDArray[np.float64],
        psi_D: float | Sequence[float] | NDArray[np.float64],
        n_steps: int,
        F_init: float | None = None,
    ) -> NDArray[np.float64]:
        """Simulate the flow state trajectory via Euler-Maruyama integration.

        Parameters
        ----------
        psi_I, psi_P, psi_D : float or 1-D array
            Factor time series.  If scalar, the value is held constant for
            all n_steps.  If array, must have length n_steps.
        n_steps : int
            Number of integration steps.
        F_init : float, optional
            Initial condition F(0).  Defaults to F0 (baseline).

        Returns
        -------
        1-D array of float, shape (n_steps + 1,)
            Full trajectory F(0), F(Δt), ..., F(n_steps · Δt).
        """
        if n_steps < 1:
            raise ValueError(f"n_steps must be ≥ 1; got {n_steps}")

        # Broadcast scalar inputs to arrays
        psi_I_arr = self._broadcast_input(psi_I, n_steps, "psi_I")
        psi_P_arr = self._broadcast_input(psi_P, n_steps, "psi_P")
        psi_D_arr = self._broadcast_input(psi_D, n_steps, "psi_D")

        F_traj = np.empty(n_steps + 1)
        F_traj[0] = self.F0 if F_init is None else float(F_init)

        for t in range(n_steps):
            F_traj[t + 1] = self.step(
                F_traj[t], psi_I_arr[t], psi_P_arr[t], psi_D_arr[t]
            )

        return F_traj

    # ------------------------------------------------------------------
    # Analytical results
    # ------------------------------------------------------------------

    def steady_state(
        self,
        psi_I: float,
        psi_P: float,
        psi_D: float,
    ) -> float:
        """Compute the deterministic steady-state flow level.

        At steady state (dF/dt = 0, ignoring noise):

            F* = F₀ + (β_I·ψ_I + β_P·ψ_P + β_D·ψ_D) / κ

        This is the mean of the stationary distribution of the OU process
        (the noise adds symmetric fluctuations around F*).

        Parameters
        ----------
        psi_I, psi_P, psi_D : float
            Constant factor inputs ∈ [0, 1].

        Returns
        -------
        float
            F* — the "deep flow" level for the given constant inputs.
        """
        driving = self.beta_I * psi_I + self.beta_P * psi_P + self.beta_D * psi_D
        return self.F0 + driving / self.kappa

    # ------------------------------------------------------------------
    # Observation model
    # ------------------------------------------------------------------

    def observe(
        self,
        F_trajectory: NDArray[np.float64],
        Lambda: NDArray[np.float64],
        R: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Generate observations from the latent flow trajectory.

        y(t) = Λ · F(t) + ε(t),  ε ~ N(0, R)

        Parameters
        ----------
        F_trajectory : 1-D array of float, shape (n_steps,)
            Latent flow state trajectory.
        Lambda : 2-D array of float, shape (n_obs, 1)
            Observation loading matrix.  Maps the scalar F to the
            n_obs-dimensional observation space.
        R : 2-D array of float, shape (n_obs, n_obs)
            Observation noise covariance matrix (symmetric positive definite).

        Returns
        -------
        2-D array of float, shape (n_obs, n_steps)
            Simulated observations y(t).
        """
        F = np.asarray(F_trajectory, dtype=float)
        Lambda = np.asarray(Lambda, dtype=float)
        R = np.asarray(R, dtype=float)

        if Lambda.ndim != 2 or Lambda.shape[1] != 1:
            raise ValueError(
                f"Lambda must have shape (n_obs, 1); got {Lambda.shape}"
            )
        n_obs = Lambda.shape[0]
        if R.shape != (n_obs, n_obs):
            raise ValueError(
                f"R must have shape ({n_obs}, {n_obs}); got {R.shape}"
            )

        # y = Λ · F(t)^T  + ε
        # F is (n_steps,); Lambda is (n_obs, 1)
        signal = Lambda @ F[np.newaxis, :]  # (n_obs, n_steps)
        noise = self.rng.multivariate_normal(
            mean=np.zeros(n_obs), cov=R, size=len(F)
        ).T  # (n_obs, n_steps)
        return signal + noise

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _broadcast_input(
        value: float | Sequence[float] | NDArray[np.float64],
        n_steps: int,
        name: str,
    ) -> NDArray[np.float64]:
        """Broadcast a scalar or array input to length n_steps."""
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(n_steps, float(arr))
        if arr.ndim == 1:
            if len(arr) != n_steps:
                raise ValueError(
                    f"{name} array length ({len(arr)}) must equal "
                    f"n_steps ({n_steps})"
                )
            return arr
        raise ValueError(f"{name} must be scalar or 1-D array; got shape {arr.shape}")
