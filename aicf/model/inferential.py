"""
Inferential Layer — ψ_P(t)
===========================

Implements the active-inference inferential factor as defined in
formalism_repair_v1.md Section 3.  The factor decomposes into two
sub-quantities:

1. **Policy negentropy** n_π(t) — how sharply the agent has selected a policy
2. **Automaticity index** A(t) — how closely the posterior matches the habitual prior

Combined:
    ψ_P(t) = w_prec · n_π(t) + w_auto · A(t)

The effective policy precision is modulated by planning depth τ through a
benefit-cost tradeoff that produces a single interior optimum τ* (§5.2).

References
----------
Parvizi-Wayne, D., Kotler, S., Mannino, M., & Friston, K. (2025).
    Active inference account of flow.
formalism_repair_v1.md §3, §5.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_distribution(p: NDArray[np.float64], name: str) -> None:
    """Validate that *p* is a 1-D probability distribution."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {p.shape}")
    if np.any(p < 0):
        raise ValueError(f"{name} must be non-negative")
    total = p.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1; got {total:.6f}")


def _shannon_entropy(p: NDArray[np.float64]) -> float:
    """Shannon entropy H(p) in nats; 0·log(0) = 0 by convention."""
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


def _kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64]) -> float:
    """KL divergence D_KL(p || q) in nats.

    Handles p_i = 0 via 0·log(0/q_i) = 0.
    Raises ValueError if q_i = 0 where p_i > 0 (infinite KL).
    """
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p_pos = p > 0
    if np.any(q[p_pos] <= 0):
        raise ValueError(
            "D_KL(p || q) is infinite: q assigns zero probability where p > 0"
        )
    return float(np.sum(p[p_pos] * (np.log(p[p_pos]) - np.log(q[p_pos]))))


class InferentialLayer:
    """Compute the inferential factor ψ_P(t) from active inference quantities.

    The inferential layer captures two aspects of the flow-state policy
    distribution: (1) how sharply the posterior is peaked (negentropy n_π),
    and (2) how closely deliberative inference agrees with habitual behavior
    (automaticity A).

    Parameters
    ----------
    w_prec : float, optional
        Weight on policy negentropy n_π(t).  Default 0.5.
    w_auto : float, optional
        Weight on automaticity index A(t).  Must satisfy w_prec + w_auto = 1.
        Default 0.5.
    tau_0 : float, optional
        Planning-benefit scale parameter τ₀ (units: planning steps).
        Controls how fast planning benefit saturates.  Default 2.0.
    tau_max : float, optional
        Computational-cost scale parameter τ_max (units: planning steps).
        Controls how fast deep planning degrades precision.  Default 10.0.

    Raises
    ------
    ValueError
        If w_prec + w_auto ≠ 1, or if τ_max ≤ τ₀.

    Notes
    -----
    Implements formalism_repair_v1.md §3 and §5.
    """

    def __init__(
        self,
        w_prec: float = 0.5,
        w_auto: float = 0.5,
        tau_0: float = 2.0,
        tau_max: float = 10.0,
    ) -> None:
        if not np.isclose(w_prec + w_auto, 1.0, atol=1e-6):
            raise ValueError(
                f"w_prec + w_auto must equal 1; got {w_prec + w_auto:.6f}"
            )
        if w_prec < 0 or w_auto < 0:
            raise ValueError("Weights must be non-negative")
        if tau_max <= tau_0:
            raise ValueError(
                f"tau_max ({tau_max}) must be greater than tau_0 ({tau_0}) "
                "for a well-defined interior optimum (§5.2)"
            )
        if tau_0 <= 0:
            raise ValueError(f"tau_0 must be positive; got {tau_0}")

        self.w_prec = float(w_prec)
        self.w_auto = float(w_auto)
        self.tau_0 = float(tau_0)
        self.tau_max = float(tau_max)

    # ------------------------------------------------------------------
    # Planning depth and precision
    # ------------------------------------------------------------------

    def compute_effective_precision(self, gamma: float, tau: float) -> float:
        """Compute effective policy precision γ_eff(t) under planning depth τ.

        The benefit-cost tradeoff (formalism §5.2):

            γ_eff = γ · [1 − exp(−τ/τ₀)] · exp(−τ/τ_max)

        The first factor is the planning *benefit* (increases, then saturates).
        The second factor is the computational *cost* (decreases exponentially).
        The product has a single interior maximum at τ* (see :meth:`optimal_planning_depth`).

        Parameters
        ----------
        gamma : float
            Base policy precision γ(t) > 0.  Units: dimensionless (inverse
            temperature of the softmax over policies).
        tau : float
            Current planning depth τ(t) > 0.  Units: planning steps.

        Returns
        -------
        float
            γ_eff ≥ 0.  Dimensionless.

        Raises
        ------
        ValueError
            If gamma ≤ 0 or tau ≤ 0.
        """
        if gamma <= 0:
            raise ValueError(f"gamma must be positive; got {gamma}")
        if tau <= 0:
            raise ValueError(f"tau must be positive; got {tau}")

        benefit = 1.0 - np.exp(-tau / self.tau_0)   # ∈ (0, 1)
        cost = np.exp(-tau / self.tau_max)            # ∈ (0, 1)
        return float(gamma * benefit * cost)

    def optimal_planning_depth(self) -> float:
        """Return the optimal planning depth τ* that maximises γ_eff.

        Setting d/dτ γ_eff = 0 and solving analytically (see derivation below)
        yields:

            τ* = τ₀ · ln(1 + τ_max / τ₀)

        **Derivation**: with γ_eff = γ · [1 − e^{-τ/τ₀}] · e^{-τ/τ_max},
        the derivative w.r.t. τ set to zero gives:

            (1/τ₀) · e^{-τ/τ₀} = (1/τ_max) · [1 − e^{-τ/τ₀}]

        Letting u = e^{-τ/τ₀}:  u(τ_max + τ₀) = τ₀  →  u = τ₀/(τ_max + τ₀)

        So  τ* = τ₀ · ln((τ_max + τ₀)/τ₀) = τ₀ · ln(1 + τ_max/τ₀).

        Note: the formula in the companion document (formalism_repair_v1.md §5.2)
        contains a typographical error.  This implementation uses the correct
        formula verified numerically.

        Returns
        -------
        float
            τ* in planning-step units.
        """
        tau_star = self.tau_0 * np.log(1.0 + self.tau_max / self.tau_0)
        return float(tau_star)

    # ------------------------------------------------------------------
    # Policy posterior
    # ------------------------------------------------------------------

    def compute_policy_posterior(
        self,
        expected_free_energies: NDArray[np.float64],
        gamma_eff: float,
    ) -> NDArray[np.float64]:
        """Compute softmax policy posterior from expected free energies.

        π_i = softmax(−γ_eff · G_i)

        Uses the numerically stable form (subtract maximum before exp).

        Parameters
        ----------
        expected_free_energies : 1-D array of float
            G_i(t; τ) for each policy i.  Units: nats (information-theoretic
            EFE as in active inference; sign convention: more negative = better).
        gamma_eff : float
            Effective precision γ_eff ≥ 0.

        Returns
        -------
        1-D array of float
            Policy posterior π(t) summing to 1.

        Raises
        ------
        ValueError
            If gamma_eff < 0 or input is not 1-D.
        """
        G = np.asarray(expected_free_energies, dtype=float)
        if G.ndim != 1:
            raise ValueError(
                f"expected_free_energies must be 1-D; got shape {G.shape}"
            )
        if gamma_eff < 0:
            raise ValueError(f"gamma_eff must be ≥ 0; got {gamma_eff}")

        log_unnorm = -gamma_eff * G
        # Numerically stable softmax: subtract max
        log_unnorm -= log_unnorm.max()
        unnorm = np.exp(log_unnorm)
        return (unnorm / unnorm.sum()).astype(float)

    # ------------------------------------------------------------------
    # Policy negentropy n_π
    # ------------------------------------------------------------------

    def compute_negentropy(
        self, policy_posterior: NDArray[np.float64]
    ) -> float:
        """Compute normalized policy negentropy n_π(t).

            n_π = 1 − H[π(t)] / log(N_π)

        where N_π is the number of available policies and H[π] is entropy in nats.

        Parameters
        ----------
        policy_posterior : 1-D array of float
            π(t) — posterior distribution over N_π policies.
            Must be non-negative and sum to 1.

        Returns
        -------
        float
            n_π ∈ [0, 1].
            n_π = 1 → deterministic policy (zero entropy).
            n_π = 0 → uniform (maximum uncertainty).

        Notes
        -----
        Boundary convention (§3.2): if N_π = 1, returns 1.0.
        """
        pi = np.asarray(policy_posterior, dtype=float)
        _validate_distribution(pi, "policy_posterior")

        N_pi = len(pi)
        if N_pi == 1:
            # Single policy: trivially certain (§3.2 boundary convention)
            return 1.0

        H_max = np.log(N_pi)   # log(N_π) in nats
        H_pi = _shannon_entropy(pi)
        n_pi = float(np.clip(1.0 - H_pi / H_max, 0.0, 1.0))
        return n_pi

    # ------------------------------------------------------------------
    # Automaticity index A
    # ------------------------------------------------------------------

    def compute_automaticity(
        self,
        policy_posterior: NDArray[np.float64],
        habitual_prior: NDArray[np.float64],
    ) -> float:
        """Compute the automaticity index A(t).

            A = 1 − D_KL[π_post(t) || π_habit(t)] / log(N_π)

        where log(N_π) is the theoretical maximum KL divergence (§3.2).

        Parameters
        ----------
        policy_posterior : 1-D array of float
            π_post(t) — posterior policy after active inference.
        habitual_prior : 1-D array of float
            π_habit(t) — habitual policy prior (model-free / learned).
            Must have the same length as policy_posterior.

        Returns
        -------
        float
            A ∈ [0, 1].
            A = 1 → posterior = habitual prior (fully automatic).
            A = 0 → maximally divergent (fully deliberative).

        Raises
        ------
        ValueError
            If posterior and prior have different lengths, or if the prior
            assigns zero probability where the posterior is positive (infinite KL).
        """
        pi_post = np.asarray(policy_posterior, dtype=float)
        pi_habit = np.asarray(habitual_prior, dtype=float)

        _validate_distribution(pi_post, "policy_posterior")
        _validate_distribution(pi_habit, "habitual_prior")

        if len(pi_post) != len(pi_habit):
            raise ValueError(
                f"policy_posterior and habitual_prior must have the same length; "
                f"got {len(pi_post)} and {len(pi_habit)}"
            )

        N_pi = len(pi_post)
        if N_pi == 1:
            # Boundary convention (§3.2)
            return 1.0

        D_KL_max = np.log(N_pi)  # theoretical maximum KL in nats (§3.2)

        try:
            kl = _kl_divergence(pi_post, pi_habit)
        except ValueError:
            # Infinite KL → A = 0
            return 0.0

        # Clamp to [0, 1]: negative kl can arise from floating-point noise
        A = float(np.clip(1.0 - kl / D_KL_max, 0.0, 1.0))
        return A

    # ------------------------------------------------------------------
    # Top-level factor
    # ------------------------------------------------------------------

    def compute(
        self,
        policy_posterior: NDArray[np.float64],
        habitual_prior: NDArray[np.float64],
    ) -> float:
        """Compute the inferential factor ψ_P(t).

            ψ_P = w_prec · n_π(t) + w_auto · A(t)

        Parameters
        ----------
        policy_posterior : 1-D array of float
            π_post(t) — posterior policy distribution.
        habitual_prior : 1-D array of float
            π_habit(t) — habitual prior policy distribution.

        Returns
        -------
        float
            ψ_P ∈ [0, 1].
        """
        n_pi = self.compute_negentropy(policy_posterior)
        A = self.compute_automaticity(policy_posterior, habitual_prior)
        psi_P = float(np.clip(self.w_prec * n_pi + self.w_auto * A, 0.0, 1.0))
        return psi_P
