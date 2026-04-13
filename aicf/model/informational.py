"""
Informational Layer — ψ_I(t)
=============================

Implements the normalized mutual information between means (M) and ends (E),
as defined in formalism_repair_v1.md Section 2.

Theory
------
ψ_I(t) = I(M; E | s(t)) / min{H(M | s(t)), H(E | s(t))}

where I(M; E) = H(M) + H(E) − H(M, E) is the Shannon mutual information
(in nats) and the denominator is the minimum marginal entropy (upper bound
on mutual information). This maps ψ_I into [0, 1] regardless of the
cardinality of the M and E supports.

Reference
---------
Melnikoff, D. E., et al. (2022). The Minimal Mind. *Psychological Review*.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _validate_distribution(p: NDArray[np.float64], name: str) -> None:
    """Check that *p* is a valid probability distribution."""
    p = np.asarray(p, dtype=float)
    if p.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array; got shape {p.shape}")
    if np.any(p < 0):
        raise ValueError(f"{name} must be non-negative")
    total = p.sum()
    if not np.isclose(total, 1.0, atol=1e-6):
        raise ValueError(f"{name} must sum to 1; got {total:.6f}")


def _shannon_entropy(p: NDArray[np.float64]) -> float:
    """Shannon entropy H(p) in nats (base e).

    Handles p_i = 0 via the convention 0 · log(0) = 0.
    """
    p = np.asarray(p, dtype=float)
    mask = p > 0
    return float(-np.sum(p[mask] * np.log(p[mask])))


class InformationalLayer:
    """Compute the normalized mutual information factor ψ_I(t).

    The informational layer quantifies the statistical coupling between the
    agent's available *means* (actions / strategies) and desired *ends*
    (goals / outcomes), normalized so that ψ_I ∈ [0, 1].

    This class is stateless: it contains no internal state and all outputs
    are deterministic functions of the inputs.

    Parameters
    ----------
    None
        The class takes no constructor arguments; all inputs are supplied
        at call time.

    Notes
    -----
    Implements formalism_repair_v1.md §2 exactly.
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(
        self,
        p_means: NDArray[np.float64],
        p_ends: NDArray[np.float64],
        p_joint: NDArray[np.float64] | None = None,
    ) -> float:
        """Compute ψ_I given marginal distributions over means and ends.

        Assumes independence (p_joint = outer product) unless *p_joint* is
        provided explicitly.

        Parameters
        ----------
        p_means : 1-D array of float
            Probability distribution over available means (actions).
            Must be non-negative and sum to 1.
        p_ends : 1-D array of float
            Probability distribution over desired ends (goals).
            Must be non-negative and sum to 1.
        p_joint : 2-D array of float, optional
            Joint distribution P(M, E) with shape (len(p_means), len(p_ends)).
            If None, independence is assumed: p_joint = outer(p_means, p_ends).

        Returns
        -------
        float
            ψ_I ∈ [0, 1].  Returns 0.0 in all degenerate edge cases
            (see §2.2 boundary convention).

        Raises
        ------
        ValueError
            If inputs are not valid probability distributions.
        """
        p_means = np.asarray(p_means, dtype=float)
        p_ends = np.asarray(p_ends, dtype=float)
        _validate_distribution(p_means, "p_means")
        _validate_distribution(p_ends, "p_ends")

        if p_joint is None:
            # Independence assumption: P(M, E) = P(M) · P(E)
            p_joint = np.outer(p_means, p_ends)
        else:
            p_joint = np.asarray(p_joint, dtype=float)
            if p_joint.shape != (len(p_means), len(p_ends)):
                raise ValueError(
                    f"p_joint shape {p_joint.shape} does not match "
                    f"(len(p_means)={len(p_means)}, len(p_ends)={len(p_ends)})"
                )
            if np.any(p_joint < 0):
                raise ValueError("p_joint must be non-negative")
            if not np.isclose(p_joint.sum(), 1.0, atol=1e-6):
                raise ValueError(
                    f"p_joint must sum to 1; got {p_joint.sum():.6f}"
                )

        return self._compute_normalized_mi(p_means, p_ends, p_joint)

    def compute_from_joint(self, p_joint: NDArray[np.float64]) -> float:
        """Compute ψ_I directly from a 2-D joint distribution P(M, E).

        Marginals are derived by summation; then the normalized mutual
        information is computed exactly as in :meth:`compute`.

        Parameters
        ----------
        p_joint : 2-D array of float
            Joint distribution P(M, E) with shape (n_means, n_ends).
            Must be non-negative and sum to 1.

        Returns
        -------
        float
            ψ_I ∈ [0, 1].
        """
        p_joint = np.asarray(p_joint, dtype=float)
        if p_joint.ndim != 2:
            raise ValueError(
                f"p_joint must be 2-D; got shape {p_joint.shape}"
            )
        if np.any(p_joint < 0):
            raise ValueError("p_joint must be non-negative")
        if not np.isclose(p_joint.sum(), 1.0, atol=1e-6):
            raise ValueError(
                f"p_joint must sum to 1; got {p_joint.sum():.6f}"
            )

        p_means = p_joint.sum(axis=1)  # marginal over means
        p_ends = p_joint.sum(axis=0)   # marginal over ends
        return self._compute_normalized_mi(p_means, p_ends, p_joint)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_normalized_mi(
        p_means: NDArray[np.float64],
        p_ends: NDArray[np.float64],
        p_joint: NDArray[np.float64],
    ) -> float:
        """Core computation: normalized mutual information.

        ψ_I = I(M; E) / min{H(M), H(E)}

        with I(M; E) = H(M) + H(E) − H(M, E)

        Returns
        -------
        float in [0, 1]
        """
        # Entropy of marginals (nats)
        H_M = _shannon_entropy(p_means)  # H(M) — nats
        H_E = _shannon_entropy(p_ends)   # H(E) — nats

        # Boundary convention (§2.2): if min{H(M), H(E)} = 0 → ψ_I = 0
        denom = min(H_M, H_E)
        if denom < 1e-12:
            return 0.0

        # Joint entropy H(M, E) — nats
        H_ME = _shannon_entropy(p_joint.ravel())

        # Mutual information I(M; E) = H(M) + H(E) - H(M, E)
        MI = H_M + H_E - H_ME  # nats — always ≥ 0

        # Clamp to [0, 1] to absorb floating-point rounding
        psi_I = float(np.clip(MI / denom, 0.0, 1.0))
        return psi_I
