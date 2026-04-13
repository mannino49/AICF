"""
Inter-Layer Coupling Functions
===============================

Alternative coupling schemes for the three AICF layers, used in the
model-comparison section of the paper.

The primary model uses **additive coupling** (as specified in §7).
The alternatives test structural hypotheses about whether flow requires
all three layers simultaneously (multiplicative), has threshold structure
(gated), or is hierarchically organized (hierarchical).

Reference
---------
formalism_repair_v1.md §7; codebase_build_instructions.md Step 5.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit  # logistic sigmoid


def additive_coupling(
    psi_I: float,
    psi_P: float,
    psi_D: float,
    beta_I: float = 1.0,
    beta_P: float = 1.0,
    beta_D: float = 1.0,
) -> float:
    """Additive coupling — the primary AICF model.

    effective_input = β_I·ψ_I + β_P·ψ_P + β_D·ψ_D

    Each layer contributes independently and additively to the drive on F(t).
    This is the default model and the one specified in formalism §7.

    Parameters
    ----------
    psi_I, psi_P, psi_D : float
        Normalized layer factors ∈ [0, 1].
    beta_I, beta_P, beta_D : float
        Coupling strengths (units: flow_units / time).

    Returns
    -------
    float
        Effective driving input to the flow state equation.
    """
    return beta_I * psi_I + beta_P * psi_P + beta_D * psi_D


def multiplicative_coupling(
    psi_I: float,
    psi_P: float,
    psi_D: float,
    beta: float = 1.0,
) -> float:
    """Multiplicative coupling — all three layers required simultaneously.

    effective_input = β · ψ_I · ψ_P · ψ_D

    Under this model, a near-zero value in any single layer drives the
    effective input toward zero, capturing the hypothesis that flow requires
    the simultaneous presence of all three components.

    Parameters
    ----------
    psi_I, psi_P, psi_D : float
        Normalized layer factors ∈ [0, 1].
    beta : float
        Single coupling strength.

    Returns
    -------
    float
        Effective driving input ∈ [0, β].
    """
    return beta * psi_I * psi_P * psi_D


def gated_coupling(
    psi_I: float,
    psi_P: float,
    psi_D: float,
    beta_I: float = 1.0,
    beta_P: float = 1.0,
    beta_D: float = 1.0,
    threshold: float = 0.3,
) -> float:
    """Gated coupling — each layer only contributes above a threshold.

    effective_input = Σ_l β_l · ψ_l · 𝟙[ψ_l > θ]

    Tests whether flow has a threshold structure: layers contribute only
    when their factor exceeds a minimum level.

    Parameters
    ----------
    psi_I, psi_P, psi_D : float
        Normalized layer factors ∈ [0, 1].
    beta_I, beta_P, beta_D : float
        Coupling strengths.
    threshold : float
        Threshold θ ∈ [0, 1].  Contributions below this value are zeroed.

    Returns
    -------
    float
        Effective driving input.
    """
    I_contrib = beta_I * psi_I if psi_I > threshold else 0.0
    P_contrib = beta_P * psi_P if psi_P > threshold else 0.0
    D_contrib = beta_D * psi_D if psi_D > threshold else 0.0
    return I_contrib + P_contrib + D_contrib


def hierarchical_coupling(
    psi_I: float,
    psi_P: float,
    psi_D: float,
    beta_D: float = 1.0,
    gate_scale: float = 5.0,
) -> float:
    """Hierarchical coupling — I gates P, P gates D.

    effective_input = β_D · ψ_D · g(ψ_P · g(ψ_I))

    where g(x) = σ(k·(x − 0.5)) is a sigmoid gate that is near 0 when
    x < 0.5 and near 1 when x > 0.5.

    Tests the hypothesis that the informational layer enables inference,
    which in turn enables the dynamical signature.

    Parameters
    ----------
    psi_I, psi_P, psi_D : float
        Normalized layer factors ∈ [0, 1].
    beta_D : float
        Coupling strength of the dynamical layer (the terminal stage).
    gate_scale : float
        Steepness parameter k of the sigmoid gate.

    Returns
    -------
    float
        Effective driving input.
    """
    def gate(x: float) -> float:
        return float(expit(gate_scale * (x - 0.5)))

    return float(beta_D * psi_D * gate(psi_P * gate(psi_I)))
