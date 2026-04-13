"""
Reduced (Ablation) Models
==========================

Factory functions that create FlowModel instances with subsets of the
three factor layers active.  Used in the model-comparison section of the
paper to test whether all three layers are necessary for capturing flow.

Each reduced model shares the same FlowModel interface, enabling direct
comparison via BIC, RMSE, and likelihood ratios.

Reference
---------
codebase_build_instructions.md Step 6.
"""

from __future__ import annotations

from aicf.model.flow_model import FlowModel


def InformationOnlyModel(
    beta_I: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven only by the informational layer (β_P = β_D = 0).

    Parameters
    ----------
    beta_I : float
        Coupling strength for the informational layer.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`
        (e.g., kappa, sigma_F, dt, seed).

    Returns
    -------
    FlowModel
        Configured with β_P = β_D = 0.
    """
    return FlowModel(beta_I=beta_I, beta_P=0.0, beta_D=0.0, **kwargs)


def InferenceOnlyModel(
    beta_P: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven only by the inferential layer (β_I = β_D = 0).

    Parameters
    ----------
    beta_P : float
        Coupling strength for the inferential layer.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Configured with β_I = β_D = 0.
    """
    return FlowModel(beta_I=0.0, beta_P=beta_P, beta_D=0.0, **kwargs)


def DynamicsOnlyModel(
    beta_D: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven only by the dynamical layer (β_I = β_P = 0).

    Parameters
    ----------
    beta_D : float
        Coupling strength for the dynamical layer.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Configured with β_I = β_P = 0.
    """
    return FlowModel(beta_I=0.0, beta_P=0.0, beta_D=beta_D, **kwargs)


def InformationInferenceModel(
    beta_I: float = 1.0,
    beta_P: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven by informational + inferential layers (β_D = 0).

    Parameters
    ----------
    beta_I, beta_P : float
        Coupling strengths.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Configured with β_D = 0.
    """
    return FlowModel(beta_I=beta_I, beta_P=beta_P, beta_D=0.0, **kwargs)


def InformationDynamicsModel(
    beta_I: float = 1.0,
    beta_D: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven by informational + dynamical layers (β_P = 0).

    Parameters
    ----------
    beta_I, beta_D : float
        Coupling strengths.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Configured with β_P = 0.
    """
    return FlowModel(beta_I=beta_I, beta_P=0.0, beta_D=beta_D, **kwargs)


def InferenceDynamicsModel(
    beta_P: float = 1.0,
    beta_D: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Flow driven by inferential + dynamical layers (β_I = 0).

    Parameters
    ----------
    beta_P, beta_D : float
        Coupling strengths.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Configured with β_I = 0.
    """
    return FlowModel(beta_I=0.0, beta_P=beta_P, beta_D=beta_D, **kwargs)


def FullModel(
    beta_I: float = 1.0,
    beta_P: float = 1.0,
    beta_D: float = 1.0,
    **kwargs,
) -> FlowModel:
    """Full AICF model with all three layers active.

    This is the primary model.  Functionally equivalent to constructing a
    :class:`FlowModel` directly, but provided here for uniform interface
    in model-comparison loops.

    Parameters
    ----------
    beta_I, beta_P, beta_D : float
        Coupling strengths.
    **kwargs
        Additional keyword arguments forwarded to :class:`FlowModel`.

    Returns
    -------
    FlowModel
        Full AICF model.
    """
    return FlowModel(beta_I=beta_I, beta_P=beta_P, beta_D=beta_D, **kwargs)


# Map from string name to factory (for programmatic model comparison)
MODEL_REGISTRY: dict[str, callable] = {
    "information_only": InformationOnlyModel,
    "inference_only": InferenceOnlyModel,
    "dynamics_only": DynamicsOnlyModel,
    "information_inference": InformationInferenceModel,
    "information_dynamics": InformationDynamicsModel,
    "inference_dynamics": InferenceDynamicsModel,
    "full": FullModel,
}
