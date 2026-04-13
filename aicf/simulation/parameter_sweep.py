"""
Parameter Sweep Infrastructure
================================

Grid search over model parameters to generate regime diagrams and
prediction landscapes.

Reference
---------
codebase_build_instructions.md Step 8.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from aicf.model.flow_model import FlowModel
from aicf.simulation.engine import SimulationEngine, SimulationResult


class ParameterSweep:
    """Grid search over one, two, or three model parameters.

    Parameters
    ----------
    n_steps : int, optional
        Number of SDE integration steps per simulation.  Default 5000.
    n_trials : int, optional
        Number of Monte Carlo trials per parameter setting.  Default 50.
    seed : int, optional
        Master random seed.  Default 0.

    Examples
    --------
    >>> from aicf.model.reduced_models import FullModel
    >>> sweep = ParameterSweep(n_steps=2000, n_trials=20)
    >>> psi_range = np.linspace(0, 1, 11)
    >>> result = sweep.sweep_2d(
    ...     FullModel,
    ...     param1_name="psi_I", param1_range=psi_range,
    ...     param2_name="psi_P", param2_range=psi_range,
    ...     fixed_params={"psi_D": 0.5},
    ...     metric_fn=lambda r: r.empirical_steady_state(),
    ... )
    """

    def __init__(
        self,
        n_steps: int = 5000,
        n_trials: int = 50,
        seed: int = 0,
    ) -> None:
        self.n_steps = n_steps
        self.n_trials = n_trials
        self.seed = seed
        self._engine = SimulationEngine()

    def sweep_2d(
        self,
        model_factory: Callable[..., FlowModel],
        param1_name: str,
        param1_range: NDArray[np.float64],
        param2_name: str,
        param2_range: NDArray[np.float64],
        fixed_params: dict,
        metric_fn: Callable[[SimulationResult], float],
        model_kwargs: dict | None = None,
    ) -> NDArray[np.float64]:
        """2-D grid search over two parameters.

        Parameters
        ----------
        model_factory : callable
            Factory that returns a :class:`FlowModel` instance.  One of the
            functions in :mod:`aicf.model.reduced_models`.
        param1_name, param2_name : str
            Names of the swept parameters.  Must be keys in the *inputs*
            dict ('psi_I', 'psi_P', 'psi_D') or FlowModel constructor args
            ('kappa', 'beta_I', etc.).
        param1_range, param2_range : 1-D array
            Values to sweep for each parameter.
        fixed_params : dict
            Fixed values for all parameters not being swept.
        metric_fn : callable
            Function that maps :class:`SimulationResult` → scalar float.
            Common choices: empirical steady state, mean peak, variance.
        model_kwargs : dict, optional
            Extra keyword arguments forwarded to *model_factory* (e.g.,
            kappa, sigma_F, dt).

        Returns
        -------
        2-D array of float, shape (len(param1_range), len(param2_range))
            Grid of metric values.
        """
        model_kwargs = model_kwargs or {}
        p1 = np.asarray(param1_range)
        p2 = np.asarray(param2_range)
        result_grid = np.empty((len(p1), len(p2)))

        _PSI_KEYS = {"psi_I", "psi_P", "psi_D"}
        _MODEL_KEYS = {"kappa", "beta_I", "beta_P", "beta_D", "sigma_F", "F0", "dt"}

        for i, v1 in enumerate(p1):
            for j, v2 in enumerate(p2):
                # Separate into model constructor params and simulation inputs
                params = {param1_name: v1, param2_name: v2, **fixed_params}
                sim_inputs = {k: params[k] for k in params if k in _PSI_KEYS}
                mk = {k: params[k] for k in params if k in _MODEL_KEYS}
                mk.update(model_kwargs)

                model = model_factory(**mk)
                res = self._engine.run(
                    model=model,
                    inputs=sim_inputs,
                    n_steps=self.n_steps,
                    n_trials=self.n_trials,
                    seed=self.seed + i * len(p2) + j,
                )
                result_grid[i, j] = metric_fn(res)

        return result_grid

    def sweep_3d(
        self,
        model_factory: Callable[..., FlowModel],
        param1_name: str,
        param1_range: NDArray[np.float64],
        param2_name: str,
        param2_range: NDArray[np.float64],
        param3_name: str,
        param3_range: NDArray[np.float64],
        fixed_params: dict,
        metric_fn: Callable[[SimulationResult], float],
        model_kwargs: dict | None = None,
    ) -> NDArray[np.float64]:
        """3-D grid search over three parameters.

        Returns
        -------
        3-D array of float, shape (len(p1), len(p2), len(p3))
        """
        model_kwargs = model_kwargs or {}
        p1 = np.asarray(param1_range)
        p2 = np.asarray(param2_range)
        p3 = np.asarray(param3_range)
        result_grid = np.empty((len(p1), len(p2), len(p3)))

        _PSI_KEYS = {"psi_I", "psi_P", "psi_D"}
        _MODEL_KEYS = {"kappa", "beta_I", "beta_P", "beta_D", "sigma_F", "F0", "dt"}

        total = len(p1) * len(p2) * len(p3)
        count = 0

        for i, v1 in enumerate(p1):
            for j, v2 in enumerate(p2):
                for k, v3 in enumerate(p3):
                    params = {
                        param1_name: v1,
                        param2_name: v2,
                        param3_name: v3,
                        **fixed_params,
                    }
                    sim_inputs = {x: params[x] for x in params if x in _PSI_KEYS}
                    mk = {x: params[x] for x in params if x in _MODEL_KEYS}
                    mk.update(model_kwargs)

                    model = model_factory(**mk)
                    res = self._engine.run(
                        model=model,
                        inputs=sim_inputs,
                        n_steps=self.n_steps,
                        n_trials=self.n_trials,
                        seed=self.seed + count,
                    )
                    result_grid[i, j, k] = metric_fn(res)
                    count += 1

        return result_grid
