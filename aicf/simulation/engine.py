"""
SDE Simulation Engine
======================

Runs multiple independent realizations of the AICF state equation and
aggregates results into a :class:`SimulationResult` container.

Reference
---------
codebase_build_instructions.md Step 7.
formalism_repair_v1.md §1.2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from aicf.model.flow_model import FlowModel


@dataclass
class SimulationResult:
    """Container for the output of a :class:`SimulationEngine` run.

    Attributes
    ----------
    trajectories : 2-D array, shape (n_trials, n_steps + 1)
        Individual F(t) trajectories for each trial.
    mean_trajectory : 1-D array, shape (n_steps + 1,)
        Trial-averaged trajectory.
    std_trajectory : 1-D array, shape (n_steps + 1,)
        Trial standard deviation at each time step.
    steady_state_theoretical : float
        Analytic steady-state F* = F₀ + (β_I·ψ_I + β_P·ψ_P + β_D·ψ_D) / κ.
        Only well-defined for constant ψ inputs.
    inputs : dict
        Copy of the input ψ values (scalar or array).
    dt : float
        Time step used in integration.
    time : 1-D array
        Time axis corresponding to the trajectory.
    """

    trajectories: NDArray[np.float64]
    mean_trajectory: NDArray[np.float64]
    std_trajectory: NDArray[np.float64]
    steady_state_theoretical: float
    inputs: dict
    dt: float
    time: NDArray[np.float64] = field(init=False)

    def __post_init__(self) -> None:
        n_steps = self.mean_trajectory.shape[0] - 1
        self.time = np.arange(n_steps + 1) * self.dt

    def empirical_steady_state(self, last_fraction: float = 0.2) -> float:
        """Estimate steady-state from the tail of the mean trajectory.

        Parameters
        ----------
        last_fraction : float
            Fraction of the trajectory to average over.  Default 0.2
            (last 20%).

        Returns
        -------
        float
            Mean F over the tail of the mean trajectory.
        """
        n = len(self.mean_trajectory)
        tail_start = int(n * (1.0 - last_fraction))
        return float(self.mean_trajectory[tail_start:].mean())

    def summary(self) -> dict:
        """Return a summary statistics dictionary."""
        return {
            "n_trials": self.trajectories.shape[0],
            "n_steps": self.trajectories.shape[1] - 1,
            "dt": self.dt,
            "steady_state_theoretical": self.steady_state_theoretical,
            "steady_state_empirical": self.empirical_steady_state(),
            "mean_F_final": float(self.mean_trajectory[-1]),
            "std_F_final": float(self.std_trajectory[-1]),
        }


class SimulationEngine:
    """Run multiple independent realizations of the AICF SDE.

    Parameters
    ----------
    None
        All configuration is passed to :meth:`run`.
    """

    def run(
        self,
        model: FlowModel,
        inputs: dict[str, float | Sequence[float] | NDArray[np.float64]],
        n_steps: int,
        n_trials: int = 100,
        seed: int | None = 0,
        F_init: float | None = None,
    ) -> SimulationResult:
        """Run *n_trials* independent SDE realizations.

        Parameters
        ----------
        model : FlowModel
            Configured AICF model instance.  Its random state is **replaced**
            by a fresh seeded generator so that results are reproducible.
        inputs : dict
            Keys: 'psi_I', 'psi_P', 'psi_D'.  Values: scalar float or
            1-D array of length n_steps.
        n_steps : int
            Number of integration steps per trial.
        n_trials : int, optional
            Number of independent Monte Carlo trials.  Default 100.
        seed : int, optional
            Master seed for reproducibility.  Each trial uses a derived seed.
            Default 0.
        F_init : float, optional
            Initial condition for F.  Defaults to model.F0.

        Returns
        -------
        SimulationResult
        """
        psi_I = inputs.get("psi_I", 0.0)
        psi_P = inputs.get("psi_P", 0.0)
        psi_D = inputs.get("psi_D", 0.0)

        # Compute theoretical steady state (only valid for constant inputs)
        ss_I = float(np.mean(psi_I)) if not np.isscalar(psi_I) else float(psi_I)
        ss_P = float(np.mean(psi_P)) if not np.isscalar(psi_P) else float(psi_P)
        ss_D = float(np.mean(psi_D)) if not np.isscalar(psi_D) else float(psi_D)
        ss_theoretical = model.steady_state(ss_I, ss_P, ss_D)

        rng_master = np.random.default_rng(seed)
        trial_seeds = rng_master.integers(0, 2**31, size=n_trials)

        trajectories = np.empty((n_trials, n_steps + 1))

        for trial_idx, trial_seed in enumerate(trial_seeds):
            # Give each trial an independent RNG
            model.rng = np.random.default_rng(int(trial_seed))
            traj = model.simulate(
                psi_I=psi_I,
                psi_P=psi_P,
                psi_D=psi_D,
                n_steps=n_steps,
                F_init=F_init,
            )
            trajectories[trial_idx] = traj

        mean_traj = trajectories.mean(axis=0)
        std_traj = trajectories.std(axis=0)

        return SimulationResult(
            trajectories=trajectories,
            mean_trajectory=mean_traj,
            std_trajectory=std_traj,
            steady_state_theoretical=ss_theoretical,
            inputs={"psi_I": psi_I, "psi_P": psi_P, "psi_D": psi_D},
            dt=model.dt,
        )
