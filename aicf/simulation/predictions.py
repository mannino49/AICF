"""
Key Model Predictions as Simulation Scripts
============================================

Each function simulates a specific theoretical prediction from the AICF model
(formalism_repair_v1.md §7, Prediction mapping table).

Prediction 1 — Tight means–ends coupling sustains flow above baseline.
Prediction 2 — Flow requires both high n_π AND high A (not either alone).
Prediction 3 — Flow has an optimal planning horizon (γ_eff peak at τ*).
Prediction 4 — Flow is associated with specific complexity signatures in ψ_D.
Prediction 5 — Flow is dynamically maintained; decays without sustained input.

Each function returns a dict of arrays suitable for direct plotting.

Reference
---------
formalism_repair_v1.md §7 Prediction mapping.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from aicf.model.flow_model import FlowModel
from aicf.model.inferential import InferentialLayer
from aicf.simulation.engine import SimulationEngine, SimulationResult


def simulate_prediction_1(
    psi_I_values: NDArray[np.float64] | None = None,
    n_steps: int = 3000,
    n_trials: int = 50,
    seed: int = 1,
) -> dict:
    """P1: High ψ_I sustains flow above baseline.

    Sweeps ψ_I from 0 to 1 (with ψ_P = ψ_D fixed at flow-optimal values)
    and returns the empirical steady-state F* for each value.

    Returns
    -------
    dict with keys:
        'psi_I_values' : 1-D array
        'steady_state' : 1-D array of empirical F*
        'theoretical'  : 1-D array of analytic F*
    """
    if psi_I_values is None:
        psi_I_values = np.linspace(0.0, 1.0, 21)

    engine = SimulationEngine()
    model_base = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0, sigma_F=0.1)

    empirical = []
    theoretical = []

    for psi_I in psi_I_values:
        res = engine.run(
            model=model_base,
            inputs={"psi_I": psi_I, "psi_P": 0.7, "psi_D": 0.7},
            n_steps=n_steps,
            n_trials=n_trials,
            seed=seed,
        )
        empirical.append(res.empirical_steady_state())
        theoretical.append(model_base.steady_state(psi_I, 0.7, 0.7))

    return {
        "psi_I_values": psi_I_values,
        "steady_state_empirical": np.array(empirical),
        "steady_state_theoretical": np.array(theoretical),
    }


def simulate_prediction_2(
    n_steps: int = 3000,
    n_trials: int = 50,
    seed: int = 2,
) -> dict:
    """P2: Flow requires both high n_π AND high A.

    Compares four conditions:
        (a) high n_π, high A  (flow)
        (b) high n_π, low A   (effortful deliberation)
        (c) low n_π, high A   (automatic but disengaged)
        (d) low n_π, low A    (neither)

    ψ_P is constructed from the InferentialLayer with manipulated
    component values.

    Returns
    -------
    dict with keys 'conditions' (list of labels) and 'steady_states' (array).
    """
    inf_layer = InferentialLayer(w_prec=0.5, w_auto=0.5)

    def make_psi_P(n_pi: float, A: float) -> float:
        return float(np.clip(inf_layer.w_prec * n_pi + inf_layer.w_auto * A, 0, 1))

    conditions = {
        "flow (high n_π, high A)": make_psi_P(0.9, 0.9),
        "deliberate (high n_π, low A)": make_psi_P(0.9, 0.1),
        "automatic (low n_π, high A)": make_psi_P(0.1, 0.9),
        "disengaged (low n_π, low A)": make_psi_P(0.1, 0.1),
    }

    engine = SimulationEngine()
    model = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0, sigma_F=0.1)

    results = {}
    for label, psi_P in conditions.items():
        res = engine.run(
            model=model,
            inputs={"psi_I": 0.7, "psi_P": psi_P, "psi_D": 0.7},
            n_steps=n_steps,
            n_trials=n_trials,
            seed=seed,
        )
        results[label] = res.empirical_steady_state()

    return {
        "conditions": list(results.keys()),
        "steady_states": np.array(list(results.values())),
        "psi_P_values": np.array(list(conditions.values())),
    }


def simulate_prediction_3(
    tau_values: NDArray[np.float64] | None = None,
    tau_0: float = 2.0,
    tau_max: float = 10.0,
    gamma: float = 5.0,
) -> dict:
    """P3: Non-monotonic γ_eff(τ) — optimal planning depth at τ*.

    Sweeps τ and computes γ_eff for each value, demonstrating the
    interior maximum at τ*.

    Returns
    -------
    dict with keys:
        'tau_values'  : 1-D array
        'gamma_eff'   : 1-D array
        'tau_star'    : float (analytical optimum)
        'gamma_eff_star': float (maximum γ_eff)
    """
    if tau_values is None:
        tau_values = np.linspace(0.1, 20.0, 200)

    inf_layer = InferentialLayer(tau_0=tau_0, tau_max=tau_max)
    tau_star = inf_layer.optimal_planning_depth()

    gamma_eff = np.array([
        inf_layer.compute_effective_precision(gamma, tau)
        for tau in tau_values
    ])

    return {
        "tau_values": tau_values,
        "gamma_eff": gamma_eff,
        "tau_star": tau_star,
        "gamma_eff_star": float(inf_layer.compute_effective_precision(gamma, tau_star)),
        "gamma": gamma,
        "tau_0": tau_0,
        "tau_max": tau_max,
    }


def simulate_prediction_5(
    n_steps_flow: int = 2000,
    n_steps_decay: int = 2000,
    n_trials: int = 50,
    seed: int = 5,
) -> dict:
    """P5: Flow decays without sustained input (mean-reversion).

    Phase 1: Simulate flow with full ψ inputs for n_steps_flow steps.
    Phase 2: Remove all inputs (ψ = 0) and observe decay to baseline.

    Returns
    -------
    dict with keys:
        'time'         : full time axis
        'mean_F'       : mean trajectory across both phases
        'std_F'        : std trajectory
        'F0'           : baseline level
        'phase_switch' : time index where inputs were removed
    """
    engine = SimulationEngine()
    model = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0,
                      sigma_F=0.1, F0=0.0, dt=0.01)

    rng = np.random.default_rng(seed)
    trial_seeds = rng.integers(0, 2**31, size=n_trials)

    all_trajectories = []

    for trial_seed in trial_seeds:
        model.rng = np.random.default_rng(int(trial_seed))

        # Phase 1: drive flow
        traj_flow = model.simulate(psi_I=0.8, psi_P=0.8, psi_D=0.8,
                                   n_steps=n_steps_flow, F_init=0.0)
        F_at_switch = float(traj_flow[-1])

        # Phase 2: remove inputs, observe decay
        traj_decay = model.simulate(psi_I=0.0, psi_P=0.0, psi_D=0.0,
                                    n_steps=n_steps_decay, F_init=F_at_switch)

        full = np.concatenate([traj_flow, traj_decay[1:]])
        all_trajectories.append(full)

    trajectories = np.array(all_trajectories)
    mean_F = trajectories.mean(axis=0)
    std_F = trajectories.std(axis=0)
    dt = model.dt
    n_total = n_steps_flow + n_steps_decay

    return {
        "time": np.arange(n_total + 1) * dt,
        "mean_F": mean_F,
        "std_F": std_F,
        "F0": model.F0,
        "phase_switch": n_steps_flow,
        "dt": dt,
    }
