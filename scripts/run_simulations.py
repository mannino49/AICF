"""
Main Simulation Script
=======================

Entry point for running all AICF model simulations and generating the
data underlying the paper's figures.  Each prediction is simulated
independently and results are printed as a summary.

Usage
-----
    python scripts/run_simulations.py

All outputs are printed to stdout.  Figure generation (Phase 2) will
read from these results.
"""

import numpy as np
from aicf.simulation.predictions import (
    simulate_prediction_1,
    simulate_prediction_2,
    simulate_prediction_3,
    simulate_prediction_5,
)


def run_all() -> None:
    print("=" * 60)
    print("AICF Model Simulations")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Prediction 1: Means-ends coupling sustains flow
    # ------------------------------------------------------------------
    print("\n[P1] Sweeping ψ_I (means-ends coupling) ...")
    r1 = simulate_prediction_1(n_steps=3000, n_trials=30, seed=1)
    print(f"  ψ_I range:         [{r1['psi_I_values'].min():.2f}, {r1['psi_I_values'].max():.2f}]")
    print(f"  F* range:          [{r1['steady_state_empirical'].min():.3f}, {r1['steady_state_empirical'].max():.3f}]")
    print(f"  Theoretical F*:    [{r1['steady_state_theoretical'].min():.3f}, {r1['steady_state_theoretical'].max():.3f}]")

    # ------------------------------------------------------------------
    # Prediction 2: Flow requires both high n_π AND high A
    # ------------------------------------------------------------------
    print("\n[P2] Four inferential conditions ...")
    r2 = simulate_prediction_2(n_steps=3000, n_trials=30, seed=2)
    for cond, ss in zip(r2["conditions"], r2["steady_states"]):
        print(f"  {cond:<40s}  F* = {ss:.3f}")

    # ------------------------------------------------------------------
    # Prediction 3: Optimal planning depth (non-monotonic γ_eff)
    # ------------------------------------------------------------------
    print("\n[P3] Planning depth sweep ...")
    r3 = simulate_prediction_3(gamma=5.0)
    print(f"  Analytic τ* =      {r3['tau_star']:.4f}")
    print(f"  γ_eff at τ* =      {r3['gamma_eff_star']:.4f}")
    tau_numeric = r3["tau_values"][np.argmax(r3["gamma_eff"])]
    print(f"  Numeric τ* =       {tau_numeric:.4f}")
    print(f"  Max γ_eff =        {r3['gamma_eff'].max():.4f}")

    # ------------------------------------------------------------------
    # Prediction 5: Flow decays without sustained input
    # ------------------------------------------------------------------
    print("\n[P5] Flow onset and decay ...")
    r5 = simulate_prediction_5(n_steps_flow=2000, n_steps_decay=2000,
                                n_trials=30, seed=5)
    F_peak = r5["mean_F"][:r5["phase_switch"] + 1].max()
    F_final = r5["mean_F"][-1]
    print(f"  Peak F during flow: {F_peak:.3f}")
    print(f"  Final F after decay: {F_final:.4f}  (baseline = {r5['F0']:.1f})")

    print("\n" + "=" * 60)
    print("All simulations complete.")


if __name__ == "__main__":
    run_all()
