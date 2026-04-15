# AICF: Active Inference–Complexity Model of Flow

A rigorous computational implementation of the Active Inference–Complexity Model of Flow (AICF), modeling psychological flow as a latent dynamical state driven by three normalized factor layers through an Ornstein-Uhlenbeck stochastic differential equation.

## Theoretical Overview

Flow is modeled as a hidden latent state F(t) governed by:

```
dF/dt = −κ[F(t) − F₀] + β_I·ψ_I(t) + β_P·ψ_P(t) + β_D·ψ_D(t) + σ_F·dW(t)
```

Three normalized factor layers (each ∈ [0, 1]) drive the latent state:

| Layer | Symbol | Theory | Quantity |
|-------|--------|---------|----------|
| **Informational** | ψ_I(t) | Melnikoff et al. (2022) | Normalized mutual information I(M; E) between means and ends |
| **Inferential** | ψ_P(t) | Parvizi-Wayne, Kotler, Mannino & Friston (2025) | Policy negentropy + automaticity index from active inference |
| **Dynamical** | ψ_D(t) | Hancock et al. (2025) | Entropy rate, dynamical complexity, and modal agility |

See `formalism_repair_v1.md` in the companion paper directory for the full mathematical specification.

## Repository Structure

```
aicf-flow-model/
├── aicf/
│   ├── model/
│   │   ├── informational.py   # ψ_I(t): mutual information layer
│   │   ├── inferential.py     # ψ_P(t): active inference layer
│   │   ├── dynamical.py       # ψ_D(t): complexity layer
│   │   ├── flow_model.py      # Full AICF state-space model
│   │   ├── coupling.py        # Inter-layer coupling alternatives
│   │   └── reduced_models.py  # Ablation models for comparison
│   └── simulation/
│       ├── engine.py          # SDE simulation runner
│       ├── parameter_sweep.py # Grid search infrastructure
│       └── predictions.py     # Key model predictions as scripts
├── tests/                     # pytest test suite (>90% coverage target)
├── scripts/
│   └── run_simulations.py     # Main entry point for paper figures
└── notebooks/
    └── 01_model_overview.ipynb
```

## Installation

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from aicf.model.flow_model import FlowModel

model = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0, sigma_F=0.1)
trajectory = model.simulate(psi_I=0.8, psi_P=0.7, psi_D=0.6, n_steps=1000)
```

## Running Tests

```bash
pytest tests/ -v --cov=aicf
```

## Key References

Melnikoff, D. E., Carlson, R. W., & Stillman, P. E. (2022). A computational theory of the subjective experience of flow. Nature Communications, 13, 2252. https://doi.org/10.1038/s41467-022-29742-2
Kotler, S., Mannino, M., Kelso, J. A. S., & Huskey, R. (2022). First few seconds for flow: A comprehensive proposal of the neurobiology and neurodynamics of state onset. Neuroscience & Biobehavioral Reviews, 143, 104956. https://doi.org/10.1016/j.neubiorev.2022.104956
Kotler, S., Parvizi-Wayne, D., Mannino, M., & Friston, K. (2025). Flow and intuition: a systems neuroscience comparison. Neuroscience of Consciousness, 2025(1), niae040.
Hancock, F., Kee, R., Rosas, F., Girn, M., Kotler, S., Mannino, M., & Huskey, R. (2025). A Complexity-Science Framework for Studying Flow: Using Media to Probe Brain–Phenomenology Dynamics. bioRxiv, 2025-07.
Durcan, O., Holland, P., & Bhattacharya, J. (2024). A framework for neurophysiological experiments on flow states. Communications Psychology, 2(1), 66.

