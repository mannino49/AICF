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

- Melnikoff, D. E., et al. (2022). The Minimal Mind. *Psychological Review*.
- Kotler, S., Parvizi-Wayne, D., Mannino, M., & Friston, K. (2025). *Active inference account of flow*.
- Hancock, P., Kee, S., Rosas, F., Girn, M., Kotler, S., Mannino, M., & Huskey, R. (2025). *Dynamical complexity signatures of flow*.
- Kotler, S., Mannino, M., Kelso, J. A. S., & Huskey, R. (2022). *First principles of flow neurobiology*. *Neuroscience & Biobehavioral Reviews*.
