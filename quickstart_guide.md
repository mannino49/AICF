# AICF Model — Quickstart Guide

How to install, run, and use the Active Inference–Complexity Model of Flow on your local machine.

---

## 1. Install

You need Python 3.10 or higher. Clone the repo and install in editable mode:

```bash
git clone https://github.com/mannino49/AICF.git
cd AICF
pip install -e ".[dev]"
```

That installs the `aicf` package plus numpy, scipy, matplotlib, seaborn, pytest, and jupyter. The `-e` flag means editable — changes to the source code take effect immediately without reinstalling.

To verify the install worked:

```bash
pytest tests/ -v
```

All tests should pass.

---

## 2. Run the paper predictions (fastest way to see the model work)

```bash
python scripts/run_simulations.py
```

This runs all five key predictions from the paper and prints summary statistics. No data needed — it uses synthetic inputs to demonstrate the model's behavior (e.g., sweeping ψ_I from 0 to 1, comparing high-vs-low precision conditions, finding the optimal planning depth τ*).

---

## 3. Run a single simulation (the basic workflow)

Open a Python script or Jupyter notebook:

```python
from aicf.model.flow_model import FlowModel
import matplotlib.pyplot as plt

# Create a model with default parameters
model = FlowModel(
    kappa=1.0,       # how fast flow decays without input
    beta_I=1.0,      # how strongly informational layer drives flow
    beta_P=1.0,      # how strongly inferential layer drives flow
    beta_D=1.0,      # how strongly dynamical layer drives flow
    sigma_F=0.1,     # stochastic noise (set to 0.0 for deterministic)
    dt=0.01,         # integration time step
    seed=42          # for reproducibility
)

# Simulate: give it constant ψ values and run for 1000 steps
trajectory = model.simulate(
    psi_I=0.8,     # high means-ends coupling
    psi_P=0.7,     # high precision + automaticity
    psi_D=0.6,     # moderate complexity signature
    n_steps=1000
)

# Check steady state
print(f"Theoretical steady state: {model.steady_state(0.8, 0.7, 0.6):.3f}")
print(f"Final F value: {trajectory[-1]:.3f}")

# Plot
plt.figure(figsize=(8, 4))
plt.plot(trajectory)
plt.xlabel("Time step")
plt.ylabel("Flow state F(t)")
plt.title("Single AICF trajectory")
plt.tight_layout()
plt.show()
```

The three ψ inputs are all you need. Each is a number between 0 and 1:

| Input | What it means | 0 = | 1 = |
|-------|--------------|-----|-----|
| `psi_I` | Means-ends coupling | Actions and goals are unrelated | Knowing the goal perfectly determines the action |
| `psi_P` | Inferential precision + automaticity | Uncertain, deliberative policy selection | Confident, automatic action |
| `psi_D` | Neural complexity signature | No flow-associated dynamical pattern | Full flow-associated complexity profile |

---

## 4. Run multiple trials and get statistics

For proper analysis, run many stochastic realizations:

```python
from aicf.model.flow_model import FlowModel
from aicf.simulation.engine import SimulationEngine

model = FlowModel(kappa=1.0, beta_I=1.0, beta_P=1.0, beta_D=1.0, sigma_F=0.1)
engine = SimulationEngine()

result = engine.run(
    model=model,
    inputs={"psi_I": 0.8, "psi_P": 0.7, "psi_D": 0.6},
    n_steps=2000,
    n_trials=100,    # 100 independent realizations
    seed=42
)

# result.trajectories is shape (100, 2001) — all 100 runs
# result.mean_trajectory — average across trials
# result.std_trajectory — standard deviation across trials

print(result.summary())

# Plot mean ± 1 SD
import matplotlib.pyplot as plt
plt.fill_between(
    result.time,
    result.mean_trajectory - result.std_trajectory,
    result.mean_trajectory + result.std_trajectory,
    alpha=0.3, label="±1 SD"
)
plt.plot(result.time, result.mean_trajectory, label="Mean F(t)")
plt.xlabel("Time")
plt.ylabel("Flow state F(t)")
plt.legend()
plt.show()
```

---

## 5. Compute ψ values from actual data

The three ψ inputs can be computed from real experimental data using the layer classes. Here's how each one works.

### ψ_I — from behavioral task data

You need probability distributions over actions (means) and outcomes (goals). For example, from a video game task where you can estimate how often each action leads to each goal:

```python
from aicf.model.informational import InformationalLayer
import numpy as np

info = InformationalLayer()

# Example: a task with 4 possible actions and 3 possible outcomes
# p_joint[i, j] = probability of taking action i AND achieving outcome j
p_joint = np.array([
    [0.30, 0.05, 0.05],
    [0.05, 0.25, 0.02],
    [0.03, 0.02, 0.15],
    [0.02, 0.03, 0.03]
])

psi_I = info.compute_from_joint(p_joint)
print(f"ψ_I = {psi_I:.3f}")
```

Or if you already have marginal distributions:

```python
p_means = np.array([0.4, 0.32, 0.2, 0.08])   # action probabilities
p_ends = np.array([0.4, 0.35, 0.25])           # outcome probabilities

psi_I = info.compute(p_means, p_ends)
```

### ψ_P — from active inference estimates

You need estimates of policy precision and automaticity. These could come from fitting a softmax choice model to behavioral data, or from explicit active inference model inversion:

```python
from aicf.model.inferential import InferentialLayer
import numpy as np

inf = InferentialLayer(w_prec=0.5, w_auto=0.5)

# Policy negentropy: how peaked is the agent's policy distribution?
# Estimate from behavioral choice frequencies across conditions
policy_posterior = np.array([0.7, 0.15, 0.1, 0.05])  # peaked = high precision
n_pi = inf.compute_negentropy(policy_posterior)

# Automaticity: how close is the posterior to the habitual prior?
habitual_prior = np.array([0.6, 0.2, 0.12, 0.08])
A = inf.compute_automaticity(policy_posterior, habitual_prior)

psi_P = inf.compute(policy_posterior, habitual_prior)
print(f"n_π = {n_pi:.3f}, A = {A:.3f}, ψ_P = {psi_P:.3f}")
```

### ψ_D — from neural time series

You need a multivariate neural time series (fMRI BOLD, EEG, MEG). The layer computes entropy rate, dynamical complexity, and modal agility over sliding windows:

```python
from aicf.model.dynamical import DynamicalLayer
import numpy as np

dyn = DynamicalLayer(w_h=1/3, w_C=1/3, w_agil=1/3)

# neural_data shape: (n_timepoints, n_channels)
# e.g., 500 TRs × 116 brain regions from an fMRI session
neural_data = np.load("your_fmri_timeseries.npy")

psi_D_series = dyn.compute(
    time_series=neural_data,
    window_size=100,
    stride=10
)
# Returns a 1D array — one ψ_D value per window
print(f"Mean ψ_D = {psi_D_series.mean():.3f}")
```

### Putting it all together with real data

```python
from aicf.model.flow_model import FlowModel
from aicf.model.informational import InformationalLayer
from aicf.model.inferential import InferentialLayer
from aicf.model.dynamical import DynamicalLayer
from aicf.simulation.engine import SimulationEngine

# 1. Compute ψ values from your data
info = InformationalLayer()
psi_I = info.compute_from_joint(your_joint_distribution)

inf = InferentialLayer()
psi_P = inf.compute(your_policy_posterior, your_habitual_prior)

dyn = DynamicalLayer()
psi_D_timeseries = dyn.compute(your_neural_data, window_size=100)

# 2. Run the model
# Option A: constant ψ values (use means from your data)
model = FlowModel(seed=42)
engine = SimulationEngine()
result = engine.run(
    model=model,
    inputs={"psi_I": psi_I, "psi_P": psi_P, "psi_D": psi_D_timeseries.mean()},
    n_steps=2000,
    n_trials=50
)

# Option B: time-varying ψ_D (pass the full series)
trajectory = model.simulate(
    psi_I=psi_I,                    # scalar (constant across time)
    psi_P=psi_P,                    # scalar
    psi_D=psi_D_timeseries,         # array (time-varying)
    n_steps=len(psi_D_timeseries)
)

# 3. Compare model output to flow ratings
# trajectory gives you the model's prediction of F(t)
# Compare to self-report flow ratings, behavioral markers, etc.
```

---

## 6. Compare model variants

The paper argues the full three-layer model outperforms simpler alternatives. To test this:

```python
from aicf.model.reduced_models import (
    InformationOnlyModel,
    InferenceOnlyModel,
    DynamicsOnlyModel,
    InformationInferenceModel,
    InformationDynamicsModel,
    InferenceDynamicsModel,
)
from aicf.model.flow_model import FlowModel

models = {
    "Full (I+P+D)": FlowModel(),
    "I only": InformationOnlyModel(),
    "P only": InferenceOnlyModel(),
    "D only": DynamicsOnlyModel(),
    "I+P": InformationInferenceModel(),
    "I+D": InformationDynamicsModel(),
    "P+D": InferenceDynamicsModel(),
}

for name, m in models.items():
    traj = m.simulate(psi_I=0.8, psi_P=0.7, psi_D=0.6, n_steps=1000)
    print(f"{name:15s}  →  steady state = {traj[-200:].mean():.3f}")
```

---

## 7. Parameter sweeps

Explore how flow depends on combinations of the three inputs:

```python
from aicf.simulation.parameter_sweep import ParameterSweep
from aicf.model.flow_model import FlowModel

sweep = ParameterSweep()

# Sweep ψ_I × ψ_P at fixed ψ_D = 0.6
results = sweep.sweep_2d(
    model_class=FlowModel,
    param1_name="psi_I",
    param1_range=np.linspace(0, 1, 20),
    param2_name="psi_P",
    param2_range=np.linspace(0, 1, 20),
    fixed_params={"psi_D": 0.6},
    metric_fn=lambda r: r.empirical_steady_state()
)
# results is a 20×20 array you can plot as a heatmap
```

---

## 8. Run the tests

```bash
# Full test suite
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=aicf

# Just the dimensional consistency tests
pytest tests/test_dimensional_consistency.py -v
```

---

## Key things to know

**The model is an SDE.** Each run is stochastic (unless you set `sigma_F=0`). Always run multiple trials and report means ± SD. Always set a seed for reproducibility.

**ψ values are the bridge between data and the model.** The model itself is simple — it's an Ornstein-Uhlenbeck process driven by three inputs. The scientific substance is in how you compute those three inputs from real data. The layer classes (`InformationalLayer`, `InferentialLayer`, `DynamicalLayer`) handle that computation, but they need appropriate experimental data as input.

**The fitting pipeline is not built yet.** The `aicf/fitting/` module is a placeholder. Right now you can run forward simulations (given ψ values, predict F) but not inverse fitting (given observed flow indicators, estimate parameters). That's Phase 3 of the project.

**Steady-state is a useful sanity check.** For any constant set of ψ inputs, `model.steady_state()` gives you the analytical answer. The mean of a long simulation should converge to this value. If it doesn't, something is wrong.
