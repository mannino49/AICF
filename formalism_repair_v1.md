# AICF Formalism Repair — v1

## 0. Summary of Changes

| Issue | Old formulation | Repaired formulation |
|-------|----------------|----------------------|
| Flow equation | Direct arithmetic combination F = f(I, P, D) | Latent state-space model: F(t) is a hidden state driven by three latent factors through distinct generative processes |
| P(t) term | π(t)/G(t) — conflates precision and free energy | Policy negentropy N_π(t) and automaticity index A(t), both properly defined from active inference |
| Planning depth τ | Generates Prediction 3 but absent from equations | Explicit parameter in the inferential layer; enters through expected free energy horizon and interacts with γ(t) |
| Dimensional analysis | Components on incommensurate scales | Each factor mapped to [0, 1] via information-theoretic normalization before entering the state equation |

---

## 1. Architecture: Latent State-Space Model

### 1.1 Why not a direct combination?

The previous formulation treated flow as an arithmetic function of three observable quantities: F(t) = f(I(t), P(t), D(t)). This has three problems:

1. **Ontological**: Flow is not directly observed — it is inferred from behavioral, neural, and phenomenological indicators. It should be modeled as a latent variable.
2. **Dimensional**: I(t) is in bits, P(t) was a dimensionless ratio of heterogeneous quantities, D(t) was in mixed complexity units. Summing or multiplying them is undefined without normalization.
3. **Dynamical**: A static mapping cannot capture the temporal evolution of flow — its onset, deepening, maintenance, and disruption.

### 1.2 The generative model

We define a **dynamic Bayesian network** (equivalently, a nonlinear state-space model) with three levels:

**Level 1 — Observations.** At each time step t, a vector of observable indicators **y**(t) ∈ ℝ^d is generated from the latent flow state:

```
y(t) = Λ · F(t) + ε(t),    ε(t) ~ N(0, R)
```

where **y**(t) includes neural observables (e.g., EEG complexity, fMRI connectivity), behavioral observables (e.g., performance metrics, response times), and phenomenological reports (e.g., flow questionnaire items). Λ is a loading matrix and R is observation noise covariance. (Note: we reserve C for the prior preference matrix in the active inference generative model; see Section 5.1.)

**Level 2 — Latent flow dynamics.** The scalar latent flow state F(t) ∈ ℝ evolves according to:

```
dF/dt = −κ[F(t) − F₀] + β_I · ψ_I(t) + β_P · ψ_P(t) + β_D · ψ_D(t) + σ_F · ξ(t)
```

where:
- κ > 0 is a **mean-reversion rate** (flow decays toward baseline F₀ without sustained input)
- ψ_I(t), ψ_P(t), ψ_D(t) ∈ [0, 1] are **normalized factor scores** from the three layers (defined below)
- β_I, β_P, β_D > 0 are **coupling strengths** (free parameters, estimable from data)
- ξ(t) is white noise; σ_F scales process stochasticity

This is an Ornstein-Uhlenbeck process with time-varying driving inputs. Key properties:
- **Mean-reversion**: flow is not self-sustaining — it requires continuous input from all three layers
- **Separability**: each layer contributes additively after normalization, but the factors themselves can be coupled at their own level
- **Stationarity condition**: in steady state, F* = F₀ + (β_I·ψ_I + β_P·ψ_P + β_D·ψ_D)/κ

*Note on the noise term*: The equation is an Itô stochastic differential equation. The white noise term σ_F · ξ(t) dt is formally written as σ_F · dW(t), where W(t) is a standard Wiener process. In Itô calculus, dW has units of √dt, so σ_F carries units of [F / √time], ensuring the integrated noise increments σ_F · ΔW ~ N(0, σ_F² · Δt) have units of [F]. This is the standard convention for SDEs and is dimensionally consistent with the drift terms when the equation is interpreted in integral form.

**Level 3 — Factor generative processes.** Each normalized factor ψ_I, ψ_P, ψ_D has its own generative model, defined in Sections 2–4 below. These are **not** simple transformations of observables — each involves latent variables specific to its theoretical layer.

### 1.3 Graphical model

```
     ψ_I(t) ─────┐
                   ↓
  F(t−1) ──→ F(t) ──→ F(t+1) ──→ ...
                   ↑
     ψ_P(t) ─────┤
                   ↑
     ψ_D(t) ─────┘
                   │
                   ↓
                 y(t)
```

The three factors drive the latent state; the latent state generates observations. F(t) also depends on F(t−1) through the mean-reversion dynamics (first-order Markov in continuous time).

---

## 2. Informational Layer: ψ_I(t)

### 2.1 Core quantity

The informational layer quantifies the **mutual information between means and ends** (Melnikoff et al., 2022). Let M denote the random variable over available means (actions, strategies) and E the random variable over desired ends (goals, outcomes). The raw quantity is:

```
I(M; E) = H(M) + H(E) − H(M, E)
```

where H(·) denotes Shannon entropy in bits.

### 2.2 Normalization

To obtain ψ_I(t) ∈ [0, 1], we normalize by the maximum achievable mutual information:

```
ψ_I(t) = I(M; E | s(t)) / min{H(M | s(t)), H(E | s(t))}
```

where conditioning on the agent's current state s(t) makes this context-dependent. The denominator is the minimum of the marginal entropies, which upper-bounds mutual information.

**Interpretation**: ψ_I = 1 when knowing the goal perfectly determines the action (or vice versa) — the means–ends coupling is maximally tight. ψ_I = 0 when means and ends are statistically independent — the agent has no informational link between what it's doing and what it's trying to achieve.

**Boundary convention**: If min{H(M|s), H(E|s)} = 0 (one variable is deterministic given state), then I(M; E) = 0 as well, and we define ψ_I = 0 by convention. Intuitively, if either there is only one possible action or only one possible goal, there is no informational coupling to speak of — the task is trivial.

### 2.3 Flow-relevance

Flow requires an intermediate optimum: ψ_I should be high enough that action is meaningfully directed (not random exploration) but the task should retain enough residual uncertainty to demand engagement. This is captured not by ψ_I alone but by its interaction with ψ_P (see Section 6.1).

---

## 3. Inferential Layer: ψ_P(t) — THE REPAIRED TERM

### 3.1 The problem with π(t)/G(t)

The previous P(t) = π(t)/G(t) was ill-defined because:
- π(t) is a probability distribution (a vector), not a scalar
- G(t) is expected free energy (also a vector, one per policy, or a scalar expectation) — dividing a distribution by an energy is dimensionally incoherent
- The ratio conflates two distinct quantities: the *sharpness* of policy selection and the *quality* of expected outcomes

### 3.2 Replacement: two-component inferential factor

We decompose the inferential layer into two sub-quantities, each with a clear definition in the active inference literature.

#### Component 1: Policy negentropy N_π(t)

In active inference, the posterior over policies is:

```
π_i(t) = σ(−γ(t) · G_i(t))    [softmax]
```

where:
- G_i(t) is the **expected free energy** of policy i, evaluated over a planning horizon of depth τ (see Section 5)
- γ(t) > 0 is the **policy precision** (inverse temperature of policy selection)

The **negentropy** of this posterior is:

```
N_π(t) = H_max − H[π(t)]
       = log(N_π) − [−Σ_i π_i(t) log π_i(t)]
```

where N_π is the number of available policies and H_max = log(N_π) is the entropy of the uniform distribution.

**Normalized form:**

```
n_π(t) = N_π(t) / H_max = 1 − H[π(t)] / log(N_π)    ∈ [0, 1]
```

**Interpretation**: n_π(t) = 1 when the agent selects a single policy with certainty (zero entropy). n_π(t) = 0 when the agent is maximally uncertain (uniform over all policies). In flow, n_π should be high — the agent knows what to do and commits without deliberative vacillation.

**Boundary convention**: If N_π = 1 (only one policy available), define n_π = 1 by convention — the agent has no choice, so selection is trivially certain. This is a degenerate case unlikely to correspond to flow (no challenge), but the convention avoids division by zero.

**Why this is the right quantity**: n_π(t) naturally combines the effects of policy precision γ(t) and the discriminability of expected free energies {G_i}. High γ with well-separated G values yields high n_π — exactly the flow condition of confident, precise action selection.

#### Component 2: Automaticity index A(t)

Parvizi-Wayne, Kotler, Mannino & Friston (2025) argue that flow involves a shift from deliberative (model-based) to habitual (model-free) control. We formalize this as:

Let π_habit(t) be the **habitual policy prior** — the distribution over policies learned through repeated experience (reflexive, fast, model-free). Let π_post(t) be the **posterior policy** after active inference (incorporating expected free energy evaluation).

The automaticity index is:

```
A(t) = 1 − D_KL[π_post(t) || π_habit(t)] / D_KL_max
```

where D_KL_max is a normalization constant. We set D_KL_max = log(N_π), the theoretical maximum KL divergence between any two distributions over N_π policies (achieved when π_post is a point mass on a policy to which π_habit assigns minimal probability).

**Boundary convention**: If N_π = 1, define A(t) = 1 (trivially automatic). If D_KL[π_post || π_habit] > D_KL_max due to numerical issues, clamp A(t) = 0.

**Normalized form:**

```
A(t) ∈ [0, 1]
```

**Interpretation**: A(t) = 1 when the posterior policy is identical to the habitual prior — the agent's deliberative inference adds nothing beyond what habit already dictates. A(t) = 0 when the posterior is maximally divergent from habit — the agent is relying entirely on deliberative, model-based control. In flow, A(t) should be high — behavior is automatic, effortless, and habitual priors dominate.

#### Combining into ψ_P(t)

The two components contribute to the normalized inferential factor:

```
ψ_P(t) = w_prec · n_π(t) + w_auto · A(t)
```

where w_prec + w_auto = 1, w_prec, w_auto > 0 are mixing weights (free parameters).

**Rationale for two components rather than one**: n_π(t) and A(t) capture overlapping but distinct aspects of the inferential signature of flow. A subject could have high n_π (confident policy selection) through effortful deliberation (low A) — this characterizes expert problem-solving under challenge, not flow. Conversely, high A (automatic behavior) with moderate n_π characterizes well-learned but non-engaging routine. Flow requires both: confident selection (high n_π) that is also automatic (high A).

#### 3.3 Why not option (a) alone?

Option (a) — using only the negentropy H_π — captures the *sharpness* of policy selection but misses the *mechanism*. Two agents with identical policy posteriors (same H_π) could arrive there by different routes: one through deep deliberation, the other through habit. Only the latter is flow. The automaticity index A(t) disambiguates these routes.

However, if parsimony is prioritized (e.g., for the computational model where estimating habitual priors is difficult), **ψ_P(t) = n_π(t) is a defensible simplification** — with the understanding that it captures a necessary but not sufficient condition for the inferential signature of flow.

---

## 4. Dynamical Layer: ψ_D(t)

### 4.1 Core quantities

The dynamical layer captures whole-brain complexity signatures associated with flow (Hancock, Kee, Rosas, Girn, Kotler, Mannino & Huskey, 2025). Three sub-quantities, each capturing a distinct aspect of neural dynamics:

#### (a) Entropy rate h(t)

The entropy rate of the neural time series captures the **irreducible unpredictability** of brain dynamics:

```
h(t) = lim_{n→∞} H(X_n | X_{n−1}, ..., X_1)
```

In practice, estimated from EEG/MEG/fMRI time series over a sliding window centered at t.

#### (b) Dynamical complexity C_D(t)

Following Rosas et al., dynamical complexity quantifies the **excess entropy** or **statistical complexity** of the neural process — the amount of information that the past carries about the future:

```
C_D(t) = I(X_past; X_future) = H(X_future) − h(t) · T
```

(where T is the future horizon). This captures the richness of temporal structure beyond mere unpredictability.

#### (c) Modal agility A_modal(t)

Modal agility captures **mode-switching dynamics** — how fluidly the brain transitions between distinct dynamical regimes (e.g., metastable states identified via HMM or recurrence analysis):

```
A_modal(t) = −Σ_{j,k} T_{jk}(t) · log T_{jk}(t)
```

where T_{jk}(t) is the empirical transition probability from mode j to mode k in a window around t. This is the entropy of the transition matrix — higher values indicate more fluid, less stereotyped mode-switching.

### 4.2 Normalization and combination

Each sub-quantity is z-scored within a recording session and passed through a logistic sigmoid to map to [0, 1]:

```
ψ_D(t) = w_h · σ(z_h(t)) + w_C · σ(z_C(t)) + w_agil · σ(z_A(t))
```

where σ(·) is the logistic function, z_x denotes z-scoring, and w_h + w_C + w_agil = 1.

### 4.3 Flow-relevance

Hancock et al. (2025) found that flow is associated with elevated dynamical complexity and specific patterns of modal agility — not simply high or low entropy, but a particular *regime* of neural dynamics. The three-component structure of ψ_D allows the model to capture this profile rather than reducing it to a single scalar.

---

## 5. Planning Depth τ — Integration into the Formal Model

### 5.1 Where τ enters

Planning depth τ appears in the computation of expected free energy. For policy π_i, the expected free energy evaluated over horizon τ is:

```
G_i(t; τ) = Σ_{t'=t+1}^{t+τ} E_{Q(o,s|π_i)} [log Q(s_{t'} | π_i) − log P(o_{t'}, s_{t'} | C)]
```

where:
- Q(s_{t'} | π_i) is the approximate posterior over hidden states under policy i
- P(o_{t'}, s_{t'} | C) is the prior preference model (C encodes preferred outcomes)
- The sum runs from the next time step to τ steps ahead

**Longer τ → more future consequences considered → better-informed but more computationally expensive policy evaluation.**

### 5.2 The τ–γ interaction and computational cost

Planning depth improves policy evaluation (deeper search → better-informed action) but also imposes a **computational cost** that degrades precision. Deeper planning requires maintaining more states in working memory, increasing uncertainty about distal predictions, and consuming cognitive resources that would otherwise support precise action selection.

We formalize this tradeoff. The **effective precision** γ_eff available for policy selection is:

```
γ_eff(t) = γ(t) · [1 − exp(−τ(t) / τ₀)] · exp(−τ(t) / τ_max)
```

The first factor, [1 − exp(−τ/τ₀)], is the **planning benefit**: it increases with τ and saturates — shallow planning is poorly informed, but returns diminish.

The second factor, exp(−τ/τ_max), is the **computational cost**: it decreases exponentially with τ — deep planning degrades precision by consuming cognitive resources.

The product has a single interior maximum at:

```
τ* = τ₀ · τ_max / (τ_max − τ₀) · ln(τ_max / τ₀)    [for τ_max > τ₀]
```

This is the **optimal planning depth** — the value of τ that maximizes effective precision for a given γ(t).

The policy posterior is then:

```
π_i(t) = softmax(−γ_eff(t) · G_i(t; τ(t)))
```

which means n_π(t) and A(t) are both implicitly functions of τ(t) through γ_eff. Planning depth does not need a separate term in the flow equation — it modulates ψ_P(t) from within.

**Key phenomenological predictions from the τ–γ_eff surface:**

| Condition | γ(t) | τ(t) | γ_eff(t) | Phenomenology |
|-----------|-------|------|----------|---------------|
| **Flow** | High | ≈ τ* | High | Confident action with optimally deep, effortless planning |
| **Anxiety** | Low | > τ* | Very low | Excessively deep planning + low base precision = paralysis |
| **Boredom** | Variable | < τ* | Low | Disengaged, shallow planning — insufficient strategic depth |
| **Hyperfocus** | High | ≪ τ* | Moderate | Confident but myopic — absorbed without strategic foresight |

### 5.3 Generating Prediction 3

**Prediction 3**: *Flow is associated with an optimal planning horizon — deep enough for strategic coherence, shallow enough for automaticity.*

This is now formally grounded by the **non-monotonic** relationship between τ and γ_eff. The benefit-cost tradeoff creates a single maximum at τ*. Agents in flow operate near τ*, achieving deep-enough planning to sustain strategic coherence while avoiding the precision-degrading costs of excessively deep look-ahead. This is not merely asserted — the optimum is a mathematical consequence of the competing exponentials in the γ_eff equation.

The parameter τ_max has a natural neurobiological interpretation: it reflects working memory capacity and the fidelity of the agent's generative model for predicting distal outcomes. Higher τ_max (better model, more working memory) shifts τ* rightward, allowing deeper planning before costs dominate — consistent with the observation that experts in flow can plan further ahead than novices.

---

## 6. Dimensional Analysis and Normalization

### 6.1 Normalization summary

| Factor | Raw quantity | Normalization | Range | Units | Edge case |
|--------|-------------|---------------|-------|-------|-----------|
| ψ_I(t) | I(M; E \| s(t)) | Divide by min{H(M\|s), H(E\|s)} | [0, 1] | Dimensionless | 0/0 → ψ_I = 0 |
| n_π(t) | H_max − H[π(t)] | Divide by H_max = log(N_π) | [0, 1] | Dimensionless | N_π = 1 → n_π = 1 |
| A(t) | D_KL[π_post \|\| π_habit] | 1 − D_KL/log(N_π) | [0, 1] | Dimensionless | N_π = 1 → A = 1; clamp at 0 |
| ψ_P(t) | w_prec·n_π + w_auto·A | Convex combination | [0, 1] | Dimensionless | — |
| ψ_D(t) | h, C_D, A_modal | z-score → logistic sigmoid → convex combination | [0, 1] | Dimensionless | — |

All three factors ψ_I, ψ_P, ψ_D ∈ [0, 1] and are dimensionless. The coupling strengths β_I, β_P, β_D in the state equation carry units of [flow units / time], and κ carries units of [1/time], ensuring dimensional consistency:

```
dF/dt  =  −κ·[F − F₀]  +  β_I·ψ_I  +  β_P·ψ_P  +  β_D·ψ_D  +  σ_F·dW/dt
[F/t]     [1/t]·[F]       [F/t]·[1]    [F/t]·[1]    [F/t]·[1]    [F/√t]·[1/√t]
```

The drift terms (first four) have units [F/time]. The noise term, interpreted as an Itô SDE (see note in Section 1.2), is dimensionally consistent in integral form: ∫σ_F dW has units [F/√time]·[√time] = [F]. ✓

### 6.2 Scale of F(t)

F(t) is a latent variable with no inherent physical units. By convention, we set F₀ = 0 (baseline = no flow) and let the coupling strengths and observation model C determine the scale. Under the steady-state condition with all factors at their flow-optimal values, F* = (β_I + β_P + β_D)/κ defines the "deep flow" level of the latent state.

---

## 7. Full Model Summary

### State equation (continuous-time)

```
dF/dt = −κ[F(t) − F₀] + β_I · ψ_I(t) + β_P · ψ_P(t) + β_D · ψ_D(t) + σ_F · ξ(t)
```

### Observation equation

```
y(t) = Λ · F(t) + ε(t),    ε ~ N(0, R)
```

### Factor definitions

```
ψ_I(t) = I(M; E | s(t)) / min{H(M|s(t)), H(E|s(t))}

ψ_P(t) = w_prec · n_π(t) + w_auto · A(t)
  where  n_π(t) = 1 − H[π(t)] / log(N_π)
         A(t)   = 1 − D_KL[π_post(t) || π_habit(t)] / log(N_π)
         π_i(t) = softmax(−γ_eff(t) · G_i(t; τ(t)))
         γ_eff(t) = γ(t) · [1 − exp(−τ/τ₀)] · exp(−τ/τ_max)

ψ_D(t) = w_h · σ(z_h(t)) + w_C · σ(z_C(t)) + w_agil · σ(z_A(t))
  where  h(t)       = entropy rate
         C_D(t)     = dynamical complexity
         A_modal(t) = modal agility (transition entropy)
```

### Free parameters

| Parameter | Role | Estimable from |
|-----------|------|----------------|
| κ | Mean-reversion rate | Time-series of flow onset/offset |
| β_I, β_P, β_D | Coupling strengths | Cross-validated regression on flow indicators |
| w_prec, w_auto | Inferential sub-component weights (sum to 1) | Factor analysis on precision/automaticity measures |
| w_h, w_C, w_agil | Dynamical sub-component weights (sum to 1) | Factor analysis on complexity measures |
| γ(t) | Base policy precision | Behavioral choice data (softmax fits) |
| τ(t) | Planning depth | Model comparison across horizon lengths |
| τ₀ | Planning benefit scale | Fitted from τ–performance curves |
| τ_max | Computational cost scale (working memory capacity) | Fitted from τ–precision degradation data |
| σ_F | Process noise scale | State-space model fitting (e.g., Kalman filter) |
| Λ, R | Observation model (loading matrix, noise covariance) | Factor analysis / measurement model |
| F₀ | Baseline flow | Set to 0 by convention |

### Prediction mapping

| Prediction | Formal grounding |
|------------|-----------------|
| P1: Flow requires tight means–ends coupling | High ψ_I(t) is necessary for sustained F(t) above baseline |
| P2: Flow involves high-precision, automatic policy selection | High ψ_P(t) requires both high n_π (precision) and high A (automaticity) |
| P3: Flow has an optimal planning horizon | The τ–γ interaction in Γ(t) creates a sweet spot; too-shallow or too-deep planning reduces ψ_P |
| P4: Flow is associated with specific complexity signatures | ψ_D(t) captures the entropy rate / dynamical complexity / modal agility profile |
| P5: Flow is dynamically maintained, not a static state | The state equation with mean-reversion κ requires continuous driving input |

---

## 8. Comparison with Previous Formulation

### What was wrong:
1. **F = f(I, P, D)** — a static mapping that couldn't capture flow dynamics (onset, maintenance, disruption)
2. **P(t) = π(t)/G(t)** — dimensionally incoherent; conflated a distribution with a scalar energy
3. **Planning depth τ** — invoked in predictions but absent from equations
4. **No normalization** — factors on incommensurate scales were combined directly

### What is now fixed:
1. **State-space model** — F(t) is a latent variable with proper dynamics (mean-reversion + driving inputs + Itô noise)
2. **P(t) → ψ_P(t) = w_prec·n_π + w_auto·A** — two properly defined quantities from active inference: policy negentropy and automaticity index
3. **τ integrated with non-monotonic cost** — enters through γ_eff(t) = γ · benefit(τ) · cost(τ), producing a formal optimum at τ* that grounds Prediction 3
4. **All factors normalized to [0, 1]** — information-theoretic normalization for ψ_I and ψ_P; z-score + sigmoid for ψ_D; boundary conventions for degenerate cases
5. **Symbol hygiene** — Λ for observation loading (vs. C for preference matrix); w_prec/w_auto (vs. w_agil) to avoid cross-layer collision

### What is preserved:
- The three-layer architecture (informational, inferential, dynamical) is unchanged
- The theoretical grounding in Melnikoff et al., Parvizi-Wayne et al., and Hancock et al. is unchanged
- All five predictions are still derivable — now with explicit formal grounding

---

## 9. Next Steps

1. **Write the state-space model as a computational simulation** (Python/Julia): implement the SDE, test parameter regimes, verify that the model produces flow-like dynamics
2. **Derive steady-state and stability conditions**: what parameter combinations sustain flow? What perturbations disrupt it?
3. **Connect to empirical observables**: specify the observation matrix C for specific experimental paradigms (EEG, fMRI, behavioral)
4. **Parameter estimation protocol**: outline how each free parameter can be estimated from data (model fitting, cross-validation)
5. **Evaluate whether β coefficients should be state-dependent**: could coupling strengths change as a function of F(t) itself (e.g., flow deepening its own maintenance)?
