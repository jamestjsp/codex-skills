# Benchmark Family Playbook

Use this reference when a user names a standard nonconvex benchmark family or
asks for a benchmark-grade convexification. It distills the raw
`nonconvex-benchmark-suite-report.md` into routing guidance.

## Table of Contents

- How to grade the transformation
- Family routing table
- Exact or near-exact reformulations
- Lifted convex relaxations
- Certifiable relaxations and recovery
- Lower-bounding convexifications
- CVXPY implementation cautions

## How to Grade the Transformation

For every benchmark instance, report five artifacts:

- The transformed convex model.
- The declared type: exact reformulation, convex relaxation, mixed-integer
  convex model, sequential/local method, or lower bound.
- The solver class: LP, QP, SOCP, SDP, EXP, POW, MICP, or outer-loop DCP.
- A recovered solution in the original variables when the convex model is only
  a relaxation.
- A quality witness: dual bound, rank certificate, rounded objective,
  feasibility residual, or local convergence trace.

Do not score a relaxation as an exact convexification unless you can state the
conditions that prove tightness.

## Family Routing Table

| Family | First route | Type to declare | Main witness |
|---|---|---|---|
| H2 state feedback | `Y = K P` plus Schur complement LMI | Exact LMI/SDP under full-state assumptions | Recovered `K = Y P^{-1}`, Lyapunov residual |
| Matrix completion | Nuclear norm | Convex surrogate, exact under incoherence/sampling assumptions | Held-out error, rank, observation residual |
| Phase retrieval | PhaseLift or PhaseMax | SDP relaxation or anchored convex program | Rank-one certificate, phase-invariant error |
| Sparse PCA | Lift `X = x x.T`, drop rank, add sparsity surrogate | SDP relaxation | Rank/eigenvector recovery, support quality |
| Max-Cut/QUBO | Goemans-Williamson SDP | SDP upper bound plus randomized rounding | Cut value versus SDP bound |
| AC-OPF | SDP, SOC, or QC voltage relaxation | Convex relaxation/lower bound | AC feasibility after recovery, optimality gap |
| Pose graph optimization | Rotation synchronization SDP / SE-Sync style relaxation | Certifiable SDP relaxation | Rank/certificate, trajectory residual |
| Essential matrix | QCQP-to-SDP or orthogonal-space relaxation | SDP relaxation | Rank/tightness, epipolar residual, pose error |
| Cardinality portfolio | MIQP/MISOCP or perspective-strengthened relaxation | Mixed-integer convex model or relaxation | MIP gap, cardinality/buy-in feasibility |
| Lennard-Jones clusters | alphaBB or interval convex underestimators | Lower-bounding relaxation in branch-and-bound | Valid lower bound, gap to best feasible energy |

## Exact or Near-Exact Reformulations

### H2 State Feedback

Original nonconvexity: `K P` and quadratic `K.T R K` terms.

Use:

```text
Y = K P
K = Y P^{-1}
```

Typical convexified continuous-time LMI:

```text
A P + P A.T + B Y + Y.T B.T + W << 0
P >> 0
[[Z, R^{1/2} Y],
 [Y.T R^{1/2}, P]] >> 0
minimize trace(Q P) + trace(Z)
```

Report this as exact for the stated full-state formulation. Recover `K` with a
linear solve rather than an explicit inverse.

### Matrix Completion

Original nonconvexity: `rank(X) <= r` or factorization `X = U V.T`.

Use nuclear norm:

```python
objective = cp.Minimize(cp.norm(X, "nuc"))
constraints = [cp.multiply(mask, X) == cp.multiply(mask, M)]
```

For noisy data:

```python
objective = cp.Minimize(
    0.5 * cp.sum_squares(cp.multiply(mask, X - M)) + lam * cp.norm(X, "nuc")
)
```

Declare it as a convex surrogate unless the instance assumptions justify exact
recovery.

## Lifted Convex Relaxations

### Phase Retrieval

For real measurements `y_i = (a_i.T x)^2`, lift `X = x x.T`:

```python
X = cp.Variable((n, n), PSD=True)
constraints = [cp.trace(np.outer(a_i, a_i) @ X) == y_i for i in range(m)]
objective = cp.Minimize(cp.trace(X))
```

Drop `rank(X) == 1`. Recover `x` from the leading eigenvector when `X` is
nearly rank one. For large instances, consider PhaseMax if a good anchor is
available because it avoids the `n x n` SDP lift.

### Sparse PCA

Lift `X = x x.T`:

```python
X = cp.Variable((p, p), PSD=True)
constraints = [cp.trace(X) == 1, cp.sum(cp.abs(X)) <= sparsity_budget]
objective = cp.Maximize(cp.trace(Sigma @ X))
```

Declare this as an SDP relaxation. Recover a loading vector from the leading
eigenvector and threshold/project to the requested support size.

### Max-Cut and QUBO

For signs `s_i in {-1, 1}`, lift `X = s s.T`:

```python
X = cp.Variable((n, n), PSD=True)
constraints = [cp.diag(X) == 1]
objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - X)))
```

Declare the SDP value as an upper bound for maximization. Use random
hyperplane rounding to produce a feasible cut.

## Certifiable Relaxations and Recovery

### AC Optimal Power Flow

Use a voltage product lift `W = v v.H` and drop rank one. In CVXPY, this is
usually an SDP for small systems or an SOC/QC relaxation for larger systems.

Report:

- Whether the relaxation is SDP, SOC, or QC.
- Objective lower bound.
- Rank/tightness of `W`.
- AC power-flow residual after recovering voltages.

For realistic large cases, prefer specialized OPF tooling for data parsing and
use CVXPY only for small educational relaxations or extracted cone submodels.

### Pose Graph Optimization

Use an SDP relaxation for rotation synchronization or the SE-Sync style
certifiable relaxation. Translation variables are often eliminated or solved
conditionally on rotations.

Report:

- Rank of the relaxed matrix.
- Certificate or duality gap if available.
- Trajectory residual after projecting rotations back to `SO(d)`.

### Essential Matrix Estimation

Use a QCQP-to-SDP relaxation over lifted polynomial variables, or an
orthogonal-space convex relaxation. Do not present a generic least-squares
essential matrix fit as a convexification of the manifold constraints.

Report:

- Epipolar/Sampson residual.
- Rank/tightness of the SDP solution.
- Rotation and translation-direction error when ground truth exists.
- RANSAC or robust estimation treatment if outliers are present.

### Cardinality-Constrained Portfolio

When the continuous model is convex and only selection is discrete, use a
mixed-integer convex model:

```python
x = cp.Variable(n)
y = cp.Variable(n, boolean=True)
constraints = [
    cp.sum(x) == 1,
    x >= 0,
    x <= upper_bounds * y,
    x >= lower_bounds * y,
    cp.sum(y) <= cardinality,
    cp.quad_form(x, Sigma) <= risk_limit,
]
objective = cp.Maximize(mu @ x)
```

For fixed transaction costs or on/off convex costs, use perspective
reformulations when available. If no MIP solver is available, state that an
`l1` surrogate is a heuristic relaxation, not the same problem.

## Lower-Bounding Convexifications

### Lennard-Jones Clusters

The Lennard-Jones objective is highly multimodal. Do not promise an exact
CVXPY convex reformulation.

Useful convex work:

- Build valid convex underestimators on bounded boxes, such as alphaBB-style
  lower bounds.
- Use them inside branch-and-bound.
- Seed local search from feasible points separately.

Report valid lower bounds, best feasible energy, and the gap. Treat CVXPY as a
subproblem solver for convex lower-bounding pieces, not a global solver for the
raw cluster problem.

## CVXPY Implementation Cautions

- Use `cp.Variable((n, n), PSD=True)` for real symmetric SDP lifts.
- For complex Hermitian lifts, use `hermitian=True` plus explicit PSD
  constraints if the target CVXPY version requires it.
- `cp.norm(X, 1)` is a matrix norm, not entrywise l1. Use
  `cp.sum(cp.abs(X))` for sparse PCA style entrywise penalties.
- Mixed-integer DCP can pass `problem.is_dcp()` but still needs an installed
  mixed-integer solver.
- Large lifted SDPs can be computationally inappropriate even when they are
  mathematically correct. Offer SOC/QC/first-order alternatives when scale is
  central to the request.
