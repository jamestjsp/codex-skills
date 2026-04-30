# Rocket Landing and Control Convexification

## Table of Contents

- Use this as an analogy, not a flight implementation
- Powered-descent structure
- Lossless convexification pattern
- Successive convexification pattern
- CVXPY sketch
- Source anchors

## Use This as an Analogy, Not a Flight Implementation

Rocket landing is a useful example because the physical problem contains nonconvex thrust lower bounds, pointing constraints, fuel use, nonlinear mass dynamics, and real-time replanning pressure. A production vehicle needs validated dynamics, actuator models, navigation uncertainty, timing guarantees, and flight software constraints. Use this reference to guide modeling patterns, not to claim flight readiness.

## Powered-Descent Structure

Typical discretized variables:

- `r[:, k]`: position.
- `v[:, k]`: velocity.
- `u[:, k]`: scaled thrust or acceleration command.
- `sigma[k]`: lifted thrust magnitude upper-bound variable.
- `z[k]`: transformed mass, often log mass or another variable introduced by the paper-specific convexification.

Convex pieces:

- Affine initial and terminal boundary constraints.
- Affine discretized linear dynamics, or linearized nonlinear dynamics inside SCP.
- Fuel surrogate `sum(sigma) * dt`.
- Speed, glide-slope, and pointing cones when written with the correct sign convention.
- Upper thrust magnitude `norm(u[:, k]) <= sigma[k]`.

Nonconvex pieces requiring care:

- Lower thrust magnitude `norm(T[:, k]) >= Tmin`.
- Coupling between force, mass, and acceleration.
- Aerodynamics and attitude dynamics.
- Obstacle/keep-out constraints.
- Free final time if it multiplies dynamics; often handled by bisection or SCP.

## Lossless Convexification Pattern

The central trick is to replace a nonconvex annular thrust constraint with a convex lifted set and prove the relaxation is tight at optimum.

Original thrust magnitude pattern:

```text
Tmin <= ||T_k||_2 <= Tmax
```

Convex lifted relaxation:

```python
sigma = cp.Variable(N, nonneg=True)
u = cp.Variable((3, N))
constraints += [
    cp.norm(u[:, k], 2) <= sigma[k],
    sigma[k] >= sigma_min[k],
    sigma[k] <= sigma_max[k],
]
objective = cp.Minimize(dt * cp.sum(sigma))
```

This is not automatically equivalent. It becomes "lossless" only under assumptions that force `sigma[k] == norm(u[:, k])` at optimality or otherwise prove the relaxed solution maps back to the original feasible set. If those assumptions are not established, call it a convex relaxation.

Pointing cone example:

```python
constraints += [u[2, k] >= np.cos(theta_max) * sigma[k]]
```

Glide slope cone example, assuming `r[2, k]` is altitude:

```python
constraints += [cp.norm(r[:2, k], 2) <= np.tan(gamma) * r[2, k]]
constraints += [r[2, k] >= 0]
```

## Successive Convexification Pattern

Use SCP when dynamics or constraints are not exactly convex after lifting.

Subproblem ingredients:

- Linearized dynamics about `(r_ref, v_ref, u_ref, z_ref)`.
- Trust regions around the reference trajectory.
- Virtual control/slack with large penalties to keep subproblems feasible.
- Acceptance ratio comparing predicted improvement to actual nonlinear improvement.
- Outer-loop convergence based on state/control changes and slack norm.

SCP is a local method. It can work well for trajectory generation, but it is not the same as a globally certified convex formulation.

## CVXPY Sketch

This is a compact SOCP-style skeleton for the convex subproblem. Fill in verified dynamics and scaling for the actual vehicle.

```python
import cvxpy as cp
import numpy as np

N = 40
dt = 0.2
g = np.array([0.0, 0.0, -9.81])

r = cp.Variable((3, N + 1))
v = cp.Variable((3, N + 1))
u = cp.Variable((3, N))
sigma = cp.Variable(N, nonneg=True)

constraints = [
    r[:, 0] == r0,
    v[:, 0] == v0,
    r[:, N] == r_target,
    v[:, N] == 0,
]

for k in range(N):
    constraints += [
        r[:, k + 1] == r[:, k] + dt * v[:, k],
        v[:, k + 1] == v[:, k] + dt * (u[:, k] + g),
        cp.norm(u[:, k], 2) <= sigma[k],
        sigma[k] >= sigma_min,
        sigma[k] <= sigma_max,
        u[2, k] >= np.cos(theta_max) * sigma[k],
        cp.norm(r[:2, k], 2) <= np.tan(glide_slope) * r[2, k],
        r[2, k] >= 0,
    ]

problem = cp.Problem(cp.Minimize(dt * cp.sum(sigma)), constraints)
assert problem.is_dcp()
problem.solve(solver=cp.CLARABEL)
```

If final time is free, first try an outer bisection/grid over `dt` or `N`. If time appears inside nonlinear dynamics, use SCP with time as a linearized variable or solve a family of fixed-time convex subproblems.

## Source Anchors

- Lars Blackmore, "Autonomous Precision Landing of Space Rockets", National Academy of Engineering, 2017: https://www.nationalacademies.org/read/23659/chapter/10
- NASA NTRS Tech Memo, "Tutorial: MATLAB Implementation of a Successive Convexification Algorithm for 3 DoF Rocket Landings": https://ntrs.nasa.gov/citations/20230009811
- Acikmese and Ploen, "Convex programming approach to powered descent guidance for Mars landing", cited by the National Academies chapter above.
- Blackmore, Acikmese, and Scharf, "Minimum landing error powered descent guidance for Mars landing using convex optimization", cited by the National Academies chapter above.
