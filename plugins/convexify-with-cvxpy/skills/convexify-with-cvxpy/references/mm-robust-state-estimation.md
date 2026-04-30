# MM Robust State Estimation Patterns

Use this reference when the user's model contains robust state estimation,
Student-t measurement noise, Cauchy-like penalties, or objectives of the form
`log(1 + squared residual / scale)`. These models are usually not exactly
convex, but they often have useful majorization-minimization (MM), IRLS, or
sequential convexification subproblems.

Motivating source: arXiv:2411.11320 formulates constrained state estimation
with Student-t measurement noise and solves repeated convex quadratic or
quadratically constrained subproblems by majorizing the nonconvex objective and
constraints.

## Original Model

At one time step, the constrained MAP estimate has variable `x` and fixed data:

- predicted state `x_pred`
- inverse predicted covariance `P_inv`, positive semidefinite
- measurement matrix `C`
- measurement `y`
- Student-t scale `sigma_i > 0`
- degrees of freedom `nu_i > 0`

The objective is

```text
quad_form(x - x_pred, P_inv)
+ sum_i (1 + nu_i) * log(1 + (C_i x - y_i)^2 / (sigma_i^2 * nu_i)).
```

The Student-t term is not DCP. Do not encode it directly as
`cp.log(1 + cp.square(...))` and try to force CVXPY to accept it.

## Classification

This is generally a nonconvex optimization problem:

- The prior term is convex quadratic when `P_inv` is PSD.
- The Student-t loss is convex near zero residual and concave for large
  residuals.
- General constraints `g(x) <= 0` may also be nonconvex.

The standard CVXPY route is therefore sequential/local:

- Grammar per subproblem: DCP.
- Overall method: MM, IRLS, CCP, or SCP depending on the constraint treatment.
- Guarantee: local stationarity or monotone decrease under the surrogate
  assumptions, not a global convex reformulation.

## Log-Tangent Student-t Majorizer

For residual

```text
r_i(x) = C_i x - y_i
a_i = sigma_i^2 * nu_i
alpha_i = 1 + nu_i
```

write the robust term as

```text
alpha_i * log(1 + z_i / a_i), where z_i = r_i(x)^2.
```

The function `log(1 + z / a_i)` is concave in `z >= 0`, so its tangent at the
current iterate `x_ref` is a global upper bound. With

```text
z_ref_i = r_i(x_ref)^2
w_i = alpha_i / (a_i + z_ref_i),
```

the nonconstant part of the majorizer is

```text
w_i * r_i(x)^2.
```

The convex subproblem is

```text
minimize_x
    quad_form(x - x_pred, P_inv)
    + sum_i w_i * (C_i x - y_i)^2
subject to
    convex constraints or convex surrogate constraints.
```

This is the weighted least-squares / IRLS form. In the paper's notation this
weight is `m_i^t`.

## CVXPY Skeleton

Keep the nonlinear weight update outside CVXPY. Use a nonnegative Parameter
for repeated solves.

```python
import cvxpy as cp
import numpy as np

nx, ny = C.shape[1], C.shape[0]
x = cp.Variable(nx)
w = cp.Parameter(ny, nonneg=True)

resid = C @ x - y
prior = cp.quad_form(x - x_pred, P_inv)
robust_majorizer = cp.sum(cp.multiply(w, cp.square(resid)))

constraints = [
    # Add exact convex constraints or surrogate constraints here.
]

problem = cp.Problem(cp.Minimize(prior + robust_majorizer), constraints)
assert problem.is_dcp()

x_ref = x_pred.copy()
for _ in range(max_iters):
    resid_ref = C @ x_ref - y
    w.value = (1.0 + nu) / (sigma**2 * nu + resid_ref**2)

    problem.solve(solver=cp.OSQP)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(problem.status)

    x_next = x.value
    if np.linalg.norm(x_next - x_ref) <= tol * max(1.0, np.linalg.norm(x_ref)):
        break
    x_ref = x_next
```

Use `OSQP` for QP subproblems with linear constraints. Use `CLARABEL` when
the subproblem has second-order cone, PSD, or convex quadratic constraints.

## L-Smooth Alternative

If the residual-level log tangent is awkward, use the descent lemma for the
entire nonconvex measurement term `F_ncvx`.

If `F_ncvx` has Lipschitz gradient constant `L`, then

```text
F_ncvx(x) <= F_ncvx(x_ref)
             + grad_F_ncvx(x_ref)^T (x - x_ref)
             + (L / 2) * ||x - x_ref||_2^2.
```

For Student-t residuals, a conservative bound used by the paper is

```text
L = 2 * sum_i ((nu_i + 1) / (nu_i * sigma_i^2)) * ||C_i||_2^2.
```

This produces a strongly convex quadratic objective. It can be less tight than
the log-tangent IRLS surrogate, but it is easy to combine with generic SCP
machinery.

## Constraint Surrogates

Keep constraints exact when they are already convex:

```python
constraints += [cp.sum(x) == 1, x >= 0]
constraints += [cp.sum_squares(position) <= radius_upper**2]
```

For differentiable nonconvex `g(x) <= 0` with Lipschitz gradient constant `G`,
an inner convex surrogate is

```text
g_tilde(x; x_ref) =
    g(x_ref)
    + grad_g(x_ref)^T (x - x_ref)
    + (G / 2) * ||x - x_ref||_2^2 <= 0.
```

Because `g_tilde` upper-bounds `g`, the surrogate constraint implies the
original constraint. If the current iterate is infeasible, add slack with a
large penalty or use a feasibility restoration step.

For a ring lower bound

```text
||p||_2^2 >= radius_lower^2,
```

linearize the convex quadratic lower bound at `p_ref`:

```text
||p_ref||_2^2 + 2 p_ref^T (p - p_ref) >= radius_lower^2.
```

For an indefinite quadratic constraint

```text
x^T D x <= 0,
```

split `D = D_plus - D_minus` with both parts PSD, then linearize the convex
term `x^T D_minus x` from below:

```text
quad_form(x, D_plus)
- (x_ref^T D_minus x_ref + 2 (D_minus x_ref)^T (x - x_ref))
<= 0.
```

## Reporting Checklist

When delivering this type of model, state:

- The original problem is nonconvex.
- The delivered method is sequential/local, not an exact convex reformulation.
- The per-iteration CVXPY problem is DCP.
- Which surrogate was used: log-tangent IRLS, L-smooth quadratic, or a custom
  constraint surrogate.
- Solver choice and why.
- Convergence criteria, final objective or surrogate value, and any slack or
  constraint residuals.

