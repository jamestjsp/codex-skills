# Reformulation Patterns for CVXPY

## Table of Contents

- Atom substitutions
- Epigraphs and hypographs
- Cone lifts
- Products and ratios
- DGP patterns
- DQCP and bisection
- Robust optimization
- Semidefinite and rank relaxations
- Sequential convexification

## Atom Substitutions

Use atoms CVXPY knows instead of equivalent algebra:

```python
cp.norm(A @ x + b, 2)              # instead of sqrt(sum_squares(...))
cp.sum_squares(A @ x - b)          # least-squares objective
cp.quad_form(x, P)                 # require constant PSD P
cp.matrix_frac(x, P)               # x.T @ inv(P) @ x
cp.quad_over_lin(x, y)             # sum_squares(x) / y, y > 0
cp.log_sum_exp(X)                  # smooth max / log partition
cp.rel_entr(x, y)                  # x * log(x / y), EXP cone
cp.geo_mean(x)                     # concave geometric mean, x >= 0
```

If a monotonicity rule depends on sign, add attributes:

```python
x = cp.Variable(nonneg=True)
y = cp.Variable(nonpos=True)
```

## Epigraphs and Hypographs

Replace convex objectives or subexpressions with auxiliary variables when it clarifies constraints or enables composition.

Absolute value:

```python
t = cp.Variable(n, nonneg=True)
constraints += [x <= t, -x <= t]
objective = cp.Minimize(cp.sum(t))
```

Maximum of convex expressions:

```python
t = cp.Variable()
constraints += [f_i <= t for f_i in convex_terms]
objective = cp.Minimize(t)
```

Minimum of concave expressions for maximization:

```python
t = cp.Variable()
constraints += [g_i >= t for g_i in concave_terms]
objective = cp.Maximize(t)
```

One-sided replacement rule:

- `t >= convex_expr` is convex and useful in minimization or upper-bound constraints.
- `t <= concave_expr` is convex and useful in maximization or lower-bound constraints.
- Do not replace `t == convex_expr` unless another proof shows tightness.

## Cone Lifts

Second-order cone:

```python
constraints += [cp.norm(A @ x + b, 2) <= c @ x + d]
constraints += [cp.SOC(c @ x + d, A @ x + b)]
```

Rotated quadratic patterns:

```python
constraints += [cp.quad_over_lin(x, y) <= t, y >= 0]
```

Semidefinite cone:

```python
X = cp.Variable((n, n), PSD=True)
constraints += [A @ X + X @ A.T << -Q]
```

Exponential cone through atoms:

```python
objective = cp.Minimize(cp.log_sum_exp(A @ x + b))
constraints += [cp.rel_entr(p, q) <= tau]
```

Power cone when no atom exists:

```python
constraints += [cp.PowCone3D(x, y, z, alpha)]
```

## Products and Ratios

Variable times variable is usually non-DCP:

```python
x * y  # non-DCP when x and y are both variables
```

Use one of these routes:

- DGP for positive monomials/posynomials.
- DQCP for ratios or sign-qualified products with quasiconvex/quasiconcave structure.
- McCormick envelope for bounded bilinear relaxation.
- Perspective or on/off convex cost when a binary/scaling variable gates a convex term.
- SCP: linearize one factor around the current iterate for a local method.

McCormick relaxation for `w = x*y`, with `x in [lx, ux]`, `y in [ly, uy]`:

```python
w = cp.Variable()
constraints += [
    w >= lx * y + ly * x - lx * ly,
    w >= ux * y + uy * x - ux * uy,
    w <= ux * y + ly * x - ux * ly,
    w <= lx * y + uy * x - lx * uy,
]
```

Linear-fractional ratio:

- If numerator and denominator are affine and denominator is positive, try DQCP.
- If optimizing a ratio, use Charnes-Cooper only when the structure is truly linear-fractional and constraints can be homogenized.
- For max-min SINR/fairness, introduce a scalar level parameter and solve feasibility by bisection or use DQCP if CVXPY certifies it.

## DGP Patterns

Use DGP when variables are positive and constraints can be written as posynomial <= monomial or monomial == monomial.

```python
x = cp.Variable(pos=True)
y = cp.Variable(pos=True)
objective = cp.Minimize(x + y)
constraints = [x**0.5 * y**-1 <= 2, x * y >= 1]
problem = cp.Problem(objective, constraints)
assert problem.is_dgp()
problem.solve(gp=True)
```

Checklist:

- Use `pos=True` for variables and positive non-exponent parameters.
- Keep constants positive.
- Avoid subtracting positive monomial terms; posynomials are sums of positive monomials.
- Use log-log curvature diagnostics: `expr.log_log_curvature`, `expr.is_dgp()`.

## DQCP and Bisection

Use DQCP for quasiconvex objectives or constraints that are convex after fixing a scalar level.

```python
problem = cp.Problem(cp.Minimize(quasiconvex_expr), constraints)
assert problem.is_dqcp()
problem.solve(qcp=True, low=lower_bound, high=upper_bound)
```

Manual bisection template:

```python
alpha = cp.Parameter(nonneg=True)
constraints = base_constraints + [level_set_expr <= alpha]
feas = cp.Problem(cp.Minimize(0), constraints)
for _ in range(max_iters):
    alpha.value = 0.5 * (lo + hi)
    feas.solve(solver=cp.CLARABEL)
    if feas.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        hi = alpha.value
    else:
        lo = alpha.value
```

Use manual bisection when the level-set model is clearer than a direct DQCP expression or when you need custom feasibility logic.

## Robust Optimization

Convert uncertainty sets to worst-case support functions.

Robust linear inequality for `a = a0 + u`, `||u||_2 <= rho`:

```python
constraints += [a0 @ x + rho * cp.norm(x, 2) <= b]
```

Box uncertainty `||u||_inf <= rho`:

```python
constraints += [a0 @ x + rho * cp.norm(x, 1) <= b]
```

Ellipsoidal uncertainty `u.T @ inv(Sigma) @ u <= rho^2`:

```python
constraints += [a0 @ x + rho * cp.norm(Sigma_sqrt @ x, 2) <= b]
```

Replace squared loss with robust penalties for outliers:

```python
objective = cp.Minimize(cp.sum(cp.huber(A @ x - y, M=1.0)))
```

## Semidefinite and Rank Relaxations

Lifting `X = x x^T` is nonconvex because it requires rank one. The convex relaxation is:

```python
X = cp.Variable((n, n), PSD=True)
constraints += [
    cp.bmat([[X, cp.reshape(x, (n, 1), order="F")],
             [cp.reshape(x, (1, n), order="F"), np.ones((1, 1))]]) >> 0
]
```

Then drop `rank(X) == 1`. State that this is an SDP relaxation. Check tightness by eigenvalues of `X.value`.

Use nuclear norm for convex low-rank promotion:

```python
objective = cp.Minimize(data_fit + lam * cp.norm(X, "nuc"))
```

## Sequential Convexification

Use for nonlinear dynamics, collision avoidance, nonconvex keep-out zones, or bilinear control terms when exact convexity is unavailable.

Loop template:

1. Initialize a feasible or dynamically plausible trajectory.
2. Linearize nonconvex terms around the current iterate.
3. Add trust region constraints such as `cp.norm(x - x_ref, "inf") <= delta`.
4. Add virtual-control/slack variables with large penalties to avoid infeasible subproblems.
5. Solve the convex subproblem.
6. Accept/reject based on actual nonlinear merit improvement.
7. Shrink or grow trust regions and repeat.

For a differentiable nonconvex equality `h(x) == 0`, a local affine model is:

```python
h_ref + grad_h_ref @ (x - x_ref) == 0
```

For a difference-of-convex objective `f(x) - g(x)`, where `f` and `g` are convex, replace `g` by its affine first-order lower bound at `x_ref`; the resulting objective upper-bounds the original locally.
