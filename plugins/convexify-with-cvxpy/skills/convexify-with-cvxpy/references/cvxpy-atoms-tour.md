# CVXPY Atoms Tour and Footguns

A guided tour of less-loved atoms that frequently solve a "this is not DCP" problem in one line, plus the most common footguns. Reach here when an expression seems to need a custom epigraph but a built-in atom already does the work.

## Less-Loved Convex / Concave Atoms

| Atom | Purpose | Cone | Notes |
|---|---|---|---|
| `cp.log_det(X)` | D-optimality, MaxEnt, graphical lasso, Laplacian learning | EXP | Concave on PSD; not the same as `cp.log(cp.det(X))` (which is not DCP). |
| `cp.matrix_frac(x, P)` | Mahalanobis distance `x.T @ inv(P) @ x` | SDP | `P` must be a PSD variable or constant. Convex jointly. |
| `cp.tr_inv(X)` | A-optimality `tr(inv(X))` | SDP | Convex on PSD. |
| `cp.lambda_max(X)` / `cp.lambda_min(X)` | Spectral bounds | SDP | E-optimality is `max lambda_min`. Convex / concave. |
| `cp.quad_over_lin(x, y)` | `||x||^2 / y` for `y > 0` | SOC | Convex jointly. The "rotated SOC" workhorse. |
| `cp.geo_mean(x, p)` | Weighted geometric mean | POW | Concave on `x >= 0`; weights must be nonnegative rationals (or floats CVXPY can rationalize). |
| `cp.harmonic_mean(x)` | `n / sum(1/x_i)` | POW | Concave on `x > 0`. Useful for rate-fairness and A-optimality proxies. |
| `cp.kl_div(p, q)` | `p log(p/q) - p + q` | EXP | Convex jointly. KL-DRO, MaxEnt fitting. |
| `cp.rel_entr(p, q)` | `p log(p/q)` | EXP | Same as `kl_div` minus the `q - p` term. Pick whichever your problem already has. |
| `cp.logistic(x)` | `log(1 + exp(x))` | EXP | Soft-plus; logistic regression. |
| `cp.entr(x)` | `-x log(x)` | EXP | Concave entropy. |
| `cp.huber(x, M)` | Huber loss | SOC + LP | Robust regression. |
| `cp.tv(image)` | Total variation | SOC | Vectors use 1D l1 TV; matrices use isotropic 2D l2 TV. Pass extra same-shaped matrices as additional arguments for multichannel TV. |
| `cp.perspective(f, s)` | `s * f(x / s)` with `s = 0` handled | (depends) | The right atom for on/off costs and indicator-gated convex losses. CVXPY 1.3+. |
| `cp.PowCone3D(x, y, z, alpha)` / `PowConeND` | Raw power cone | POW | When no atom matches, build from the cone directly. |

## Footguns

### `cp.norm(X, 1)` is not entrywise l1

For a *matrix* `X`, `cp.norm(X, 1)` is the **operator (max-column-sum) norm**, not the entrywise `l1`. For graphical lasso, sparse PCA, Laplacian learning, and any other problem where you mean `sum |X_ij|`, write **`cp.sum(cp.abs(X))`**. This is the single most common silently-wrong CVXPY pattern.

```python
# WRONG for graphical lasso
penalty = alpha * cp.norm(S, 1)            # operator norm, not what you want

# RIGHT
penalty = alpha * cp.sum(cp.abs(S))         # entrywise l1
```

### `cp.quad_form(x, P)` requires constant PSD `P`

If `P` is itself a variable, `quad_form` is not DCP. Distinguish these two cases:

```python
# Want x^T P^{-1} x with P variable PSD: use cp.matrix_frac
expr = cp.matrix_frac(x, P)

# Want x^T P x with both x and P variable: generally nonconvex.
# There is no generic exact DCP rewrite.
```

The Schur complement block `[[P, x], [x.T, t]] >> 0` models
`t >= x.T @ inv(P) @ x`, not `t >= x.T @ P @ x`. For `x.T @ P @ x` with
variable `P`, use a problem-specific relaxation, bound, or lift and state what
was relaxed. If `x` is fixed data and only `P` is variable, the expression is
linear: `trace(P @ np.outer(x, x))`.

### `cp.log(cp.det(X))` is not DCP

`cp.det` is not a CVXPY atom. The composition `log . det` is concave on PSD but CVXPY cannot certify it through that path. Use the dedicated atom **`cp.log_det(X)`**.

### Sign of `cp.abs` collapses when sign is forced

If you've constrained `x <= 0`, then `cp.abs(x) == -x`. CVXPY can sometimes simplify; sometimes it cannot. If `cp.abs(x)` triggers a DCP rejection but you know the sign, write `-x` (or `+x`) directly to bypass the (now pointless) absolute value. Example: in graph Laplacian learning with `L_ij <= 0` for off-diagonals, the entrywise penalty `sum(|L_off|)` becomes `-sum(L_off)`, a linear objective, no `cp.abs` epigraph needed.

### Mixed-integer DCP needs an MIP solver

`problem.is_dcp()` returning True for a model with `boolean=True` or `integer=True` variables only means the **continuous relaxation** is DCP. You still need a MIP solver:

- LP / MILP: GLPK_MI, CBC, SCIP, HIGHS, MOSEK, GUROBI.
- MIQP / MISOCP: SCIP (limited), MOSEK, GUROBI, COPT.
- MICP with **EXP cone** (e.g. mixed-integer logistic): currently only MOSEK, COPT among the routinely-installed.

If MOSEK is unavailable, an exhaustive enumeration over `C(p, k)` supports for sparse logistic with small `p` is mathematically equivalent to MICP and often faster than randomly-selected heuristics. Mention this fallback explicitly when you take it.

### DPP and parameters

Disciplined Parametric Programming (DPP) is required for fast re-solves with `cp.Parameter`. Operations that **break DPP**: `cp.Parameter` inside a nonlinear atom, `Parameter * Variable * Parameter`, division by a `Parameter`. The fix is usually to introduce an auxiliary variable to factor the parameter out. Verify with `problem.is_dcp(dpp=True)` or `problem.is_dgp(dpp=True)`.

### DGP requires `pos=True`, not `>= 0`

Geometric programming sees positivity through the `pos=True` attribute on `cp.Variable`, not from a `>= 0` constraint. Forgetting this is a common reason `is_dgp()` returns False.

### `cp.Variable((n, n), PSD=True)` is real symmetric

For complex Hermitian PSD lifts (e.g. complex beamforming covariance), use `cp.Variable((n, n), hermitian=True)` plus `>> 0`, or use `complex=True` with explicit `>> 0` and Hermitian symmetry. Some CVXPY versions handle this automatically — check `is_dcp()` and the canonicalization shape.

### `cp.norm` axis convention

`cp.norm(X, 2, axis=0)` returns column norms, `axis=1` returns row norms, default returns the spectral / max-singular-value norm of the matrix as a whole. Mixing these up makes "total power" objectives wrong by a permutation.
