# Hidden Convexity, S-Lemma, and Lossless SDR

Use this reference when a problem looks like a *generic* nonconvex QCQP, bilinear program, or ratio-of-quadratics, but in fact admits an **exact** convex reformulation. Many practitioners reach for a local solver here when an SDP or SOC reformulation is provably tight. Knowing *when* the relaxation is exact is the actual skill — not the lift itself.

## Decision Tree

1. Is the nonconvexity a single quadratic constraint? → **Trust Region / GTRS family.** Exact SDP under Slater + dual definiteness (S-lemma).
2. Is it a ratio of quadratics with a real-valued numerator (e.g. SINR)? → **Phase-rotation SOC** if you can rotate so numerator is real. Exact for unicast downlink beamforming (Bengtsson-Ottersten, Wiesel-Eldar-Shamai).
3. Is it `max x^T A x` over an ellipsoid or the sphere? → **Eigenvalue reformulation**, no SDP needed.
4. Is it a bilinear product `xy` with one variable bounded? → **McCormick envelope** (relaxation, not exact unless `x` or `y` are binary).
5. Generic indefinite QCQP with multiple constraints? → **Shor SDP** is a relaxation. Tightness must be argued problem-by-problem (rank-one certificate, randomized rounding for feasibility).

## The S-Lemma (Yakubovich)

For symmetric `A`, `B` with Slater (some `x_0` with `x_0^T B x_0 < 0`):

> `x^T A x >= 0` for all `x` with `x^T B x <= 0`
> ⟺ ∃ `mu >= 0` with `A + mu B ⪰ 0`.

Consequence: any QCQP with **one** quadratic constraint has an *exact* SDP relaxation. This is the hidden-convexity result behind the trust-region subproblem and its generalizations.

## Generalized Trust-Region (GTRS) — Exact SDP

Problem (`A`, `C` symmetric, possibly indefinite):

```
minimize    x' A x + b' x
subject to  x' C x + d' x <= e
```

Lift `X = x x'`, homogenize with the block:

```python
import cvxpy as cp
import numpy as np

n = A.shape[0]
X = cp.Variable((n, n), symmetric=True)
x = cp.Variable(n)
M = cp.bmat([[X, x[:, None]], [x[None, :], np.array([[1.0]])]])
constraints = [
    M >> 0,
    cp.trace(C @ X) + d @ x <= e,
]
obj = cp.Minimize(cp.trace(A @ X) + b @ x)
prob = cp.Problem(obj, constraints)
prob.solve(solver=cp.CLARABEL)
```

Tightness conditions (Polik-Terlaky; Beck-Eldar):

- Primal Slater: ∃ `x_0` with `x_0' C x_0 + d' x_0 < e`.
- Dual definiteness: ∃ `mu >= 0` with `A + mu C ⪰ 0`.

When both hold, the SDP value is exact and a rank-one optimum exists. A generic
solver can still return a higher-rank point on an optimal face, so treat rank as
a recovery diagnostic, not as the exactness theorem itself. If the returned
matrix is numerically rank one (top-eigenvalue / trace near 1), recover
`x^* = M^*[:n, n]` (the off-block).

Always include a numerical rank check and report the eigenvalue ratio. If rank
is higher than one despite the exactness conditions, say that the returned
matrix did not directly certify primal recovery; use a GTRS/KKT recovery method
or report leading-eigenvector projection as a heuristic separately from the SDP
bound.

## Single-Quadratic-Constrained Quadratic Programs (CDT, ETRS)

The same exact-SDR result extends to:

- Trust region with extra linear cuts: still exact under suitable Slater (Beck-Vaisbourd 2016).
- Two-quadratic-constraint problems (CDT): exact under interior + dimension conditions (Burer-Anstreicher; Sakaue-Nakatsukasa 2017).
- Quadratic equality `x' x = 1`: lift `X = x x'`, `trace(X) = 1`, `X ⪰ 0`. Tight when the bilinear term is "trivial" (Polyakovskiy-Sahinidis).

## Downlink Beamforming — Lossless SOC (Bengtsson-Ottersten)

The SINR-constrained power-minimization problem with single-stream unicast looks like a ratio-of-quadratics nonconvex constraint, but is **exactly convex** via a phase-rotation trick.

Problem:

```
min  sum_k ||w_k||^2
s.t. |h_k^H w_k|^2 / (sum_{j != k} |h_k^H w_j|^2 + sigma^2) >= gamma_k    for all k
```

For each `k`, pick the global phase of `w_k` so that `h_k^H w_k` is real and nonnegative. Then the SINR constraint becomes:

```
Re(h_k^H w_k) >= sqrt(gamma_k) * || stack(h_k^H w_j for j != k, sigma) ||_2
```

This is a second-order cone constraint. The whole problem is a single SOCP. **Exact**, not a relaxation, for any number of users. Implementation:

```python
W = cp.Variable((Nt, K), complex=True)
constraints = []
for k in range(K):
    interferers = [W[:, j] for j in range(K) if j != k]
    interference = cp.hstack([h[:, k].conj() @ wj for wj in interferers] + [sigma])
    constraints += [
        cp.imag(h[:, k].conj() @ W[:, k]) == 0,
        cp.real(h[:, k].conj() @ W[:, k]) >= np.sqrt(gamma) * cp.norm(interference, 2),
    ]
obj = cp.Minimize(cp.sum(cp.norm(W, 2, axis=0) ** 2))
```

Alternative SDR route (lift `W_k = w_k w_k^H`, drop rank-1) yields the same optimum on this problem; the rank-1 condition is satisfied at the optimum, so SDR is also exact here. Prefer the SOC form because it scales better and avoids randomization.

## Sensor Network Localization (Biswas-Ye)

Distance equalities `||x_i - x_j||^2 = d_ij^2` are nonconvex. Relax with `Z = [I, X; X^T, Y] ⪰ 0`, `Y_ii + Y_jj - 2 Y_ij = d_ij^2`. Exact under universal rigidity with anchors. For typical noisy data, declare it a **relaxation** with feasibility residual.

## Lossless OPF (Farivar-Low)

The branch-flow SOC relaxation of AC-OPF on a **radial (tree) network** is exact under all of:

- Tree topology.
- Objective is monotone-increasing in branch current.
- No upper bounds on injections that would create reverse flow.

For meshed networks or with binding upper bounds, declare it as a relaxation and report tightness via `W = v v^H` rank.

## Reporting Discipline

When you use a lifted SDR, always:

- Print `lambda_2 / lambda_1` of the lifted matrix as the rank-one indicator.
- State Slater + dual definiteness checks if you claim exactness via S-lemma.
- If rank > 1, do not let the rank diagnostic override a valid exactness theorem. Recover with a theorem-specific method when available; otherwise report leading-eigenvector projection as a heuristic and keep the SDP bound separate.
