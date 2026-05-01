# Distributionally Robust Optimization and Chance Constraints

Use this reference when the user mentions distribution shift, Wasserstein DRO, ambiguity sets, chance constraints, robustness to data perturbations, or moment-based DRO. The most common mistake is to set up an explicit min-max iteration when a clean closed-form convex reformulation already exists.

## Wasserstein-1 DRO with Lipschitz Loss — Regularization Equivalence

For loss `ell(x; xi)` that is `L`-Lipschitz in `xi` (logistic, hinge, Huber, l1 regression are 1-Lipschitz; squared loss is **not**):

```
sup_{Q in W_1(P_hat, eps)} E_Q[ell(x; xi)]   =   E_{P_hat}[ell(x; xi)] + eps * L * ||x||_*
```

where `||·||_*` is the dual norm of the Wasserstein ground metric.

**Implication:** Wasserstein-1 DRO with a Lipschitz loss is just regularized empirical risk minimization. No min-max, no iterative solve. References: Mohajerin Esfahani-Kuhn 2018; Shafieezadeh-Abadeh-Esfahani-Kuhn 2019; Blanchet-Murthy 2019.

### DRO Logistic Regression

```python
import cvxpy as cp
beta = cp.Variable(p)
b0   = cp.Variable()
margin = cp.multiply(y, X @ beta + b0)         # y in {-1, +1}
loss   = cp.sum(cp.logistic(-margin)) / n
penalty = epsilon * cp.norm(beta, q)            # q = dual norm of ground metric
prob = cp.Problem(cp.Minimize(loss + penalty))
```

Dual-norm pairing for the ground metric:

| Wasserstein ground metric on `xi` | Dual norm `q` on `beta` |
|---|---|
| `l_2`                             | `2`                      |
| `l_1`                             | `inf`                    |
| `l_inf`                           | `1`                      |

Choose the ground metric to match how features can shift. Symmetric label flips usually need an extra `kappa` term — see Shafieezadeh-Abadeh §5; for typical practice, set `kappa = inf` (label-respecting transport) and use the regularization form above.

### DRO Linear Regression with l1 / Huber Loss

Same equivalence. For squared loss it does **not** apply directly; use the Wasserstein-2 ball with the Gao-Kleywegt or Blanchet-Kang-Murthy reformulation, or fall back to a tractable affine-decision-rule approximation.

## Bernstein / Nemirovski-Shapiro Safe Approximation of Chance Constraints

For `Pr_xi[ a(xi)^T x <= b(xi) ] >= 1 - eps` with sub-Gaussian or known-MGF `xi`:

```
inf_{t > 0}  (1/t) * ( log E_xi[ exp( t * (a(xi)^T x - b(xi)) ) ] - log(eps) )  <=  0
```

For Gaussian `xi ~ N(mu, Sigma)` and affine `a(xi)^T x - b(xi) = c^T x + xi^T Q x + ...`, the log-MGF is closed-form quadratic and the safe approximation collapses to a single SOC or SDP constraint. Implementation tip: introduce `t > 0` as an auxiliary variable and use `cp.log_sum_exp` plus `cp.quad_over_lin` to build the EXP-cone form, then minimize jointly over `(x, t)`.

This is far less conservative than scenario-based VaR for moderate `eps`, and it is convex in `x` (after the `t`-minimization, the resulting envelope is convex).

## KL-Divergence DRO

For ambiguity set `{Q : KL(Q || P_hat) <= rho}`, the worst-case expected loss is:

```
sup_{KL <= rho} E_Q[ ell ]  =  inf_{alpha > 0} alpha * log E_{P_hat}[ exp(ell / alpha) ] + alpha * rho
```

In CVXPY this is `cp.log_sum_exp(...)` plus a perspective term. Use `cp.kl_div` or `cp.rel_entr` as the building block. Tractable when `ell` is jointly convex in `x` and the exponential moment is bounded.

## Polyhedral Uncertainty (Bertsimas-Sim, Soyster)

`a^T x <= b` for all `a in U` where `U` is a polytope `{a : C a <= d}`:

```
sup_{a : C a <= d} a^T x <= b
iff exists y >= 0 with C^T y == x and d^T y <= b   (LP duality)
```

For deviations around a nominal vector, `a = a_hat + D u`, `C u <= d`, use:

```
a_hat^T x + sup_{C u <= d} u^T D^T x <= b
iff exists y >= 0 with C^T y == D^T x and a_hat^T x + d^T y <= b.
```

Pure LP, assuming the support set is nonempty and the support-function dual is
attained. Bertsimas-Sim "budget" uncertainty `sum |a_i - a_hat_i| / sigma_i <=
Gamma` gives an LP-tractable reformulation that interpolates between Soyster
(worst-case, conservative) and nominal (no robustness).

## Common Mistakes

- **Treating Wasserstein DRO as a min-max black box.** For Lipschitz losses, it is just a single regularized convex problem.
- **Wrong dual norm.** The ground metric on data and the regularizer on parameters must be a Holder dual pair.
- **Squared loss with W1.** Squared loss is not Lipschitz globally — the closed-form W1 equivalence does not apply. Use W2 with a different reformulation, or constrain the data domain.
- **Scenario-based chance constraints with too-few samples.** The required sample count for a confidence-(1-delta) inner approximation grows like `1/eps`, often >>1000. Bernstein safe approximations sidestep this.
- **Sign on `eps`.** A larger `eps` means a larger ambiguity ball means **more** regularization — flag for users who pass `eps` from a held-out validation routine.

## Reporting Discipline

- Name the ambiguity set (W1/W2/KL/moment/polyhedral) and the assumption that makes it tractable (Lipschitz loss, sub-Gaussian noise, polytope, etc).
- Identify whether the result is exact (Wasserstein-1 with Lipschitz loss is exact via duality) or a safe inner approximation (Bernstein).
- Compare against the nominal (non-DRO) solution: weight magnitude shrinkage is the qualitative DRO signature.
