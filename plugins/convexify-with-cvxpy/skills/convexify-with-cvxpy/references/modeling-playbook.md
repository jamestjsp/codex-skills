# CVXPY Convex Modeling Playbook

## Table of Contents

- Goal and contract
- CVXPY grammar routing
- Diagnostic process
- Modeling heuristics
- Solver selection
- Verification checklist

## Goal and Contract

Convert a user's optimization idea into a CVXPY-solvable model while preserving the mathematical intent. If exact convexification is not available, explicitly label the result as a relaxation, approximation, or sequential local method.

Ask for missing data only when it blocks the formulation: objective direction, fixed versus decision quantities, variable domains, units, tolerance, and whether an approximate or relaxed solution is acceptable.

## CVXPY Grammar Routing

Use the narrowest grammar that matches the model:

- DCP: convex minimization, concave maximization, affine equalities, and convex <= concave inequalities.
- DGP: positive variables and parameters with monomial/posynomial/log-log-convex structure. Declare variables as `cp.Variable(pos=True)` and solve with `problem.solve(gp=True)`.
- DQCP: quasiconvex minimization or quasiconcave maximization over a convex feasible set. Solve with `problem.solve(qcp=True)`, optionally with `low=` and `high=` for bisection bounds.
- DPP: parametrized DCP/DGP models intended for repeated solves. Keep nonlinear parameter transformations outside CVXPY or introduce auxiliary variables so parameter use remains DPP-compliant.
- SCP/CCP: use only when the original model is truly nonconvex and an exact convex reformulation is not known. Linearize nonconvex parts around a current point, add trust regions and slack penalties, and solve a sequence of DCP problems.

## Diagnostic Process

Use this sequence:

```python
print(problem.is_dcp(), problem.is_dgp(), problem.is_dqcp())
print(problem.is_dcp(dpp=True))
print(objective_expr.curvature, objective_expr.sign)
for c in constraints:
    print(c, c.is_dcp(), c.is_dgp(), c.is_dqcp())
```

When CVXPY says curvature is unknown, inspect the smallest failing subexpression. CVXPY's analysis is conservative: a mathematically convex expression can still fail if expressed in a non-DCP form. Classic example:

```python
cp.sqrt(1 + cp.square(x))      # unknown under DCP
cp.norm(cp.hstack([1, x]), 2)  # convex and DCP
```

Use attributes to inform the analyzer:

```python
x = cp.Variable(nonneg=True)
p = cp.Parameter(nonpos=True)
z = cp.Variable(pos=True)      # required for DGP variables
X = cp.Variable((n, n), PSD=True)
```

Remember that explicit constraints such as `x >= 0` do not always give the same sign information to DCP analysis as `Variable(nonneg=True)`. Use attributes for analysis; use explicit constraints when dual values are needed.

## Modeling Heuristics

Prefer these routes, in order:

1. Change syntax to a known atom: `norm`, `sum_squares`, `quad_form`, `matrix_frac`, `quad_over_lin`, `log_sum_exp`, `rel_entr`, `geo_mean`, `maximum`, `minimum`.
2. Add lifted variables and epigraph/hypograph constraints.
3. Convert a robust or norm-bounded family of inequalities to cone constraints.
4. Try DGP for products, powers, ratios, and posynomials over positive variables.
5. Try DQCP for ratios and max-min fairness objectives; CVXPY will solve by bisection.
6. Relax nonconvex equalities or rank/product constraints when a bound is useful.
7. Use sequential convexification with trust regions when only a local trajectory or controller is needed.

Common anti-patterns:

- `x * y` where both are variables. Use DGP if positive and monomial-like, McCormick envelopes if bounded and relaxing, or an outer approximation/SCP if local.
- `norm(x) >= c`. Superlevel sets of convex functions are generally nonconvex. Look for a lossless relaxation, disjunction/MIP, or a problem-specific theorem.
- `convex == something` unless both sides are affine. Replace equality to a convex expression with a one-sided epigraph only if the objective forces tightness.
- `minimize(concave)` or `maximize(convex)`. Flip the objective only if the mathematical objective actually changes to an equivalent convex form.

## Solver Selection

Start with these defaults:

- LP/QP with linear constraints: default or `cp.OSQP` for QP.
- SOCP/EXP/POW/SDP: `cp.CLARABEL` first for open-source conic solves.
- Very large or first-order tolerant conic models: `cp.SCS`.
- Mixed-integer models: installed MIP solver such as `GLPK_MI`, `CBC`, `SCIP`, commercial solvers if available.
- DGP: conic solver through `solve(gp=True)`.
- DQCP: `solve(qcp=True)` with bisection; provide bounds when possible.

Check installed solvers:

```python
import cvxpy as cp
print(cp.installed_solvers())
```

## Verification Checklist

Before delivering a reformulation:

- Assert the grammar: `assert problem.is_dcp()` or `assert problem.is_dgp()` or `assert problem.is_dqcp()`.
- Solve with a named solver when reproducibility matters.
- Check `problem.status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}` and explain `*_INACCURATE`.
- Validate domain constraints, especially positivity for logs/DGP and PSD/symmetry for matrix atoms.
- Compare against a small brute-force grid or simulation when the reformulation is subtle.
- For relaxations, inspect tightness of lifted constraints such as `abs(x) <= t`, `norm(u) <= sigma`, `X >> x @ x.T` surrogates, or McCormick envelopes.
- For SCP, report convergence criteria, trust-region behavior, slack values, and sensitivity to initialization.

## Source Anchors

- CVXPY DCP tutorial: https://www.cvxpy.org/tutorial/dcp/index.html
- CVXPY DGP tutorial: https://www.cvxpy.org/tutorial/dgp/index.html
- CVXPY DQCP tutorial: https://www.cvxpy.org/tutorial/dqcp/index.html
- CVXPY solver tutorial: https://www.cvxpy.org/tutorial/solvers/
- CVXPY advanced constraints: https://www.cvxpy.org/tutorial/constraints/index.html
