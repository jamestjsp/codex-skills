---
name: convexify-with-cvxpy
description: Use when Codex needs to reformulate an optimization problem into a convex, DCP, DGP, DQCP, conic, mixed-integer convex, relaxed, or sequentially convex model and solve it with CVXPY. Trigger for requests about making a model convex, fixing CVXPY DCP/DGP/DQCP errors, choosing CVXPY atoms/solvers, lifting variables, epigraph or cone reformulations, semidefinite relaxations, geometric programming, quasiconvex bisection, robust convex optimization, benchmark nonconvex problems, or powered-descent/rocket-landing style convexification examples.
---

# Convexify With CVXPY

## Core Rule

Treat convexification as modeling work, not syntax repair. First preserve the user's original mathematical intent, then choose one of three outcomes:

- Exact convex reformulation: equivalent and CVXPY-certifiable.
- Convex relaxation: tractable bound or heuristic; state what was relaxed.
- Sequential/local convexification: iterative approximation; state that it is not a global convex proof unless a problem-specific theorem applies.

Never hide nonconvexity by forcing an expression through constants, `.value`, NumPy evaluation of variables, or unsupported atom compositions.

## Workflow

1. Write the original model explicitly: variables, parameters/data, objective, constraints, domains, units, and which quantities are decisions versus fixed data.
2. Classify the problem before rewriting:
   - Try DCP with `problem.is_dcp()`.
   - If all decision variables are positive and expressions are monomial/posynomial-like, try DGP with `problem.is_dgp()` and `solve(gp=True)`.
   - If the objective or a constraint is quasiconvex/quasiconcave, try DQCP with `problem.is_dqcp()` and `solve(qcp=True)`.
   - If the only nonconvexity is discrete selection, cardinality, or fixed-charge structure around convex continuous constraints, use mixed-integer DCP/MISOCP/MIQP rather than calling the whole model generic NLP.
   - If parameters change across repeated solves, check DPP with `problem.is_dcp(dpp=True)` or `problem.is_dgp(dpp=True)`.
3. Diagnose the failing expression tree, not only the top-level problem. Use expression `.curvature`, `.sign`, `.is_dcp()`, `.is_dgp()`, `.is_dqcp()`, `.log_log_curvature`, and the bundled script.
4. Reformulate using the smallest exact pattern that applies. Prefer CVXPY atoms over algebraically equivalent forms CVXPY cannot certify.
5. Validate with `assert problem.is_dcp()` / `is_dgp()` / `is_dqcp()` before solving. Use `solver=cp.CLARABEL` as the first open-source conic default unless the problem is a QP where `OSQP` is a better fit.
6. After solving, report status, objective value, primal variable values needed by the user, residual-sensitive caveats, and whether the returned model is exact, relaxed, or local.

## Reference Routing

Load **only** the reference file that matches the current modeling task. Use this routing table to pick:

| User signal | Load this reference |
|---|---|
| Single indefinite quadratic constraint, ratio of quadratics, "QCQP", trust region, GTRS, beamforming, sensor localization, "is there a way that isn't a local solver?" | `references/hidden-convexity-and-sdr.md` |
| Distribution shift, distributionally robust, Wasserstein DRO, ambiguity set, chance constraint, Bernstein approximation, KL-DRO, robust regression to outliers in the test distribution | `references/dro-and-chance-constraints.md` |
| `cp.log(cp.det(...))` errors, `cp.norm(X, 1)` confusion, "what atom should I use", `quad_form` with variable matrix, mixed-integer DCP not solving, DPP slow re-solve, complex Hermitian PSD, `cp.perspective` for fixed-charge / on-off costs | `references/cvxpy-atoms-tour.md` |
| Standard benchmark families named by acronym: H2 control, matrix completion, phase retrieval / PhaseLift, sparse PCA, Max-Cut, AC-OPF, pose graph, essential matrix, cardinality portfolio, Lennard-Jones | `references/benchmark-family-playbook.md` |
| Robust state estimation, Student-t / Cauchy heavy-tailed losses, IRLS, MM (majorization-minimization), repeated QP subproblems | `references/mm-robust-state-estimation.md` |
| Powered descent, rocket landing, trajectory optimization, lossless convexification, successive convexification, SCvx, virtual control + trust region | `references/rocket-landing-and-control.md` |
| General modeling workflow, grammar choice, solver choice, safety checks (start here when unsure) | `references/modeling-playbook.md` |
| Common exact rewrites, lifting tricks, cone forms, DGP/DQCP patterns, relaxation templates | `references/reformulation-patterns.md` |

`references/nonconvex-benchmark-suite-report.md` is the raw long-form benchmark report. Prefer the concise playbook during normal use.

## Bundled Scripts

- `scripts/cvxpy_convex_audit.py`: Load a CVXPY model from a Python file, print DCP/DGP/DQCP/DPP status, show noncompliant top-level pieces, inspect expression trees, and optionally solve a checked problem.
- `scripts/cvxpy_benchmark_smoke.py`: Build toy versions of representative convexified benchmark families and assert that CVXPY recognizes the resulting models as DCP.

## Quick Commands

Audit a module that exposes `build_problem()` or a `problem`/`prob` variable:

```bash
python3 ~/.codex/skills/convexify-with-cvxpy/scripts/cvxpy_convex_audit.py path/to/model.py
```

Audit and solve after the model is disciplined:

```bash
python3 ~/.codex/skills/convexify-with-cvxpy/scripts/cvxpy_convex_audit.py path/to/model.py --solve --mode auto
```

Run the script's smoke test:

```bash
python3 ~/.codex/skills/convexify-with-cvxpy/scripts/cvxpy_convex_audit.py --self-test
```

## Acceptance Checklist

- State whether the final model is exact, relaxed, or sequential/local.
- Include the CVXPY grammar used: DCP, DGP, DQCP, or outer-loop SCP/bisection around DCP.
- Use variable and parameter attributes when they are needed for sign/monotonicity analysis.
- Use persistent runnable code when changing a user's project; do not leave throwaway snippets as the only verification.
- Prefer atoms and conic forms CVXPY canonicalizes directly.
- Make solver choice explicit when it matters for cones, QP, SDP, EXP, POW, or mixed-integer structure.
