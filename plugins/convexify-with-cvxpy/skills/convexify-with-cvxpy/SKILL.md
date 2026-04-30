---
name: convexify-with-cvxpy
description: Use when Codex needs to reformulate an optimization problem into a convex, DCP, DGP, DQCP, conic, or sequentially convex model and solve it with CVXPY. Trigger for requests about making a model convex, fixing CVXPY DCP/DGP/DQCP errors, choosing CVXPY atoms/solvers, lifting variables, epigraph or cone reformulations, geometric programming, quasiconvex bisection, robust convex optimization, or powered-descent/rocket-landing style convexification examples.
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
   - If parameters change across repeated solves, check DPP with `problem.is_dcp(dpp=True)` or `problem.is_dgp(dpp=True)`.
3. Diagnose the failing expression tree, not only the top-level problem. Use expression `.curvature`, `.sign`, `.is_dcp()`, `.is_dgp()`, `.is_dqcp()`, `.log_log_curvature`, and the bundled script.
4. Reformulate using the smallest exact pattern that applies. Prefer CVXPY atoms over algebraically equivalent forms CVXPY cannot certify.
5. Validate with `assert problem.is_dcp()` / `is_dgp()` / `is_dqcp()` before solving. Use `solver=cp.CLARABEL` as the first open-source conic default unless the problem is a QP where `OSQP` is a better fit.
6. After solving, report status, objective value, primal variable values needed by the user, residual-sensitive caveats, and whether the returned model is exact, relaxed, or local.

## Bundled Resources

- `scripts/cvxpy_convex_audit.py`: Load a CVXPY model from a Python file, print DCP/DGP/DQCP/DPP status, show noncompliant top-level pieces, inspect expression trees, and optionally solve a checked problem.
- `references/modeling-playbook.md`: Practical convexification workflow, grammar choice, solver choice, and safety checks.
- `references/reformulation-patterns.md`: Common exact rewrites, lifting tricks, cone forms, DGP/DQCP patterns, and relaxation templates.
- `references/mm-robust-state-estimation.md`: Majorization-minimization and IRLS patterns for Student-t or Cauchy-style robust losses, constrained state estimation, and repeated CVXPY QP/QCQP subproblems.
- `references/rocket-landing-and-control.md`: Powered-descent and SpaceX-style convexification patterns, including lossless convexification and successive convexification sketches.

Load only the reference file that matches the current modeling task.

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
