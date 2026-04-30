#!/usr/bin/env python3
"""
Build toy convexified benchmark models and verify CVXPY discipline checks.

This script does not solve real benchmark instances. It checks that the core
CVXPY skeletons used in the benchmark-family playbook are DCP-compliant.
"""

from __future__ import annotations

import json
import math
from typing import Callable

import numpy as np

from cvxpy_convex_audit import _import_cvxpy


def h2_state_feedback_lmi(cp):
    n, m = 3, 1
    A = np.array([[0.2, 1.0, 0.0], [-1.0, 0.1, 0.4], [0.0, -0.3, -0.2]])
    B = np.array([[0.0], [1.0], [0.5]])
    Q = np.eye(n)
    R_sqrt = np.eye(m)
    W = np.eye(n)

    P = cp.Variable((n, n), PSD=True)
    Y = cp.Variable((m, n))
    Z = cp.Variable((m, m), PSD=True)
    r_y = R_sqrt @ Y
    constraints = [
        P >> 1e-3 * np.eye(n),
        A @ P + P @ A.T + B @ Y + Y.T @ B.T + W << 0,
        cp.bmat([[Z, r_y], [r_y.T, P]]) >> 0,
    ]
    return cp.Problem(cp.Minimize(cp.trace(Q @ P) + cp.trace(Z)), constraints)


def matrix_completion_nuclear(cp):
    M = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0], [4.0, 0.0, 5.0]])
    mask = (M != 0).astype(float)
    X = cp.Variable(M.shape)
    objective = cp.Minimize(
        0.5 * cp.sum_squares(cp.multiply(mask, X - M)) + 0.1 * cp.norm(X, "nuc")
    )
    return cp.Problem(objective)


def phase_lift_sdp(cp):
    n = 3
    measurements = [
        np.array([1.0, 0.0, 1.0]),
        np.array([0.0, 1.0, -1.0]),
        np.array([1.0, 1.0, 0.0]),
    ]
    y = [1.0, 0.25, 2.25]
    X = cp.Variable((n, n), PSD=True)
    constraints = [
        cp.trace(np.outer(a, a) @ X) == yi for a, yi in zip(measurements, y)
    ]
    return cp.Problem(cp.Minimize(cp.trace(X)), constraints)


def sparse_pca_sdp(cp):
    sigma = np.array([[2.0, 0.3, 0.0], [0.3, 1.5, 0.2], [0.0, 0.2, 1.0]])
    X = cp.Variable((3, 3), PSD=True)
    constraints = [cp.trace(X) == 1, cp.sum(cp.abs(X)) <= 2.0]
    return cp.Problem(cp.Maximize(cp.trace(sigma @ X)), constraints)


def max_cut_sdp(cp):
    W = np.array([[0.0, 1.0, 2.0], [1.0, 0.0, 1.5], [2.0, 1.5, 0.0]])
    X = cp.Variable(W.shape, PSD=True)
    objective = cp.Maximize(0.25 * cp.sum(cp.multiply(W, 1 - X)))
    return cp.Problem(objective, [cp.diag(X) == 1])


def ac_opf_voltage_sdp_skeleton(cp):
    n = 3
    W = cp.Variable((n, n), PSD=True)
    voltage_min = np.array([0.95, 0.95, 0.95]) ** 2
    voltage_max = np.array([1.05, 1.05, 1.05]) ** 2
    c = np.array([1.0, 0.2, 0.1])
    constraints = [
        cp.diag(W) >= voltage_min,
        cp.diag(W) <= voltage_max,
        cp.trace(W) <= 3.2,
    ]
    return cp.Problem(cp.Minimize(c @ cp.diag(W)), constraints)


def pose_graph_rotation_sdp_skeleton(cp):
    d, n_poses = 2, 3
    size = d * n_poses
    C = np.eye(size)
    X = cp.Variable((size, size), PSD=True)
    constraints = []
    for i in range(n_poses):
        block = X[i * d:(i + 1) * d, i * d:(i + 1) * d]
        constraints.append(block == np.eye(d))
    return cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)


def portfolio_micp(cp):
    n = 4
    mu = np.array([0.08, 0.06, 0.11, 0.04])
    Sigma = np.diag([0.06, 0.04, 0.09, 0.03])
    upper = 0.7 * np.ones(n)
    lower = 0.05 * np.ones(n)
    x = cp.Variable(n)
    y = cp.Variable(n, boolean=True)
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        x <= cp.multiply(upper, y),
        x >= cp.multiply(lower, y),
        cp.sum(y) <= 3,
        cp.quad_form(x, Sigma) <= 0.05,
    ]
    return cp.Problem(cp.Maximize(mu @ x), constraints)


def portfolio_continuous_relaxation(cp):
    n = 4
    mu = np.array([0.08, 0.06, 0.11, 0.04])
    Sigma = np.diag([0.06, 0.04, 0.09, 0.03])
    upper = 0.7 * np.ones(n)
    lower = 0.05 * np.ones(n)
    x = cp.Variable(n)
    y = cp.Variable(n)
    constraints = [
        cp.sum(x) == 1,
        x >= 0,
        y >= 0,
        y <= 1,
        x <= cp.multiply(upper, y),
        x >= cp.multiply(lower, y),
        cp.sum(y) <= 3,
        cp.quad_form(x, Sigma) <= 0.05,
    ]
    return cp.Problem(cp.Maximize(mu @ x), constraints)


def alpha_bb_lower_bound_skeleton(cp):
    x = cp.Variable(2)
    center = np.array([0.2, -0.1])
    gradient = np.array([1.0, -0.5])
    alpha = 0.25
    lower_model = gradient @ (x - center) + 0.5 * alpha * cp.sum_squares(x - center)
    constraints = [x >= -1, x <= 1]
    return cp.Problem(cp.Minimize(lower_model), constraints)


CASES: dict[str, Callable] = {
    "h2_state_feedback_lmi": h2_state_feedback_lmi,
    "matrix_completion_nuclear": matrix_completion_nuclear,
    "phase_lift_sdp": phase_lift_sdp,
    "sparse_pca_sdp": sparse_pca_sdp,
    "max_cut_sdp": max_cut_sdp,
    "ac_opf_voltage_sdp_skeleton": ac_opf_voltage_sdp_skeleton,
    "pose_graph_rotation_sdp_skeleton": pose_graph_rotation_sdp_skeleton,
    "portfolio_micp": portfolio_micp,
    "portfolio_continuous_relaxation": portfolio_continuous_relaxation,
    "alpha_bb_lower_bound_skeleton": alpha_bb_lower_bound_skeleton,
}


def solve_problem(cp, problem):
    candidates = ["HIGHS", "SCIPY"] if problem.is_mixed_integer() else ["CLARABEL", "SCS"]
    last_error = None
    for solver in candidates:
        try:
            value = problem.solve(solver=getattr(cp, solver), verbose=False)
            clean_value = None
            if value is not None and not (isinstance(value, float) and math.isnan(value)):
                clean_value = float(value)
            return {
                "status": problem.status,
                "value": clean_value,
                "solver": solver,
                "error": None,
            }
        except Exception as exc:
            last_error = f"{solver}: {type(exc).__name__}: {exc}"
    return {
        "status": None,
        "value": None,
        "solver": None,
        "error": last_error,
    }


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solve", action="store_true", help="Solve each toy model after DCP checks")
    args = parser.parse_args()

    cp = _import_cvxpy()
    results = []
    for name, build in CASES.items():
        problem = build(cp)
        result = {
            "name": name,
            "is_dcp": problem.is_dcp(),
            "is_dgp": problem.is_dgp(),
            "is_dqcp": problem.is_dqcp(),
            "is_mixed_integer": problem.is_mixed_integer(),
            "variables": len(problem.variables()),
            "constraints": len(problem.constraints),
        }
        if args.solve:
            result["solve"] = solve_problem(cp, problem)
        results.append(result)
        if not result["is_dcp"]:
            raise AssertionError(f"{name} is not DCP: {result}")
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
