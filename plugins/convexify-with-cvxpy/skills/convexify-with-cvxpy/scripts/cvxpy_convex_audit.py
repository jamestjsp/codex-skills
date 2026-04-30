#!/usr/bin/env python3
"""
Audit CVXPY problem discipline and solve only after a grammar check passes.

Target modules should expose one of:
- build_problem()
- make_problem()
- problem
- prob
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from typing import Any


def _seed_cvxpy_version_from_checkout(cwd: Path) -> bool:
    versioning_path = cwd / "setup" / "versioning.py"
    if not versioning_path.is_file():
        return False
    spec = importlib.util.spec_from_file_location("_cvxpy_setup_versioning", versioning_path)
    if spec is None or spec.loader is None:
        return False
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    full_version, git_revision, commit_count = module.get_version_info()
    version_module = types.ModuleType("cvxpy.version")
    version_module.short_version = module.VERSION
    version_module.full_version = full_version
    version_module.git_revision = git_revision
    version_module.commit_count = commit_count
    version_module.release = module.IS_RELEASED
    version_module.version = module.VERSION if module.IS_RELEASED else full_version
    sys.modules["cvxpy.version"] = version_module
    return True


def _import_cvxpy():
    cwd = Path.cwd()
    if (cwd / "cvxpy").is_dir() and str(cwd) not in sys.path:
        sys.path.insert(0, str(cwd))
    try:
        import cvxpy as cp
    except ModuleNotFoundError as exc:
        if exc.name == "cvxpy.version" and _seed_cvxpy_version_from_checkout(cwd):
            sys.modules.pop("cvxpy", None)
            import cvxpy as cp
            return cp
        raise RuntimeError("cvxpy must be importable to use this script") from exc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("cvxpy must be importable to use this script") from exc
    return cp


def _bool_call(obj: Any, name: str, *args: Any, **kwargs: Any) -> bool | str:
    method = getattr(obj, name, None)
    if method is None:
        return "missing"
    try:
        return bool(method(*args, **kwargs))
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _value(obj: Any, name: str) -> Any:
    try:
        return getattr(obj, name)
    except Exception as exc:
        return f"error: {type(exc).__name__}: {exc}"


def _shape_text(obj: Any) -> str:
    shape = _value(obj, "shape")
    if shape == ():
        return "()"
    return str(shape)


def expression_info(expr: Any) -> dict[str, Any]:
    info = {
        "type": type(expr).__name__,
        "shape": _shape_text(expr),
        "sign": str(_value(expr, "sign")),
        "curvature": str(_value(expr, "curvature")),
        "log_log_curvature": str(_value(expr, "log_log_curvature")),
        "is_affine": _bool_call(expr, "is_affine"),
        "is_convex": _bool_call(expr, "is_convex"),
        "is_concave": _bool_call(expr, "is_concave"),
        "is_dcp": _bool_call(expr, "is_dcp"),
        "is_dgp": _bool_call(expr, "is_dgp"),
        "is_dqcp": _bool_call(expr, "is_dqcp"),
    }
    name = getattr(expr, "name", None)
    if callable(name):
        try:
            info["name"] = name()
        except Exception:
            pass
    return info


def _child_expressions(expr: Any) -> list[Any]:
    if hasattr(expr, "expr"):
        return [expr.expr]
    return list(getattr(expr, "args", []) or [])


def walk_expression(expr: Any, max_depth: int = 4, depth: int = 0) -> dict[str, Any]:
    node = expression_info(expr)
    if depth < max_depth:
        children = _child_expressions(expr)
        if children:
            node["children"] = [
                walk_expression(child, max_depth=max_depth, depth=depth + 1)
                for child in children
            ]
    return node


def discipline_summary(problem: Any) -> dict[str, Any]:
    return {
        "is_dcp": _bool_call(problem, "is_dcp"),
        "is_dcp_dpp": _bool_call(problem, "is_dcp", dpp=True),
        "is_dgp": _bool_call(problem, "is_dgp"),
        "is_dgp_dpp": _bool_call(problem, "is_dgp", dpp=True),
        "is_dqcp": _bool_call(problem, "is_dqcp"),
        "num_variables": len(problem.variables()),
        "num_parameters": len(problem.parameters()),
        "num_constraints": len(problem.constraints),
    }


def noncompliant_items(problem: Any, mode: str) -> list[dict[str, Any]]:
    check_name = {"dcp": "is_dcp", "dgp": "is_dgp", "dqcp": "is_dqcp"}[mode]
    items: list[tuple[str, Any]] = [("objective", problem.objective)]
    items += [(f"constraint[{i}]", c) for i, c in enumerate(problem.constraints)]
    failures = []
    for label, item in items:
        ok = _bool_call(item, check_name)
        if ok is not True:
            failures.append({"label": label, "check": check_name, "result": ok, "repr": str(item)})
    return failures


def choose_mode(problem: Any) -> str | None:
    if _bool_call(problem, "is_dcp") is True:
        return "dcp"
    if _bool_call(problem, "is_dgp") is True:
        return "dgp"
    if _bool_call(problem, "is_dqcp") is True:
        return "dqcp"
    return None


def solve_checked(problem: Any, mode: str = "auto", solver: str | None = None, **kwargs: Any) -> dict[str, Any]:
    cp = _import_cvxpy()
    chosen = choose_mode(problem) if mode == "auto" else mode
    if chosen not in {"dcp", "dgp", "dqcp"}:
        raise ValueError(f"problem is not DCP, DGP, or DQCP: {discipline_summary(problem)}")
    if chosen == "dcp" and problem.is_dcp() is not True:
        raise ValueError("requested DCP solve but problem.is_dcp() is false")
    if chosen == "dgp" and problem.is_dgp() is not True:
        raise ValueError("requested DGP solve but problem.is_dgp() is false")
    if chosen == "dqcp" and problem.is_dqcp() is not True:
        raise ValueError("requested DQCP solve but problem.is_dqcp() is false")

    solve_kwargs = dict(kwargs)
    if solver is not None:
        solve_kwargs["solver"] = getattr(cp, solver, solver)
    if chosen == "dgp":
        solve_kwargs["gp"] = True
    elif chosen == "dqcp":
        solve_kwargs["qcp"] = True
    value = problem.solve(**solve_kwargs)
    return {
        "mode": chosen,
        "status": problem.status,
        "value": value,
        "solver": getattr(problem.solver_stats, "solver_name", None),
        "solve_time": getattr(problem.solver_stats, "solve_time", None),
        "num_iters": getattr(problem.solver_stats, "num_iters", None),
    }


def load_problem(path: Path, builder: str | None = None) -> Any:
    cp = _import_cvxpy()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(path.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)

    names = [builder] if builder else ["build_problem", "make_problem", "problem", "prob"]
    for name in names:
        if not name:
            continue
        obj = getattr(module, name, None)
        if callable(obj):
            obj = obj()
        if isinstance(obj, cp.Problem):
            return obj
    raise ValueError(f"{path} does not expose a CVXPY Problem via {', '.join(names)}")


def print_problem_audit(problem: Any, mode: str = "auto", max_depth: int = 2) -> None:
    chosen = choose_mode(problem) if mode == "auto" else mode
    print("Discipline summary")
    print(json.dumps(discipline_summary(problem), indent=2, sort_keys=True))
    if chosen in {"dcp", "dgp", "dqcp"}:
        failures = noncompliant_items(problem, chosen)
        print(f"\nNoncompliant top-level items for {chosen.upper()}: {len(failures)}")
        print(json.dumps(failures, indent=2))
    else:
        for candidate in ("dcp", "dgp", "dqcp"):
            failures = noncompliant_items(problem, candidate)
            print(f"\nNoncompliant top-level items for {candidate.upper()}: {len(failures)}")
            print(json.dumps(failures, indent=2))
    print("\nObjective expression tree")
    print(json.dumps(walk_expression(problem.objective, max_depth=max_depth), indent=2))


def self_test() -> None:
    cp = _import_cvxpy()
    x = cp.Variable(2)
    problem = cp.Problem(cp.Minimize(cp.sum_squares(x - 1)), [x >= 0])
    assert problem.is_dcp()
    summary = discipline_summary(problem)
    assert summary["is_dcp"] is True
    result = solve_checked(problem, mode="dcp", solver="CLARABEL", canon_backend="SCIPY")
    assert result["status"] in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    print("self-test passed")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, help="Python file exposing a CVXPY Problem")
    parser.add_argument("--builder", help="Function or variable name that provides the problem")
    parser.add_argument("--mode", choices=["auto", "dcp", "dgp", "dqcp"], default="auto")
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--solve", action="store_true")
    parser.add_argument("--solver", help="Solver constant name, e.g. CLARABEL, OSQP, SCS")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON summary")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)

    if args.self_test:
        self_test()
        return 0
    if args.path is None:
        parser.error("path is required unless --self-test is used")

    problem = load_problem(args.path, builder=args.builder)
    if args.json:
        output: dict[str, Any] = {"summary": discipline_summary(problem)}
        if args.solve:
            output["solve"] = solve_checked(problem, mode=args.mode, solver=args.solver)
        print(json.dumps(output, indent=2, sort_keys=True))
    else:
        print_problem_audit(problem, mode=args.mode, max_depth=args.max_depth)
        if args.solve:
            print("\nSolve result")
            print(json.dumps(solve_checked(problem, mode=args.mode, solver=args.solver), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
