from __future__ import annotations

import platform
import sys
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

from qnm.bfgs import bfgs
from qnm.lbfgs import lbfgs
from qnm.problems import Problem, quadratic_problem, rosenbrock_problem
from qnm.utils import gradient_check, grad_norm


@dataclass(frozen=True)
class ReferenceResult:
    method: str
    fun: float
    nit: Optional[int]
    nfev: Optional[int]


def _try_scipy_bfgs(problem: Problem, tol: float) -> Optional[ReferenceResult]:
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        return None

    scipy_res = minimize(problem.fun, problem.x0, method="BFGS", jac=problem.grad, tol=tol)
    return ReferenceResult(method="scipy:BFGS", fun=float(scipy_res.fun), nit=getattr(scipy_res, "nit", None), nfev=getattr(scipy_res, "nfev", None))


def _try_scipy_lbfgsb_nobounds(problem: Problem, tol: float) -> Optional[ReferenceResult]:
    """Informational-only reference for L-BFGS: SciPy's L-BFGS-B without bounds."""
    try:
        from scipy.optimize import minimize  # type: ignore
    except Exception:
        return None

    scipy_res = minimize(problem.fun, problem.x0, method="L-BFGS-B", jac=problem.grad, tol=tol)
    return ReferenceResult(
        method="scipy:L-BFGS-B(no-bounds)",
        fun=float(scipy_res.fun),
        nit=getattr(scipy_res, "nit", None),
        nfev=getattr(scipy_res, "nfev", None),
    )


def _status(
    *,
    primary_ok: bool,
    f: float,
    reference: Optional[ReferenceResult],
    f_diff_tol: float,
) -> str:
    """Evidence status aligned with docs/evidence/methodology.md."""
    if not primary_ok:
        return "FAILED"
    if reference is None:
        return "PASSED"
    f_diff = abs(f - reference.fun)
    if f_diff <= f_diff_tol:
        return "PASSED"
    # Differences can be acceptable when solver passes primary checks but reference ends elsewhere
    return "ACCEPTABLE_DIFF"


def verify_one(
    solver_name: str,
    solver_func: Callable[..., object],
    problem: Problem,
    tol: float,
    reference: Optional[ReferenceResult],
) -> dict:
    gradcheck_ok, _, _, gradcheck_diff = gradient_check(problem.fun, problem.grad, problem.x0)

    history_f: list[float] = []

    def _cb(r) -> None:
        # callback receives OptimizeResult(x, f, g, ...)
        history_f.append(float(getattr(r, "fun")))

    res = solver_func(problem.fun, problem.grad, problem.x0, tol=tol, callback=_cb)

    # qnm result contract
    success = bool(getattr(res, "success"))
    f = float(getattr(res, "fun"))
    g = np.asarray(getattr(res, "grad"), dtype=float)
    gnorm = grad_norm(g)
    iters = int(getattr(res, "n_iter"))
    fevals = int(getattr(res, "n_fun"))

    # Primary checks (methodology.md)
    primary_ok = success and gradcheck_ok and (gnorm <= 10.0 * tol)
    x_err = None
    if getattr(problem, "solution", None) is not None:
        x_star = np.asarray(problem.solution, dtype=float)
        x = np.asarray(getattr(res, "x"), dtype=float)
        x_err = float(np.linalg.norm(x - x_star))
        primary_ok = primary_ok and (x_err <= 1e-3)

    monotone_ok = True
    for a, b in zip(history_f, history_f[1:]):
        if b > a + 1e-12:
            monotone_ok = False
            break
    # Only enforce monotonic decrease when we actually saw iterations (history exists).
    if history_f:
        primary_ok = primary_ok and monotone_ok

    f_diff = None if reference is None else abs(f - reference.fun)

    status = _status(primary_ok=primary_ok, f=f, reference=reference, f_diff_tol=1e-4)

    return {
        "problem": problem.name,
        "solver": solver_name,
        "success": success,
        "f": f,
        "gnorm": gnorm,
        "iters": iters,
        "fevals": fevals,
        "f_diff": f_diff,
        "ref_method": reference.method if reference else None,
        "x_err": x_err,
        "gradcheck_ok": gradcheck_ok,
        "gradcheck_diff": float(gradcheck_diff),
        "monotone_ok": monotone_ok if history_f else None,
        "status": status,
    }


def _print_environment():
    print("## Environment")
    print(f"- Python: {sys.version.split()[0]}")
    print(f"- Platform: {platform.platform()}")
    try:
        import numpy as _np  # noqa: N812

        print(f"- NumPy: {_np.__version__}")
    except Exception:
        pass
    try:
        import scipy as _sp  # type: ignore

        print(f"- SciPy: {_sp.__version__}")
    except Exception:
        print("- SciPy: (not installed)")


def _print_markdown_table(results: list[dict]) -> None:
    # Keep the table compatible with docs/evidence/baseline_results.md
    print("| Problem | Solver | Success | f(x*) | ‖∇f‖∞ | Iters | Func evals | f_diff (SciPy) | Status |")
    print("|---------|--------|---------|-------|-------|-------|------------|----------------|--------|")
    for r in results:
        success_icon = "✓" if r["success"] else "✗"
        f_diff = r["f_diff"]
        f_diff_str = "—" if f_diff is None else f"{f_diff:.1e}"
        print(
            f"| {r['problem']} | {r['solver']} | {success_icon} | {r['f']:.4e} | {r['gnorm']:.1e} | {r['iters']} | {r['fevals']} | {f_diff_str} | {r['status']} |"
        )


def main():
    tol = 1e-6
    problems = [
        rosenbrock_problem(dim=2),
        rosenbrock_problem(dim=10),
        quadratic_problem(dim=5, condition_number=10),
        quadratic_problem(dim=50, condition_number=100),
    ]

    results: list[dict] = []
    for prob in problems:
        prob_label = f"{prob.name} (d={len(np.asarray(prob.x0))})"

        # BFGS: reference comparison (SciPy BFGS) if available
        bfgs_ref = _try_scipy_bfgs(prob, tol=tol)
        r = verify_one("BFGS", bfgs, prob, tol=tol, reference=bfgs_ref)
        r["problem"] = prob_label
        results.append(r)

        # L-BFGS: informational reference only (SciPy L-BFGS-B without bounds) if available
        lbfgs_ref = _try_scipy_lbfgsb_nobounds(prob, tol=tol)
        r = verify_one("L-BFGS", lbfgs, prob, tol=tol, reference=lbfgs_ref)
        r["problem"] = prob_label
        results.append(r)

    print("# Baseline Results (Generated)")
    _print_environment()
    print("\n## Verification Table\n")
    _print_markdown_table(results)
    print("\n## Notes\n")
    print("- BFGS is compared against SciPy BFGS when SciPy is installed.")
    print("- L-BFGS uses SciPy L-BFGS-B (no bounds) as an informational reference only; primary checks are property-based.")


if __name__ == "__main__":
    main()

