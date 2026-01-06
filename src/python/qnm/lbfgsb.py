from __future__ import annotations

from typing import Callable, Iterable, Optional, Sequence, Tuple

import numpy as np

from .utils import OptimizeResult, ensure_1d


def lbfgsb(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    bounds: Optional[Sequence[Tuple[Optional[float], Optional[float]]]] = None,
    max_iter: int = 15000,
    tol: float = 1e-6,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
    **kwargs,
) -> OptimizeResult:
    """L-BFGS-B via SciPy's reference implementation.

    This keeps the API consistent with qnm.bfgs/lbfgs while delegating bound
    handling to SciPy. SciPy is an optional dependency; an informative error is
    raised if it is missing.

    Notes on counters:

    - `n_fun`: number of objective evaluations. When using SciPy's `fmin_l_bfgs_b`
      with a callable that returns both `(f, g)`, SciPy reports `funcalls` which
      corresponds to how many times that callable was invoked.
    - `n_grad`: since the same callable returns the gradient each time, we set
      `n_grad == n_fun == funcalls`.
    """
    try:
        from scipy.optimize import fmin_l_bfgs_b
    except ImportError as exc:  # pragma: no cover - exercised only without SciPy
        raise ImportError("SciPy is required for lbfgsb; install qnm[dev] or scipy") from exc

    x0 = ensure_1d(x0)

    def f_and_g(x: np.ndarray) -> Tuple[float, np.ndarray]:
        return float(fun(x)), grad(x)

    x_opt, f_opt, info = fmin_l_bfgs_b(f_and_g, x0, bounds=bounds, pgtol=tol, maxiter=max_iter, **kwargs)
    grad_opt = info.get("grad", np.zeros_like(x_opt))
    success = info.get("warnflag", 1) == 0
    message = info.get("task", "unknown")

    result = OptimizeResult(
        x=ensure_1d(x_opt),
        fun=float(f_opt),
        grad=ensure_1d(grad_opt),
        n_iter=int(info.get("nit", 0)),
        n_fun=int(info.get("funcalls", 0)),
        n_grad=int(info.get("funcalls", 0)),
        success=success,
        status="converged" if success else "warning",
        message=message,
    )
    if callback is not None:
        callback(result)
    return result
