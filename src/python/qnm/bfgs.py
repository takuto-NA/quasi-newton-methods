from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .line_search import line_search
from .utils import OptimizeResult, ensure_1d, grad_norm


def bfgs(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-6,
    line_search_kwargs: Optional[dict] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Basic BFGS optimizer with strong-Wolfe line search."""
    line_search_kwargs = line_search_kwargs or {}
    x = ensure_1d(x0)
    n = x.size
    n_fun = 0
    n_grad = 0

    f = float(fun(x))
    g = grad(x)
    n_fun += 1
    n_grad += 1
    H = np.eye(n)

    for k in range(1, max_iter + 1):
        if grad_norm(g) <= tol:
            return OptimizeResult(x, f, g, k - 1, n_fun, n_grad, True, "converged", "Gradient norm below tolerance")

        p = -H @ g
        alpha, f_new, g_new, ls_fun, ls_grad = line_search(fun, grad, x, p, f0=f, g0=g, **line_search_kwargs)
        n_fun += ls_fun
        n_grad += ls_grad
        s = alpha * p
        if alpha == 0.0:
            return OptimizeResult(x, f, g, k - 1, n_fun, n_grad, False, "line_search_failed", "Line search failed to find descent")

        x_new = x + s
        y = g_new - g
        ys = float(np.dot(y, s))
        if ys <= 1e-12:
            # Reset to identity if curvature is lost; continue optimization.
            H = np.eye(n)
        else:
            rho = 1.0 / ys
            I = np.eye(n)
            H = (I - rho * np.outer(s, y)) @ H @ (I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        x, f, g = x_new, f_new, g_new

        if callback is not None:
            callback(OptimizeResult(x, f, g, k, n_fun, n_grad, True, "iter", "In-progress"))

    return OptimizeResult(x, f, g, max_iter, n_fun, n_grad, False, "max_iter", "Reached maximum iterations")

