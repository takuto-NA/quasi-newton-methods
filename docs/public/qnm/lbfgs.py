from __future__ import annotations

from collections import deque
from typing import Callable, Deque, Optional, Tuple

import numpy as np

from .line_search import line_search
from .utils import OptimizeResult, ensure_1d, grad_norm


def two_loop_recursion(
    grad_k: np.ndarray,
    s_history: Deque[np.ndarray],
    y_history: Deque[np.ndarray],
) -> np.ndarray:
    """Compute -H_k * grad_k using the L-BFGS two-loop recursion.

    Following Algorithm 7.4 in Nocedal & Wright,
    'Numerical Optimization' (2nd Ed, 2006, p. 178).
    """
    q = grad_k.copy()
    alpha_list: list[float] = []
    rho_list: list[float] = []

    for s, y in reversed(list(zip(s_history, y_history))):
        rho = 1.0 / float(np.dot(y, s))
        rho_list.append(rho)
        alpha = rho * float(np.dot(s, q))
        alpha_list.append(alpha)
        q = q - alpha * y

    if y_history:
        last_s = s_history[-1]
        last_y = y_history[-1]
        # H_k^0 scaling factor (Eq. 7.20, p. 178)
        gamma = float(np.dot(last_s, last_y) / np.dot(last_y, last_y))
    else:
        gamma = 1.0
    r = gamma * q

    for (s, y, alpha, rho) in zip(s_history, y_history, reversed(alpha_list), reversed(rho_list)):
        beta = rho * float(np.dot(y, r))
        r = r + s * (alpha - beta)

    return -r


def lbfgs(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x0: np.ndarray,
    m: int = 10,
    max_iter: int = 200,
    tol: float = 1e-6,
    line_search_kwargs: Optional[dict] = None,
    callback: Optional[Callable[[OptimizeResult], None]] = None,
) -> OptimizeResult:
    """Limited-memory BFGS with strong-Wolfe line search.

    This implementation follows the L-BFGS method described in Chapter 7
    of Nocedal & Wright, 'Numerical Optimization' (2nd Ed, 2006).
    """
    line_search_kwargs = line_search_kwargs or {}
    x = ensure_1d(x0)
    n_fun = 0
    n_grad = 0

    f = float(fun(x))
    g = grad(x)
    n_fun += 1
    n_grad += 1

    s_history: Deque[np.ndarray] = deque(maxlen=m)
    y_history: Deque[np.ndarray] = deque(maxlen=m)

    for k in range(1, max_iter + 1):
        if grad_norm(g) <= tol:
            return OptimizeResult(x, f, g, k - 1, n_fun, n_grad, True, "converged", "Gradient norm below tolerance")

        p = two_loop_recursion(g, s_history, y_history)
        if np.dot(p, g) >= 0:
            # Reset memory if direction is not descent.
            s_history.clear()
            y_history.clear()
            p = -g

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
            s_history.clear()
            y_history.clear()
        else:
            s_history.append(s)
            y_history.append(y)

        x, f, g = x_new, f_new, g_new

        if callback is not None:
            res = OptimizeResult(x, f, g, k, n_fun, n_grad, True, "iter", "In-progress")
            # Attach s_history and y_history for visualization
            res.extra_info = {
                "s_history": [s_i.tolist() for s_i in s_history],
                "y_history": [y_i.tolist() for y_i in y_history]
            }
            callback(res)

    return OptimizeResult(x, f, g, max_iter, n_fun, n_grad, False, "max_iter", "Reached maximum iterations")

