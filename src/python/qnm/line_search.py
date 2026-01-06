from __future__ import annotations

from typing import Callable, Tuple

import numpy as np

from .utils import ensure_1d


def _strong_wolfe(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    xk: np.ndarray,
    pk: np.ndarray,
    f0: float,
    g0: np.ndarray,
    alpha0: float,
    c1: float,
    c2: float,
    max_iter: int,
    alpha_max: float,
) -> Tuple[float, float, np.ndarray, int, int]:
    """Line search satisfying strong Wolfe conditions (Nocedal-Wright, Alg. 3.5).

    The strong Wolfe conditions (Eq. 3.7, p. 34) are:
    1. f(xk + alpha*pk) <= f(xk) + c1 * alpha * grad_f(xk).T @ pk (Sufficient decrease)
    2. |grad_f(xk + alpha*pk).T @ pk| <= c2 * |grad_f(xk).T @ pk| (Curvature condition)
    """
    n_fun = 0
    n_grad = 0
    phi0 = f0
    derphi0 = float(np.dot(g0, pk))
    if derphi0 >= 0:
        # Not a descent direction; fallback to tiny step.
        return 0.0, f0, g0, n_fun, n_grad

    def eval_phi(alpha: float) -> Tuple[float, np.ndarray, float]:
        x_new = xk + alpha * pk
        f_new = float(fun(x_new))
        g_new = grad(x_new)
        return f_new, g_new, float(np.dot(g_new, pk))

    alpha_prev = 0.0
    f_prev = phi0
    derphi_prev = derphi0
    alpha = alpha0
    f_curr = phi0
    g_curr = g0
    derphi = derphi0

    for i in range(max_iter):
        f_curr, g_curr, derphi = eval_phi(alpha)
        n_fun += 1
        n_grad += 1

        # Sufficient decrease condition (Eq. 3.7a, p. 33)
        if (f_curr > phi0 + c1 * alpha * derphi0) or (i > 0 and f_curr >= f_prev):
            return _zoom(
                fun, grad, xk, pk, phi0, derphi0, alpha_prev, alpha, f_prev, derphi_prev, c1, c2, max_iter, alpha_max, n_fun, n_grad
            )

        # Curvature condition (Eq. 3.7b, p. 34)
        if abs(derphi) <= -c2 * derphi0:
            return alpha, f_curr, g_curr, n_fun, n_grad

        if derphi >= 0:
            return _zoom(
                fun, grad, xk, pk, phi0, derphi0, alpha, alpha_prev, f_curr, derphi, c1, c2, max_iter, alpha_max, n_fun, n_grad
            )

        alpha_prev = alpha
        f_prev = f_curr
        derphi_prev = derphi
        alpha = min(alpha * 2.0, alpha_max)

    return alpha, f_curr, g_curr, n_fun, n_grad


def _zoom(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    xk: np.ndarray,
    pk: np.ndarray,
    phi0: float,
    derphi0: float,
    alo: float,
    ahi: float,
    f_lo: float,
    derphi_lo: float,
    c1: float,
    c2: float,
    max_iter: int,
    alpha_max: float,
    n_fun: int,
    n_grad: int,
) -> Tuple[float, float, np.ndarray, int, int]:
    """Zoom phase of strong-Wolfe line search (Algorithm 3.6, p. 61)."""
    alpha = alo
    f_curr = f_lo
    g_curr = np.zeros_like(pk)
    derphi = derphi_lo

    def eval_phi(a: float) -> Tuple[float, np.ndarray, float]:
        x_new = xk + a * pk
        f_new = float(fun(x_new))
        g_new = grad(x_new)
        return f_new, g_new, float(np.dot(g_new, pk))

    for _ in range(max_iter):
        alpha = 0.5 * (alo + ahi)
        f_curr, g_curr, derphi = eval_phi(alpha)
        n_fun += 1
        n_grad += 1

        if (f_curr > phi0 + c1 * alpha * derphi0) or (f_curr >= f_lo):
            ahi = alpha
        else:
            if abs(derphi) <= -c2 * derphi0:
                return alpha, f_curr, g_curr, n_fun, n_grad
            if derphi * (ahi - alo) >= 0:
                ahi = alo
            alo = alpha
            f_lo = f_curr
            derphi_lo = derphi

        if alpha <= 1e-12 or alpha > alpha_max:
            break

    return alpha, f_curr, g_curr, n_fun, n_grad


def line_search(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    xk: np.ndarray,
    pk: np.ndarray,
    f0: float | None = None,
    g0: np.ndarray | None = None,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    max_iter: int = 25,
    alpha_max: float = 50.0,
) -> Tuple[float, float, np.ndarray, int, int]:
    """Run a strong-Wolfe line search.

    Returns (alpha, f_new, g_new, n_fun, n_grad) where the counts only include
    evaluations performed inside this routine.
    """
    xk = ensure_1d(xk)
    pk = ensure_1d(pk)
    if f0 is None:
        f0 = float(fun(xk))
        n_fun = 1
    else:
        n_fun = 0
    if g0 is None:
        g0 = grad(xk)
        n_grad = 1
    else:
        n_grad = 0

    alpha, f_new, g_new, extra_fun, extra_grad = _strong_wolfe(
        fun, grad, xk, pk, f0, g0, alpha0, c1, c2, max_iter, alpha_max
    )
    return alpha, f_new, g_new, n_fun + extra_fun, n_grad + extra_grad

