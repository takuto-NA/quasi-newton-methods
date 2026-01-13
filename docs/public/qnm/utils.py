from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Tuple

import numpy as np


@dataclass
class OptimizeResult:
    x: np.ndarray
    fun: float
    grad: np.ndarray
    n_iter: int
    n_fun: int
    n_grad: int
    success: bool
    status: str
    message: str


def ensure_1d(x: np.ndarray | list[float]) -> np.ndarray:
    """Convert input to a 1D float64 NumPy array."""
    return np.asarray(x, dtype=float).reshape(-1)


def finite_difference_grad(fun: Callable[[np.ndarray], float], x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Central finite-difference gradient used for gradient checks."""
    x = ensure_1d(x)
    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        step = np.zeros_like(x)
        step[i] = eps
        grad[i] = (fun(x + step) - fun(x - step)) / (2.0 * eps)
    return grad


def gradient_check(
    fun: Callable[[np.ndarray], float],
    grad: Callable[[np.ndarray], np.ndarray],
    x: np.ndarray,
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> Tuple[bool, np.ndarray, np.ndarray, float]:
    """Compare analytical gradient with finite differences.

    Returns a tuple of (ok, numerical_grad, analytical_grad, diff_norm).
    """
    analytical = grad(x)
    numerical = finite_difference_grad(fun, x, eps=eps)
    diff = analytical - numerical
    diff_norm = np.linalg.norm(diff)
    ok = diff_norm <= atol + rtol * np.linalg.norm(numerical)
    return ok, numerical, analytical, diff_norm


def grad_norm(g: np.ndarray) -> float:
    """Infinity norm of gradient used for stopping conditions."""
    return float(np.linalg.norm(g, ord=np.inf))

