from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .utils import ensure_1d


@dataclass
class Problem:
    name: str
    fun: Callable[[np.ndarray], float]
    grad: Callable[[np.ndarray], np.ndarray]
    x0: np.ndarray
    solution: np.ndarray | None = None


def quadratic_problem(dim: int = 2, condition_number: float = 10.0, seed: int | None = 0) -> Problem:
    rng = np.random.default_rng(seed)
    # Create a symmetric positive definite matrix with a modest condition number.
    Q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    eigenvalues = np.linspace(1.0, condition_number, dim)
    A = Q @ np.diag(eigenvalues) @ Q.T
    b = rng.normal(size=dim)

    def fun(x: np.ndarray) -> float:
        x = ensure_1d(x)
        return 0.5 * float(x.T @ A @ x) - float(b.T @ x)

    def grad(x: np.ndarray) -> np.ndarray:
        x = ensure_1d(x)
        return A @ x - b

    x_star = np.linalg.solve(A, b)
    x0 = rng.normal(size=dim)
    return Problem(name="quadratic", fun=fun, grad=grad, x0=x0, solution=x_star)


def rosenbrock_problem(dim: int = 2, a: float = 1.0, b: float = 100.0) -> Problem:
    if dim < 2:
        raise ValueError("Rosenbrock problem requires dim >= 2")

    def fun(x: np.ndarray) -> float:
        x = ensure_1d(x)
        xi = x[:-1]
        xnext = x[1:]
        return float(np.sum(b * (xnext - xi**2) ** 2 + (a - xi) ** 2))

    def grad(x: np.ndarray) -> np.ndarray:
        x = ensure_1d(x)
        g = np.zeros_like(x)
        g[:-1] = -4 * b * x[:-1] * (x[1:] - x[:-1] ** 2) - 2 * (a - x[:-1])
        g[1:] += 2 * b * (x[1:] - x[:-1] ** 2)
        return g

    x0 = np.full(dim, -1.2)
    x0[::2] = -1.2
    x0[1::2] = 1.0
    x_star = np.full(dim, a)
    return Problem(name="rosenbrock", fun=fun, grad=grad, x0=x0, solution=x_star)

