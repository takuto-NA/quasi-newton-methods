import numpy as np

from qnm.line_search import line_search


def test_line_search_strong_wolfe_conditions_on_quadratic():
    # Simple convex quadratic: f(x)=0.5 x^T A x, grad = A x
    A = np.array([[3.0, 0.0], [0.0, 1.0]])

    def fun(x):
        return 0.5 * float(x.T @ A @ x)

    def grad(x):
        return A @ x

    xk = np.array([1.0, -2.0])
    g0 = grad(xk)
    pk = -g0  # steepest descent direction
    f0 = fun(xk)

    alpha, f_new, g_new, n_fun, n_grad = line_search(fun, grad, xk, pk, f0=f0, g0=g0)

    assert n_fun >= 0 and n_grad >= 0
    assert alpha > 0.0

    c1 = 1e-4
    c2 = 0.9
    derphi0 = float(np.dot(g0, pk))
    derphi = float(np.dot(g_new, pk))

    # Strong Wolfe (sufficient decrease)
    assert f_new <= f0 + c1 * alpha * derphi0 + 1e-12
    # Strong Wolfe (curvature)
    assert abs(derphi) <= c2 * abs(derphi0) + 1e-12


def test_line_search_returns_zero_for_non_descent_direction():
    def fun(x):
        return float(np.sum(x**2))

    def grad(x):
        return 2.0 * x

    xk = np.array([1.0, 1.0])
    g0 = grad(xk)
    pk = g0  # ascent direction => not descent
    f0 = fun(xk)

    alpha, f_new, g_new, n_fun, n_grad = line_search(fun, grad, xk, pk, f0=f0, g0=g0)

    assert alpha == 0.0
    assert f_new == f0
    assert np.allclose(g_new, g0)
    assert n_fun == 0
    assert n_grad == 0


