import numpy as np

from qnm import bfgs, lbfgs, rosenbrock_problem


def _solve(solver, problem):
    result = solver(problem.fun, problem.grad, problem.x0, tol=1e-5, max_iter=400)
    assert result.success, f"{solver.__name__} failed: {result.status} ({result.message})"
    assert np.allclose(result.x, problem.solution, atol=1e-3)
    assert np.linalg.norm(result.grad, ord=np.inf) < 1e-3


def test_rosenbrock_convergence():
    problem = rosenbrock_problem(dim=2)
    _solve(bfgs, problem)
    _solve(lbfgs, problem)


def test_solvers_monotone_decrease_on_rosenbrock_dim2():
    # Strong Wolfe sufficient decrease should make objective non-increasing.
    problem = rosenbrock_problem(dim=2)

    for solver in (bfgs, lbfgs):
        history = []

        def cb(res):
            history.append(float(res.fun))

        result = solver(problem.fun, problem.grad, problem.x0, tol=1e-5, max_iter=200, callback=cb)
        assert result.success
        assert len(history) > 0
        for a, b in zip(history, history[1:]):
            assert b <= a + 1e-12

