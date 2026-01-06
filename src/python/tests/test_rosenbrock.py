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

