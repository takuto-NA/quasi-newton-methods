import numpy as np

from qnm import bfgs, lbfgs, quadratic_problem


def _run_solver(solver, problem):
    result = solver(problem.fun, problem.grad, problem.x0, tol=1e-8, max_iter=200)
    assert result.success, f"{solver.__name__} failed: {result.status} ({result.message})"
    assert np.allclose(result.x, problem.solution, atol=1e-6)
    assert np.linalg.norm(problem.grad(result.x)) < 1e-6


def test_quadratic_bfgs_and_lbfgs():
    problem = quadratic_problem(dim=5, condition_number=5.0, seed=1)
    _run_solver(bfgs, problem)
    _run_solver(lbfgs, problem)

