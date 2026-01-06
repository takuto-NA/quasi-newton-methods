import numpy as np

from qnm import gradient_check, rosenbrock_problem


def test_gradcheck_rosenbrock():
    problem = rosenbrock_problem()
    ok, numerical, analytical, diff_norm = gradient_check(problem.fun, problem.grad, problem.x0)
    assert ok, f"Gradient check failed: ||diff||={diff_norm}, analytical={analytical}, numerical={numerical}"

