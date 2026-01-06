import numpy as np
import pytest

from qnm import lbfgsb

scipy = pytest.importorskip("scipy")


def test_lbfgsb_with_bounds():
    fun = lambda x: (x[0] - 1.0) ** 2
    grad = lambda x: np.array([2.0 * (x[0] - 1.0)])

    result = lbfgsb(fun, grad, np.array([3.0]), bounds=[(0.0, 2.0)], tol=1e-9)
    assert result.success
    assert np.allclose(result.x, [1.0], atol=1e-6)
