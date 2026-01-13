# Implementations

References to external implementations.
Intended for tracing "where this processing is generally implemented" when reading this repository's code.

SciPy's official documentation is summarized in [SciPy](/references/scipy).

Related:
- [Implementation Comparison (BFGS / L-BFGS)](/references/implementation_comparison)

## SciPy
- [optimize.minimize(method='BFGS')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)
- [optimize.minimize(method='L-BFGS-B')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

## L-BFGS-B (Alternative Implementation)
- [bgranzow/L-BFGS-B](https://github.com/bgranzow/L-BFGS-B): Independent implementation of L-BFGS-B (for learning/comparison)

## PyTorch
- [hjmshi/PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS): L-BFGS implementation for PyTorch

## C++ (CppNumericalSolvers / cppoptlib)
- [PatWie/CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers): C++ optimization library (includes BFGS / L-BFGS / L-BFGS-B)
