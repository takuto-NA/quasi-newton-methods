# Baseline Results

Verification results comparing `qnm` implementations with `scipy.optimize.minimize` (BFGS).
Results generated on Jan 6, 2026.

## Verification Table

| Problem | Solver | Success | f(x*) | ‖∇f‖∞ | Iters | Func evals | f_diff (SciPy) | Status |
|---------|--------|---------|-------|-------|-------|------------|----------------|--------|
| rosenbrock (d=2) | BFGS | ✓ | 1.8932e-18 | 5.5e-08 | 34 | 55 | 9.9e-19 | PASSED |
| rosenbrock (d=2) | L-BFGS | ✓ | 1.2445e-19 | 1.2e-08 | 30 | 87 | 7.7e-19 | PASSED |
| rosenbrock (d=10) | BFGS | ✓ | 2.8043e-16 | 4.2e-07 | 81 | 187 | 4.0e+00* | PASSED |
| rosenbrock (d=10) | L-BFGS | ✓ | 9.3388e-15 | 6.7e-07 | 73 | 148 | 4.0e+00* | PASSED |
| quadratic (d=5) | BFGS | ✓ | -2.8161e-01 | 5.4e-09 | 10 | 18 | 3.4e-14 | PASSED |
| quadratic (d=5) | L-BFGS | ✓ | -2.8161e-01 | 1.4e-07 | 12 | 16 | 3.1e-14 | PASSED |
| quadratic (d=50) | BFGS | ✓ | -7.0240e-01 | 3.7e-07 | 59 | 294 | 3.5e-14 | PASSED |
| quadratic (d=50) | L-BFGS | ✓ | -7.0240e-01 | 9.8e-07 | 53 | 63 | 6.8e-13 | PASSED |

\* Note: For Rosenbrock (d=10), SciPy's `minimize(method='BFGS')` converged to a local minimum ($f \approx 3.986$), whereas our implementation reached the global minimum ($f \approx 0$). This discrepancy is due to differences in line search heuristics.

## Fact-Check Summary

1.  **Gradient Check**: All analytical gradients for benchmark problems verified via central finite differences (diff < 1e-6).
2.  **Convergence**: Our implementations match or exceed SciPy's reference performance on standard problems.
3.  **Positive Definiteness**: Curvature condition $s_k^T y_k > 0$ is explicitly checked; the update resets to identity if violated (though not triggered in these benchmarks).
4.  **Source Integrity**: Algorithm steps in `bfgs.py` and `lbfgs.py` mapped to Nocedal & Wright (2006).
