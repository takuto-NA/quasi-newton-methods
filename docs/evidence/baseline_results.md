# Baseline Verification Results

Baseline verification results for `qnm`.

- Generated: 2026-01-06
- Generation procedure: `PYTHONPATH=src/python python src/python/scripts/verify_implementation.py`

For definitions of pass/fail (`PASSED` / `ACCEPTABLE_DIFF` / `FAILED`) and treatment of primary/secondary comparisons, see [methodology](/evidence/methodology).

## Reproducibility (Environment)

The numerical values on this page were generated in the following environment.

- Python: 3.12.10
- Platform: Windows-11-10.0.26200-SP0
- NumPy: 2.2.6
- SciPy: 1.14.1

## How to Read the Table

- `‖∇f‖∞`: Infinity norm of gradient (used for stopping condition)
- `Iters`: Number of iterations (`n_iter`)
- `Func evals`: Number of objective function evaluations (`n_fun`)
- `f_diff (SciPy)`: Difference in final objective function value from SciPy reference implementation (**auxiliary metric**)

Regarding `f_diff (SciPy)`:

- **BFGS**: Difference from SciPy `minimize(method='BFGS')` (reference comparison)
- **L-BFGS**: Difference from SciPy `minimize(method='L-BFGS-B')` (without bounds) (**informational**. Not a primary pass/fail criterion)

Guidelines (intuitive interpretation):

- If `f_diff <= 1e-6`, it can be considered to have converged to approximately the same solution
- If `f_diff >= 1e-0` (large), it may have converged to a different local solution (see `Status` column and notes)

## Verification Table

| Problem | Solver | Success | f(x*) | ‖∇f‖∞ | Iters | Func evals | f_diff (SciPy) | Status |
|---------|--------|---------|-------|-------|-------|------------|----------------|--------|
| rosenbrock (d=2) | BFGS | ✓ | 1.8932e-18 | 5.5e-08 | 34 | 55 | 9.9e-19 | PASSED |
| rosenbrock (d=2) | L-BFGS | ✓ | 1.2445e-19 | 1.2e-08 | 30 | 87 | 5.3e-10 | PASSED |
| rosenbrock (d=10) | BFGS | ✓ | 2.8043e-16 | 4.2e-07 | 81 | 187 | 4.0e+00* | ACCEPTABLE_DIFF |
| rosenbrock (d=10) | L-BFGS | ✓ | 9.3388e-15 | 6.7e-07 | 73 | 148 | 8.2e-08 | PASSED |
| quadratic (d=5) | BFGS | ✓ | -2.8161e-01 | 5.4e-09 | 10 | 18 | 3.4e-14 | PASSED |
| quadratic (d=5) | L-BFGS | ✓ | -2.8161e-01 | 1.4e-07 | 12 | 16 | 2.3e-09 | PASSED |
| quadratic (d=50) | BFGS | ✓ | -7.0240e-01 | 3.7e-07 | 59 | 294 | 3.5e-14 | PASSED |
| quadratic (d=50) | L-BFGS | ✓ | -7.0240e-01 | 9.8e-07 | 53 | 63 | 8.8e-07 | PASSED |

\* Note (BFGS only): For Rosenbrock (d=10), SciPy's `minimize(method='BFGS')` converged to a local minimum ($f \approx 3.986$), whereas our implementation reached the global minimum ($f \approx 0$). This discrepancy is due to differences in line search heuristics.

## Notes

- This table verifies **`qnm` (core implementation: BFGS / L-BFGS)**.
- **L-BFGS-B** is excluded from "correctness of self-implementation" here because `qnm.lbfgsb` delegates to SciPy's reference implementation.

## Fact-Check Summary

1. **Gradient Check**: Verified analytical gradients of benchmark problems using central differences (difference < 1e-6).
2. **Convergence**: Confirmed reasonable optimization results (final value, gradient norm) on representative problems.
3. **Positive Definiteness**: Explicitly checks curvature condition $s_k^T y_k > 0$, resets update on violation (not triggered in this benchmark).
4. **Source Integrity**: Maps major steps in `bfgs.py` / `lbfgs.py` to Nocedal & Wright (2006).
