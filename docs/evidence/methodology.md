# Methodology

This project's Evidence does not aim for "exact match with reference implementations" but rather confirms that **(1) theoretical properties** and **(2) differences from reference implementations (primarily SciPy) are explainable**.

## 1. Common Settings

- Initial values: According to problem definitions (`qnm.problems`)
- Stopping condition: `‖∇f(x)‖∞ <= tol`
- Iteration limit: solver's `max_iter`
- Line search: Strong Wolfe (`qnm.line_search`, default `c1=1e-4, c2=0.9`)

## 2. Metrics Recorded

- Iterations (`n_iter`)
- Function evaluations (`n_fun`)
- Gradient evaluations (`n_grad`)
- Final objective function value `f(x*)`
- Gradient infinity norm `‖∇f‖∞`
- Reference: Difference from SciPy (however, **not used as a primary criterion** for the reasons below)

## 3. Comparison Strategy (SciPy as primary reference, others used as needed)

- **BFGS**: Compare primarily with SciPy `minimize(method='BFGS')`. However, since line search heuristics differ between SciPy and `qnm`, final values/solutions may differ. In such cases, judge validity using **proximity to known solutions** and **gradient norm** as primary criteria.
- **L-BFGS**: SciPy's `minimize(method='L-BFGS-B')` exists, but it is not the same implementation as `qnm.lbfgs` (bound-free L-BFGS). Therefore, use a two-tier approach for L-BFGS:
  - Primary: Property tests (descent direction, Wolfe satisfaction, convergence to known solutions)
  - Secondary: Reference comparison (e.g., final value from running SciPy L-BFGS-B without bounds)
- **L-BFGS-B**: `qnm.lbfgsb` is a wrapper that delegates to SciPy's reference implementation (`scipy.optimize.fmin_l_bfgs_b`), so it is excluded from primary verification of "correctness of self-implementation" in Evidence (depends on SciPy's correctness).

## 4. Evidence Status (Pass/Fail Definition)

Status indicates "what this verification can claim."

- `PASSED`: Solver terminates with `success=True` and satisfies primary criteria (e.g., proximity to known solutions/gradient norm/property tests).
- `ACCEPTABLE_DIFF`: Solver itself satisfies primary criteria but results differ significantly from reference implementation (e.g., SciPy). The cause of the difference (e.g., local solution, line search differences) can be explained in footnotes.
- `FAILED`: `success=False`, or does not satisfy primary criteria.

## 5. Reproducibility

- Wall-clock time is highly environment-dependent, so it is used as an auxiliary metric.
- Execution environment (Python/SciPy/NumPy versions) is output and included in Evidence.
