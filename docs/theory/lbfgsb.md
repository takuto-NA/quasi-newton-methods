# L-BFGS-B (Limited-memory BFGS with Bounds)

L-BFGS-B is an algorithm for solving **box-constrained** optimization problems where variables must remain within certain bounds.

## 1. Problem Definition

We consider minimization under the following constraints:

$$\min_{x \in \mathbb{R}^n} f(x)$$
$$\text{subject to } l_i \le x_i \le u_i, \quad i = 1, \dots, n$$

where $l_i$ is the lower bound and $u_i$ is the upper bound (unbounded cases use $\pm \infty$).

## 2. Key Algorithm Concepts

Unlike unconstrained L-BFGS, simply moving in the search direction may violate constraints (bounds). L-BFGS-B combines the following two steps.

### Projected Gradient

At the current point $x$, determines the direction with the most expected improvement within the constraint-satisfying region. Convergence is determined using the norm of the **projected gradient** ($||g^{pg}||_\infty$), which accounts for components hitting bounds, rather than the usual gradient. This matches the first-order criterion (`pgtol`) in implementations like SciPy and bgranzow (Matlab).

$$x^{pg} = P_{[l,u]}(x - \nabla f(x))$$
$$g^{pg}(x) = x - x^{pg}$$

where $P_{[l,u]}(\cdot)$ is component-wise clipping (projection).

### 1. Generalized Cauchy Point (GCP)

Uses the current L-BFGS approximate quadratic model while exploring along the gradient direction until hitting constraint boundaries (or reaching a local minimum). This process determines "which active constraints (bounds) should be maintained."

### 2. Subspace Minimization

Further optimizes "free variables" (other than boundaries fixed by the Cauchy Point) using L-BFGS information. This maintains the high convergence of quasi-Newton methods even for constrained problems.

## 3. Implementation Form (`qnm.lbfgsb`)

Constrained optimization, especially GCP computation and active set management, is very complex. This project's `qnm.lbfgsb` provides a consistent API while delegating internal computations to **SciPy's `fmin_l_bfgs_b`** (a wrapper around a Fortran implementation).

### Reasons for Delegating to SciPy

1. **Reliability**: The original L-BFGS-B algorithm (Byrd et al., 1995) has a very delicate implementation, and SciPy's implementation has been widely used and verified for many years.
2. **Performance**: Loop processing such as bound checking and subspace minimization is optimized in Fortran.

## 4. Notes on Stopping Conditions

L-BFGS-B terminates when any of the following conditions are met:
- **pgtol**: Maximum component of the projected gradient falls below this value (success).
- **factr**: Objective function improvement rate falls below threshold.
- **maxiter / maxfun**: Maximum iterations or function evaluations reached.

## References
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). "A Limited Memory Algorithm for Bound Constrained Optimization". *SIAM Journal on Scientific and Statistical Computing*.
- Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). "Algorithm 778: L-BFGS-B". *ACM Transactions on Mathematical Software*.
