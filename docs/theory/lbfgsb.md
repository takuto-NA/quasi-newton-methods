# L-BFGS-B (Limited-memory BFGS with Bounds)

L-BFGS-B is an algorithm for solving **box-constrained** optimization problems where variables must remain within certain bounds.

## Reader Guide

### After reading this page, you should be able to

- State the constrained problem $\min f(x)\ \text{s.t. } l \le x \le u$ and what makes it different from unconstrained L-BFGS.
- Explain what the **projected gradient** measures and why it is used as a stopping criterion.
- Describe (at a high level) why L-BFGS-B uses **(1) a generalized Cauchy point** and **(2) subspace minimization**.
- Map common stopping parameters (`pgtol`, `factr`, `maxiter`, `maxfun`) to practical meanings.

### Prerequisites

- Unconstrained L-BFGS basics: **[`lbfgs.md`](lbfgs.md)**
- Shared invariants/line search context: **[`concepts.md`](concepts.md)**

## 1. Problem Definition

We consider minimization under the following constraints:

$$\min_{x \in \mathbb{R}^n} f(x)$$
$$\text{subject to } l_i \le x_i \le u_i, \quad i = 1, \dots, n$$

where $l_i$ is the lower bound and $u_i$ is the upper bound (unbounded cases use $\pm \infty$).

## 2. Key Algorithm Concepts

Unlike unconstrained L-BFGS, simply moving in the search direction may violate constraints (bounds). L-BFGS-B combines the following two steps.

### Projected Gradient

At the current point $x$, determines the direction with the most expected improvement within the constraint-satisfying region. Convergence is determined using the norm of the **projected gradient** ($\|g^{pg}\|_\infty$), which accounts for components hitting bounds, rather than the usual gradient. This matches the first-order criterion (`pgtol`) in implementations like SciPy and bgranzow (Matlab).

$$x^{pg} = P_{[l,u]}(x - \nabla f(x))$$
$$g^{pg}(x) = x - x^{pg}$$

where $P_{[l,u]}(\cdot)$ is component-wise clipping (projection).
Here and below, $g=\nabla f(x)$ denotes the (raw) gradient at the current point.

#### Interpretation (Why this makes sense)

At an optimum with bounds, not every component can move freely. A variable sitting on a bound can have a nonzero gradient component without violating optimality, as long as the gradient points “out of the feasible region.”

You can view the projected-gradient test as a compact way to check the bound-constrained first-order (KKT) condition:

- If $x_i$ is **strictly inside** the bounds, then we want $g_i \approx 0$.
- If $x_i = l_i$ (lower bound), then we require $g_i \ge 0$ (if $g_i < 0$, decreasing $x_i$ would reduce $f$, but that move is infeasible).
- If $x_i = u_i$ (upper bound), then we require $g_i \le 0$ (if $g_i > 0$, increasing $x_i$ would reduce $f$, but that move is infeasible).

This matches the projected-gradient definition on this page:

$$g^{pg}(x) = x - P_{[l,u]}(x - g)$$

Component-wise, this implies:

- If $l_i < x_i < u_i$, then $g^{pg}_i = g_i$.
- If $x_i = l_i$, then $g^{pg}_i = \min(g_i, 0)$.
- If $x_i = u_i$, then $g^{pg}_i = \max(g_i, 0)$.

So $\|g^{pg}\|_\infty$ becomes small exactly when there is no meaningful first-order improvement left *within the box*.

### 1. Generalized Cauchy Point (GCP)

Uses the current L-BFGS approximate quadratic model while exploring along the gradient direction until hitting constraint boundaries (or reaching a local minimum). This process determines "which active constraints (bounds) should be maintained."

### 2. Subspace Minimization

Further optimizes "free variables" (other than boundaries fixed by the Cauchy Point) using L-BFGS information. This maintains the high convergence of quasi-Newton methods even for constrained problems.

### Active set intuition (informal)

During the iteration, some variables are treated as **active** (fixed at a bound) and the rest are **free** (allowed to move). The generalized Cauchy point is a principled way to pick a plausible active set given the current model, and subspace minimization then does a quasi-Newton step restricted to the free variables.

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

### Practical notes

- **Always provide a feasible starting point**: if $x_0$ is outside bounds, most implementations project it internally, but you should not rely on this silently.
- **Unbounded components**: represent “no bound” as $\pm \infty$ (implementation-specific API usually accepts `None`/`np.inf`).
- **Don’t confuse convergence tests**: for bound constraints, $\|\nabla f\|$ can remain nonzero at the solution; the projected gradient is the meaningful first-order measure.

## 5. Self-check Questions (Quick)

1. Why can $\nabla f(x^\star)$ be nonzero at a bound-constrained optimum?
2. What does it mean if the projected gradient is small but the raw gradient is not?
3. Conceptually, what roles do the generalized Cauchy point and subspace minimization play?

## FAQ

### Q: Why isn’t “just clip to the bounds” enough?

A: Clipping preserves feasibility, but it does not address optimality (KKT conditions) or convergence behavior. L-BFGS-B combines projected gradients, active-set logic, and subspace minimization so it can make efficient progress **while properly handling variables stuck on bounds** (see “Projected Gradient” and “Active set intuition”).

### Q: Is it a bug if the projected gradient is small but $\|\nabla f\|$ is large?

A: Not necessarily. At a bound-constrained optimum, components on bounds can have nonzero raw gradients in directions that are infeasible to move. What matters is whether there is remaining first-order improvement **within the box**, and that is what the projected gradient measures (see “Interpretation”).

### Q: How should I think about `pgtol` vs `factr`?

A: `pgtol` is closer to a first-order (KKT-like) optimality measure, while `factr` is a progress/relative-improvement threshold on the objective. In practice, it’s often simplest to primarily rely on `pgtol`, and use `factr` as an additional “diminishing returns” stop (see “Notes on Stopping Conditions”).

## References
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). "A Limited Memory Algorithm for Bound Constrained Optimization". *SIAM Journal on Scientific and Statistical Computing*.
- Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). "Algorithm 778: L-BFGS-B". *ACM Transactions on Mathematical Software*.
