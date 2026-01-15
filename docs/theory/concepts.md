# Theory (Theoretical Concepts)

This project implements quasi-Newton methods based on the formulation in **Nocedal & Wright's "Numerical Optimization" (2006)**.

## Reader Guide (What to Understand First)

If you're reading these notes to *implement* or *debug* quasi-Newton methods, make sure you can answer the following after finishing `concepts.md`.

### After reading this page, you should be able to

- Explain why quasi-Newton methods are used instead of Newton’s method (cost of Hessian/inverse).
- Define the core objects and symbols used throughout (`f`, `g`, `s`, `y`, `H_k`, line search).
- State what **strong Wolfe line search** guarantees and why it matters for stability.
- List the most important invariants you should monitor in code (descent direction and curvature).

### Minimal prerequisites (don’t overthink)

- You know what a gradient is and why \(p^\top g < 0\) implies a descent direction.
- You are comfortable with basic linear algebra (inner products, positive definiteness).

## Core Algorithms

For detailed theory and implementation points for each method, refer to the following individual articles.

- **[BFGS Method (Standard BFGS)](bfgs.md)**
  - Standard method that directly updates the inverse Hessian matrix. Suitable for small to medium-scale problems.
- **[L-BFGS Method (Limited-memory BFGS)](lbfgs.md)**
  - Memory-efficient method that only maintains update history. Essential for large-scale problems.
- **[L-BFGS-B Method (Bound-constrained)](lbfgsb.md)**
  - Extension for cases with bound constraints on variables. Uses concepts of projected gradient and active sets.

---

## Common Foundation Techniques

Important components common to all algorithms.

### Notation Cheat Sheet (Used Everywhere)

- \(f(x)\): objective function to minimize
- \(g_k = \nabla f(x_k)\): gradient at iterate \(x_k\)
- \(x_{k+1} = x_k + \alpha_k p_k\): iterate update (step length \(\alpha_k\), direction \(p_k\))
- \(s_k = x_{k+1} - x_k\): step (displacement)
- \(y_k = g_{k+1} - g_k\): gradient change
- \(B_k \approx \nabla^2 f(x_k)\): Hessian approximation (sometimes used)
- \(H_k \approx B_k^{-1}\): inverse-Hessian approximation (used by these docs and `qnm`)

### Newton's Method and Quasi-Newton Methods

Newton's method uses a second-order Taylor expansion to determine the search direction $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$. However, the computational cost of the Hessian matrix $\nabla^2 f(x_k)$ ($O(n^2)$) and its inverse computation ($O(n^3)$) become bottlenecks.

**Quasi-Newton methods** replace the inverse Hessian matrix with an approximate matrix $H_k$ and iteratively update it using only gradient information (first-order derivatives), achieving both high convergence and computational efficiency.

### Strong Wolfe Line Search

To ensure numerical stability and convergence, all algorithms determine the step size $\alpha$ to satisfy the **strong Wolfe conditions**.

1.  **Sufficient Decrease Condition**: The objective function decreases more than expected from the current gradient.
2.  **Curvature Condition**: The gradient after the step is relaxed, maintaining positive definiteness.

This implementation uses Nocedal & Wright's Alg. 3.5 with default values $c_1 = 10^{-4}, c_2 = 0.9$.

### Practical Invariants (Debug Checklist)

These are the “always check these first” items when an optimizer behaves strangely:

- **Descent direction**: verify \(p_k^\top g_k < 0\) (if not, something is wrong with \(H_k\), scaling, or numerical stability).
- **Curvature**: verify \(s_k^\top y_k > 0\) before applying BFGS/L-BFGS updates.
  - Strong Wolfe line search is commonly used because it *tends to enforce* this curvature condition in practice.
- **Step sanity**: if \(\alpha_k\) collapses to extremely small values repeatedly, suspect scaling issues, noisy gradients, or a bug in line search.

### Where to Go Next (Reading Order)

- Start with **[`bfgs.md`](bfgs.md)** to learn the full-memory update and the key invariants.
- Then read **[`lbfgs.md`](lbfgs.md)** to see how BFGS is implemented efficiently via two-loop recursion.
- If you have bounds, read **[`lbfgsb.md`](lbfgsb.md)** for projected gradients and stopping conditions (and why implementations are delicate).

## References
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
