# Theory (Theoretical Concepts)

This project implements quasi-Newton methods based on the formulation in **Nocedal & Wright's "Numerical Optimization" (2006)**.

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

### Newton's Method and Quasi-Newton Methods

Newton's method uses a second-order Taylor expansion to determine the search direction $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$. However, the computational cost of the Hessian matrix $\nabla^2 f(x_k)$ ($O(n^2)$) and its inverse computation ($O(n^3)$) become bottlenecks.

**Quasi-Newton methods** replace the inverse Hessian matrix with an approximate matrix $H_k$ and iteratively update it using only gradient information (first-order derivatives), achieving both high convergence and computational efficiency.

### Strong Wolfe Line Search

To ensure numerical stability and convergence, all algorithms determine the step size $\alpha$ to satisfy the **strong Wolfe conditions**.

1.  **Sufficient Decrease Condition**: The objective function decreases more than expected from the current gradient.
2.  **Curvature Condition**: The gradient after the step is relaxed, maintaining positive definiteness.

This implementation uses Nocedal & Wright's Alg. 3.5 with default values $c_1 = 10^{-4}, c_2 = 0.9$.

## References
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
