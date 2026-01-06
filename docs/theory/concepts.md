# Theoretical Concepts

This project implements Quasi-Newton methods based on the mathematical foundations laid out in **Nocedal & Wright, "Numerical Optimization" (2006)**.

## Newton vs. Quasi-Newton

Newton's method uses the Second-Order Taylor expansion:
$$f(x_k + p) \approx f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2 f(x_k) p$$

The search direction is $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$. However, calculating and inverting the Hessian $\nabla^2 f(x_k)$ is computationally expensive ($O(n^3)$) or sometimes impossible.

**Quasi-Newton methods** replace the exact Hessian with an approximation $B_k$ (or its inverse $H_k$) that is updated using only gradient information.

## BFGS (Broyden-Fletcher-Goldfarb-Shanno)

BFGS is the most popular Quasi-Newton method. It maintains an approximation of the *inverse* Hessian $H_k$.

### The Secant Equation
The update must satisfy the secant equation:
$$H_{k+1} y_k = s_k$$
where $s_k = x_{k+1} - x_k$ and $y_k = \nabla f_{k+1} - \nabla f_k$.

### The Update Formula (Eq. 6.17, p. 140)
$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$
where $\rho_k = 1 / (y_k^T s_k)$.

### Curvature Condition
For $H_{k+1}$ to remain positive definite, we require the curvature condition:
$$s_k^T y_k > 0$$
This is guaranteed if the line search satisfies the **Wolfe conditions**.

## L-BFGS (Limited-memory BFGS)

L-BFGS is designed for large-scale problems where storing the full $n \times n$ Hessian is impractical. Instead of storing $H_k$, it stores the last $m$ pairs of $\{s_i, y_i\}$.

### Two-Loop Recursion (Alg. 7.4, p. 178)
The search direction $p_k = -H_k \nabla f_k$ is computed efficiently without explicitly forming $H_k$.

1.  Compute $q$ from current gradient.
2.  Backward pass: Update $q$ using stored $s_i, y_i$.
3.  Apply initial scaling (Eq. 7.20).
4.  Forward pass: Compute $r$ (result).

## Strong Wolfe Line Search

To ensure stability and convergence, the step size $\alpha$ must satisfy the **Strong Wolfe conditions** (Eq. 3.7, p. 34):

1.  **Sufficient Decrease**: $f(x_k + \alpha p_k) \le f(x_k) + c_1 \alpha \nabla f_k^T p_k$
2.  **Curvature Condition**: $|\nabla f(x_k + \alpha p_k)^T p_k| \le c_2 |\nabla f_k^T p_k|$

We use $c_1 = 10^{-4}$ and $c_2 = 0.9$ as recommended by Nocedal & Wright.

## References
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer Science & Business Media.
- Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization". *Mathematical Programming*.

