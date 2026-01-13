# BFGS (Broyden-Fletcher-Goldfarb-Shanno)

BFGS is one of the most widely used algorithms among quasi-Newton methods. While Newton's method directly computes the Hessian matrix $\nabla^2 f(x)$ and finds its inverse, BFGS efficiently updates an approximation $H_k$ of the inverse Hessian matrix using gradient information.

## 1. Basic Principles

The update step in Newton's method is $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$, but BFGS approximates this as $p_k = -H_k \nabla f(x_k)$.

### Secant Equation

The new approximate matrix $H_{k+1}$ must satisfy the following **secant condition** to correctly reflect recent gradient changes.

$$H_{k+1} y_k = s_k$$

where,
- $s_k = x_{k+1} - x_k$ (displacement)
- $y_k = \nabla f_{k+1} - \nabla f_k$ (gradient change)

### Update Formula

Based on Eq. 6.17 of Nocedal & Wright (2006), the update formula for the inverse Hessian approximation $H$ is as follows.

$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$

where $\rho_k = \frac{1}{y_k^T s_k}$. This formula ensures that if $H_k$ is positive definite, the updated $H_{k+1}$ also maintains positive definiteness.

## 2. Curvature Condition

For $H_{k+1}$ to be positive definite, the denominator $\rho_k$ must be well-defined, i.e., the following **curvature condition** must be satisfied.

$$s_k^T y_k > 0$$

In practice, this condition is guaranteed by imposing **strong Wolfe conditions** in line search.

## 3. Algorithm Flow

The procedure implemented in `qnm.bfgs` is as follows.

```mermaid
flowchart TD
    Start(["Start"]) --> Init["Initialize: x_0, H_0 = I"]
    Init --> Loop{"Convergence check: ||grad|| < tol?"}
    Loop -- Yes --> End(["End"])
    Loop -- No --> SearchDir["Determine search direction: p_k = -H_k grad_k"]
    SearchDir --> LineSearch["Line search: determine alpha satisfying strong Wolfe conditions"]
    LineSearch --> UpdateX["Update variables: x_k+1 = x_k + alpha * p_k"]
    UpdateX --> CalcGrad["Compute new gradient grad_k+1"]
    CalcGrad --> CheckCurvature{"Curvature condition s^T y > 1e-12?"}
    CheckCurvature -- Yes --> UpdateH["Compute H_k+1 using BFGS update formula"]
    CheckCurvature -- No --> ResetH["Reset H_k+1 = I"]
    UpdateH --> Loop
    ResetH --> Loop
```

## 4. Implementation Points (`qnm.bfgs`)

- **Maintaining Positive Definiteness**: When curvature is lost ($s_k^T y_k \le 10^{-12}$), the approximate matrix is reset to the identity matrix for numerical stability. This is a common safeguard adopted in major implementations such as SciPy and CppNumericalSolvers.
- **Computational Efficiency**: The update formula consists of outer products (`np.outer`) and matrix products, avoiding matrix inversion ($O(n^3)$) and enabling updates in $O(n^2)$ complexity.
- **Initial Approximation**: The initial value $H_0$ is set to the identity matrix $I$, consistent with designs like SciPy BFGS.

## 5. Interactive Demo

You can run the actual implementation (`qnm.bfgs`) in your browser and observe how the approximate inverse Hessian matrix $H$ is updated.

<ClientOnly>
  <OptimizerVisualizer algorithm="bfgs" problemType="rosenbrock" :dim="2" />
</ClientOnly>

## 6. References
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer. (Chapter 6)
