from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from qnm.bfgs import bfgs
from qnm.lbfgs import lbfgs
from qnm.problems import rosenbrock_problem, quadratic_problem
from qnm.utils import gradient_check, grad_norm

def verify_solver(solver_name, solver_func, problem, tol=1e-6):
    print(f"--- Verifying {solver_name} on {problem.name} ---")
    
    # 1. Gradient Check
    ok, num_g, ana_g, diff = gradient_check(problem.fun, problem.grad, problem.x0)
    print(f"Gradient Check: {'PASSED' if ok else 'FAILED'} (diff={diff:.2e})")
    
    # 2. Solver Execution
    res = solver_func(problem.fun, problem.grad, problem.x0, tol=tol)
    
    # 3. SciPy Ground Truth
    # Note: SciPy's BFGS doesn't use the exact same line search, so we compare final f values.
    scipy_res = minimize(problem.fun, problem.x0, method='BFGS', jac=problem.grad, tol=tol)
    
    f_diff = abs(res.fun - scipy_res.fun)
    x_diff = np.linalg.norm(res.x - scipy_res.x)
    
    print(f"Success: {res.success}")
    print(f"Iterations: {res.n_iter} (SciPy: {scipy_res.nit})")
    print(f"Function Evals: {res.n_fun} (SciPy: {scipy_res.nfev})")
    print(f"Final f: {res.fun:.6e} (SciPy: {scipy_res.fun:.6e})")
    print(f"f difference: {f_diff:.2e}")
    
    status = "PASSED" if res.success and f_diff < 1e-4 else "FAILED"
    print(f"Verification Status: {status}")
    print()
    
    return {
        "problem": problem.name,
        "solver": solver_name,
        "success": res.success,
        "f": res.fun,
        "gnorm": grad_norm(res.grad),
        "iters": res.n_iter,
        "fevals": res.n_fun,
        "f_diff": f_diff,
        "status": status
    }

def main():
    problems = [
        rosenbrock_problem(dim=2),
        rosenbrock_problem(dim=10),
        quadratic_problem(dim=5, condition_number=10),
        quadratic_problem(dim=50, condition_number=100)
    ]
    
    results = []
    for prob in problems:
        results.append(verify_solver("BFGS", bfgs, prob))
        results.append(verify_solver("L-BFGS", lbfgs, prob))
    
    # Generate Markdown Table for docs/evidence/baseline_results.md
    print("\n### Generated Baseline Results Table\n")
    print("| Problem | Solver | Success | f(x*) | ‖∇f‖∞ | Iters | Func evals | f_diff (SciPy) | Status |")
    print("|---------|--------|---------|-------|-------|-------|------------|----------------|--------|")
    for r in results:
        success_icon = "✓" if r["success"] else "✗"
        print(f"| {r['problem']} | {r['solver']} | {success_icon} | {r['f']:.4e} | {r['gnorm']:.1e} | {r['iters']} | {r['fevals']} | {r['f_diff']:.1e} | {r['status']} |")

if __name__ == "__main__":
    main()

