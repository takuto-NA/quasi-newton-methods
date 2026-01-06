# quasi-newton-methods

Evidence-first implementations and benchmarks for quasi-Newton methods (BFGS, L-BFGS, L-BFGS-B).

This project provides clear, pedagogical implementations of optimization algorithms, cross-referenced with standard literature and verified against industry-standard libraries (SciPy).

## Why This Project?
- **Algorithmic Transparency**: Code is mapped line-by-line to Algorithm 6.1 (BFGS) and Algorithm 7.4 (L-BFGS) in Nocedal & Wright's *Numerical Optimization*.
- **Evidence-First**: Every implementation is verified by an automated fact-checking suite that compares results against SciPy.
- **Implementer-Focused**: Designed for those who want to understand *how* these methods work under the hood.

## Verification Status (Jan 6, 2026)
| Method | Verification | Status |
|--------|--------------|--------|
| **BFGS** | Nocedal & Wright (2006) Alg 6.1 | [PASSED](./docs/evidence/baseline_results.md) |
| **L-BFGS** | Nocedal & Wright (2006) Alg 7.4 | [PASSED](./docs/evidence/baseline_results.md) |
| **Line Search** | Strong Wolfe (Alg 3.5/3.6) | [PASSED](./docs/evidence/baseline_results.md) |

## Quick Start

### 1. Installation
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1 | macOS/Linux: source .venv/bin/activate
pip install -e ./src/python[dev]
```

### 2. Verify Implementation
Run the automated fact-checker to compare with SciPy:
```bash
export PYTHONPATH=src/python
python src/python/scripts/verify_implementation.py
```

### 3. Usage Example
```python
from qnm import bfgs, rosenbrock_problem

problem = rosenbrock_problem(dim=2)
result = bfgs(problem.fun, problem.grad, problem.x0)

print(f"Optimal x: {result.x}")
print(f"Objective value: {result.fun}")
```

## Documentation structure
- [**Theoretical Concepts**](./docs/theory/concepts.md): Mathematical derivation and mapping to code.
- [**Evidence & Benchmarks**](./docs/evidence/baseline_results.md): Detailed verification logs.
- [**References**](./docs/references/papers.md): Key papers and textbooks.

## Project Layout
- `src/python/qnm/`: Core implementations (BFGS, L-BFGS, Line Search).
- `src/python/scripts/`: Verification and benchmarking scripts.
- `docs/`: Narrative documentation and evidence.

## License
MIT License
