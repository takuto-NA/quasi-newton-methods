# quasi-newton-methods

A project organizing implementations of quasi-Newton methods (BFGS / L-BFGS / L-BFGS-B) along with their theoretical foundations and verification results.

## For First-Time Visitors

This project emphasizes not just providing implementations, but also the **evidence** for "whether it works correctly" and "why this implementation approach was chosen."

- **[Theory](./theory/concepts.md)**: Correspondence between Nocedal & Wright algorithms and code.
- **[Evidence](./evidence/baseline_results.md)**: Comparison results with SciPy and gradient check results.
- **[References](./references/papers.md)**: List of foundational papers and textbooks.

## Quick Start

```bash
# (Recommended) Installation
# Requires Python >= 3.10.
# NOTE: PowerShell needs quotes because `[dev]` may be treated as a wildcard pattern.
pip install -e ./src/python[dev]  # PowerShell: pip install -e ".\src\python[dev]"

# Implementation Verification (Fact-check)
python src/python/scripts/verify_implementation.py
```

If running without editable installation, set `PYTHONPATH`:

```bash
# macOS/Linux (bash/zsh) / Git Bash
export PYTHONPATH=src/python
python src/python/scripts/verify_implementation.py
```

```bash
# Windows PowerShell
$env:PYTHONPATH = "src/python"
python src/python/scripts/verify_implementation.py
```

```bash
# Windows cmd.exe
set PYTHONPATH=src/python
python src/python/scripts/verify_implementation.py
```

## Implemented Methods

1.  **[BFGS](./theory/bfgs.md)**: Standard method that directly updates the inverse Hessian matrix.
2.  **[L-BFGS](./theory/lbfgs.md)**: Memory-efficient quasi-Newton method (Two-loop recursion).
3.  **Line Search**: Step size determination algorithm satisfying strong Wolfe conditions.
4.  **[L-BFGS-B](./theory/lbfgsb.md)**: Bound-constrained version (SciPy wrapper).
