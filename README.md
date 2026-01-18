# quasi-newton-methods

An **Evidence-first** implementation and verification repository for quasi-Newton methods (BFGS / L-BFGS / L-BFGS-B).

This repository aims to clearly map standard textbook algorithms (Nocedal & Wright) to code, and to explain "why this implementation is correct" through comparisons with reference implementations (primarily SciPy).

**Project Website / Documentation**: [takuto-na.github.io/quasi-newton-methods](https://takuto-na.github.io/quasi-newton-methods/)

Note:

- **L-BFGS-B** is provided as `qnm.lbfgsb`, but this is a **wrapper that delegates to SciPy's reference implementation** (separate from core implementation verification).

## For First-Time Visitors (Where to Start)

This project emphasizes not just providing implementations, but also the evidence for "whether it works correctly" and "why this implementation approach was chosen."

- [**Theory**](./docs/theory/concepts.md): Correspondence between Nocedal & Wright algorithms and code (includes diagrams of Two-loop recursion).
- [**Evidence**](./docs/evidence/baseline_results.md): Comparison results with SciPy and the criteria for pass/fail.
- [**References**](./docs/references/papers.md): Primary sources such as papers and textbooks.

## Project Philosophy
- **Algorithm Transparency**: Maps Nocedal & Wright algorithms (BFGS: Alg 6.1 / L-BFGS: Alg 7.4) to code.
- **Evidence-first**: Emphasizes theoretical properties and explainability of differences rather than "exact match with reference implementations."
- **For Implementers**: Designed for the shortest path to "implementing" methods, not just "using" them.

## Verification Status (Jan 6, 2026)
| Method | Verification | Status |
|--------|--------------|--------|
| **BFGS** | Nocedal & Wright (2006) Alg 6.1 | [PASSED](./docs/evidence/baseline_results.md) |
| **L-BFGS** | Nocedal & Wright (2006) Alg 7.4 | [PASSED](./docs/evidence/baseline_results.md) |
| **Line Search** | Strong Wolfe (Alg 3.5/3.6) | [PASSED](./docs/evidence/baseline_results.md) |

## Quick Start

### 1. Installation

Requires Python >= 3.10.
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1 | macOS/Linux: source .venv/bin/activate
# NOTE: PowerShell needs quotes because `[dev]` may be treated as a wildcard pattern.
pip install -e ./src/python[dev]  # PowerShell: pip install -e ".\src\python[dev]"
```

### 2. Implementation Verification (Fact-check)

Run the automated verification script to compare with SciPy and other implementations.

```bash
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

### 3. Usage Example

```python
from qnm import bfgs, rosenbrock_problem

problem = rosenbrock_problem(dim=2)
result = bfgs(problem.fun, problem.grad, problem.x0)

print(f"Optimal x: {result.x}")
print(f"Objective value: {result.fun}")
```

## Documentation Structure
- [**Theory**](./docs/theory/concepts.md): Mathematical background and correspondence to code.
- [**Evidence**](./docs/evidence/baseline_results.md): Comparison results and verification logs.
- [**References**](./docs/references/papers.md): List of papers, textbooks, etc.

## Directory Structure
- `src/python/qnm/`: Core implementations (BFGS, L-BFGS, Line Search).
- `src/python/scripts/`: Verification and benchmarking scripts.
- `docs/`: Narrative documentation and evidence.

## License
MIT License
