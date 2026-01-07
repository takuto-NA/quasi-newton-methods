# quasi-newton-methods

準ニュートン法（BFGS / L-BFGS / L-BFGS-B）の **Evidence-first** な実装と検証をまとめたリポジトリです。

標準的な教科書（Nocedal & Wright）とコードの対応を明確にし、参照実装（主に SciPy）との比較を通じて「なぜこの実装でよいのか」を説明できる状態を目指します。

補足:

- **L-BFGS-B** は `qnm.lbfgsb` として提供していますが、これは **SciPy の参照実装に委譲するラッパー**です（core 実装の検証とは別扱いです）。

## 初見の人へ（まず読む場所）

このプロジェクトは、単に実装を提供するだけでなく、「正しく動いているか」「なぜその実装なのか」という根拠を重視しています。

- [**理論 (Theory)**](./docs/theory/concepts.md): Nocedal & Wright のアルゴリズムとコードの対応（Two-loop recursion の図解あり）。
- [**検証 (Evidence)**](./docs/evidence/baseline_results.md): SciPy との比較結果と、合否の考え方。
- [**出典 (References)**](./docs/references/papers.md): 論文・教科書などの一次情報。

## このプロジェクトの方針
- **アルゴリズムの透明性**: Nocedal & Wright のアルゴリズム（BFGS: Alg 6.1 / L-BFGS: Alg 7.4）とコードを対応づけます。
- **Evidence-first**: 「参照実装との完全一致」ではなく、理論上の性質と差分の説明可能性を重視します。
- **実装者向け**: 手法を “使う” だけでなく “実装する” ための最短距離を意識します。

## Verification Status (Jan 6, 2026)
| Method | Verification | Status |
|--------|--------------|--------|
| **BFGS** | Nocedal & Wright (2006) Alg 6.1 | [PASSED](./docs/evidence/baseline_results.md) |
| **L-BFGS** | Nocedal & Wright (2006) Alg 7.4 | [PASSED](./docs/evidence/baseline_results.md) |
| **Line Search** | Strong Wolfe (Alg 3.5/3.6) | [PASSED](./docs/evidence/baseline_results.md) |

## クイックスタート

### 1. インストール
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1 | macOS/Linux: source .venv/bin/activate
pip install -e ./src/python[dev]
```

### 2. 実装の正当性検証（Fact-check）

自動検証スクリプトを実行して、SciPy などとの比較を行います。

```bash
python src/python/scripts/verify_implementation.py
```

編集インストールを使わずに実行する場合は、`PYTHONPATH` を設定してください。

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

### 3. 使用例

```python
from qnm import bfgs, rosenbrock_problem

problem = rosenbrock_problem(dim=2)
result = bfgs(problem.fun, problem.grad, problem.x0)

print(f"Optimal x: {result.x}")
print(f"Objective value: {result.fun}")
```

## ドキュメント構成
- [**理論 (Theory)**](./docs/theory/concepts.md): 数学的背景とコードへの対応。
- [**検証 (Evidence)**](./docs/evidence/baseline_results.md): 比較結果と検証ログ。
- [**出典 (References)**](./docs/references/papers.md): 論文・教科書などの一覧。

## ディレクトリ構成
- `src/python/qnm/`: Core implementations (BFGS, L-BFGS, Line Search).
- `src/python/scripts/`: Verification and benchmarking scripts.
- `docs/`: Narrative documentation and evidence.

## License
MIT License
