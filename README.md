# quasi-newton-methods

Evidence-first implementations and benchmarks for quasi-Newton methods (BFGS, L-BFGS, L-BFGS-B).

## 特長
- NumPy 実装の BFGS / L-BFGS（強 Wolfe 条件ラインサーチ付き）
- SciPy への薄いラッパーで L-BFGS-B も利用可能
- Rosenbrock / 対称二次のベンチマーク問題と有限差分による勾配チェックユーティリティ
- VitePress 製ドキュメント（参考文献・エビデンス）

## セットアップ
```bash
python -m venv .venv
# Windows (PowerShell): .\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -e ./src/python[dev]

# ドキュメント（任意）
cd docs
npm install
cd ..
```

## 使い方
```python
from qnm import bfgs, lbfgs, lbfgsb, rosenbrock_problem

problem = rosenbrock_problem()
result = bfgs(problem.fun, problem.grad, problem.x0)
print("x*", result.x)
print("f(x*)", result.fun)

# Bounds 付きの L-BFGS-B（SciPy に依存）
# lbfgsb(problem.fun, problem.grad, problem.x0, bounds=[(-2.0, 2.0), (-1.0, 3.0)])
```

## テスト
```bash
set PYTHONPATH=src/python  # PowerShell: $env:PYTHONPATH="src/python"
python -m pytest src/python/tests
```

## ドキュメント（任意）
```bash
cd docs
npm run docs:dev      # ローカルプレビュー
npm run docs:build    # 静的ビルド
```
公開済み: https://takuto-NA.github.io/quasi-newton-methods/

## ディレクトリ
- `src/python/qnm/`: Python 実装
- `provenance/`: 参考文献・前提のメモ
- `docs/`: VitePress ドキュメント
- `.github/workflows/`: CI 設定

## ライセンス
MIT License
