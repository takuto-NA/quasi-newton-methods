# quasi-newton-methods

Evidence-first implementations and benchmarks for quasi-Newton methods (BFGS, L-BFGS, L-BFGS-B).

## これを読めばわかること（初見向け）
- 「何があるか」: NumPy 実装の BFGS / L-BFGS、SciPy ラッパーの L-BFGS-B、ベンチマーク問題、勾配チェック。
- 「どう動かすか」: venv を作り `pip install -e ./src/python[dev]` して `python -m pytest` を叩けば動作確認できる。
- 「どこが根拠か」: `docs/references/*` に出典、`docs/evidence/*` に検証結果と条件を記録。

## 特長
- NumPy 実装の BFGS / L-BFGS（強 Wolfe 条件ラインサーチ付き）
- SciPy への薄いラッパーで L-BFGS-B も利用可能（SciPy を入れれば使える）
- Rosenbrock / 対称二次ベンチマークと有限差分による勾配チェックユーティリティ
- VitePress 製ドキュメント（参考文献・エビデンス）

## 前提
- Python 3.10+
- SciPy は L-BFGS-B を使う場合にのみ必要（`pip install scipy` または `pip install -e ./src/python[dev]`）
- Node.js はドキュメントを触るときだけ必要

## 最短クイックスタート（動作確認まで）
```bash
python -m venv .venv
# Windows (PowerShell): .\\.venv\\Scripts\\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -U pip
pip install -e ./src/python[dev]

# 動作確認: 勾配チェックとベンチマーク問題をまとめて実行
set PYTHONPATH=src/python  # PowerShell: $env:PYTHONPATH="src/python"; macOS/Linux: export PYTHONPATH=src/python
python -m pytest src/python/tests
```

## すぐ使えるサンプル
```python
from qnm import bfgs, lbfgs, lbfgsb, rosenbrock_problem

problem = rosenbrock_problem()
result = bfgs(problem.fun, problem.grad, problem.x0)
print("x*", result.x)
print("f(x*)", result.fun)

# Bounds 付きの L-BFGS-B（SciPy に依存）
# lbfgsb(problem.fun, problem.grad, problem.x0, bounds=[(-2.0, 2.0), (-1.0, 3.0)])
```

## API チートシート
- `bfgs(fun, grad, x0, max_iter=200, tol=1e-6, line_search_kwargs=None)` → `OptimizeResult`
- `lbfgs(fun, grad, x0, m=10, max_iter=200, tol=1e-6, line_search_kwargs=None)` → `OptimizeResult`
- `lbfgsb(fun, grad, x0, bounds=[(lo, hi), ...], tol=1e-6, max_iter=15000)` → `OptimizeResult`（SciPy 依存）
- ベンチマーク問題: `rosenbrock_problem(dim=2)`, `quadratic_problem(dim=5, condition_number=10, seed=0)`
- 勾配チェック: `gradient_check(fun, grad, x)`

## ドキュメント（任意）
```bash
cd docs
npm run docs:dev      # ローカルプレビュー
npm run docs:build    # 静的ビルド
```
公開済み: https://takuto-NA.github.io/quasi-newton-methods/

## エビデンス・出典
- 出典: `docs/references/*`
- 検証条件・結果: `docs/evidence/*`（ベースライン表あり）

## ディレクトリ
- `src/python/qnm/`: Python 実装
- `provenance/`: 参考文献・前提のメモ
- `docs/`: VitePress ドキュメント
- `.github/workflows/`: CI 設定

## ライセンス
MIT License
