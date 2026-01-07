# ベースライン検証結果（Baseline Results）

`qnm` のベースライン検証結果です。

- 生成日: 2026-01-06
- 生成手順: `PYTHONPATH=src/python python src/python/scripts/verify_implementation.py`

合否の定義（`PASSED` / `ACCEPTABLE_DIFF` / `FAILED`）や、比較の一次/二次の扱いは [methodology](/evidence/methodology) を参照してください。

## 再現性（Environment）

このページの数値は、以下の環境で生成しました。

- Python: 3.12.10
- Platform: Windows-11-10.0.26200-SP0
- NumPy: 2.2.6
- SciPy: 1.14.1

## 表の見方

- `‖∇f‖∞`: 勾配の無限大ノルム（停止条件に使用）
- `Iters`: 反復回数（`n_iter`）
- `Func evals`: 目的関数評価回数（`n_fun`）
- `f_diff (SciPy)`: SciPy 参照実装との最終目的関数値の差（**補助指標**）

`f_diff (SciPy)` について:

- **BFGS**: SciPy `minimize(method='BFGS')` との差（参照比較）
- **L-BFGS**: SciPy `minimize(method='L-BFGS-B')`（bounds無し）との差（**情報用**。一次の合否基準ではない）

目安（直感的な解釈）:

- `f_diff <= 1e-6` であれば、概ね同一の解に収束したとみなせます
- `f_diff >= 1e-0` のように大きい場合、異なる局所解に落ちている可能性があります（`Status` 列と注記を参照）

## 検証テーブル（Verification Table）

| Problem | Solver | Success | f(x*) | ‖∇f‖∞ | Iters | Func evals | f_diff (SciPy) | Status |
|---------|--------|---------|-------|-------|-------|------------|----------------|--------|
| rosenbrock (d=2) | BFGS | ✓ | 1.8932e-18 | 5.5e-08 | 34 | 55 | 9.9e-19 | PASSED |
| rosenbrock (d=2) | L-BFGS | ✓ | 1.2445e-19 | 1.2e-08 | 30 | 87 | 5.3e-10 | PASSED |
| rosenbrock (d=10) | BFGS | ✓ | 2.8043e-16 | 4.2e-07 | 81 | 187 | 4.0e+00* | ACCEPTABLE_DIFF |
| rosenbrock (d=10) | L-BFGS | ✓ | 9.3388e-15 | 6.7e-07 | 73 | 148 | 8.2e-08 | PASSED |
| quadratic (d=5) | BFGS | ✓ | -2.8161e-01 | 5.4e-09 | 10 | 18 | 3.4e-14 | PASSED |
| quadratic (d=5) | L-BFGS | ✓ | -2.8161e-01 | 1.4e-07 | 12 | 16 | 2.3e-09 | PASSED |
| quadratic (d=50) | BFGS | ✓ | -7.0240e-01 | 3.7e-07 | 59 | 294 | 3.5e-14 | PASSED |
| quadratic (d=50) | L-BFGS | ✓ | -7.0240e-01 | 9.8e-07 | 53 | 63 | 8.8e-07 | PASSED |

\* Note (BFGS only): For Rosenbrock (d=10), SciPy's `minimize(method='BFGS')` converged to a local minimum ($f \approx 3.986$), whereas our implementation reached the global minimum ($f \approx 0$). This discrepancy is due to differences in line search heuristics.

## 注記

- 本テーブルは **`qnm`（core implementation: BFGS / L-BFGS）** の検証です。
- **L-BFGS-B** は `qnm.lbfgsb` が SciPy の参照実装に委譲するため、ここでは「自前実装の正当性」の対象から外しています。

## Fact-Check Summary（要約）

1. **勾配チェック（Gradient Check）**: ベンチマーク問題の解析的勾配を中央差分で検証（差分 < 1e-6）。
2. **収束（Convergence）**: 代表的な問題で、妥当な最適化結果（最終値・勾配ノルム）を得ることを確認。
3. **正定性（Positive Definiteness）**: 曲率条件 $s_k^T y_k > 0$ を明示的にチェックし、違反時は更新をリセット（本ベンチでは未発火）。
4. **理論との対応（Source Integrity）**: `bfgs.py` / `lbfgs.py` の主要ステップを Nocedal & Wright（2006）に対応付け。
