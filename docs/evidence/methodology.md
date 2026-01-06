# Methodology

このプロジェクトの Evidence は「参照実装との完全一致」を目的とせず、**(1) 理論上の性質**と **(2) 参照実装（主に SciPy）との差分が説明可能であること**を確認するためのものです。

## 1. 共通設定

- 初期値: 問題定義（`qnm.problems`）に従う
- 停止条件: `‖∇f(x)‖∞ <= tol`
- 反復上限: solverの `max_iter`
- ラインサーチ: Strong Wolfe（`qnm.line_search`、既定 `c1=1e-4, c2=0.9`）

## 2. 記録する指標

- Iterations (`n_iter`)
- Function evaluations (`n_fun`)
- Gradient evaluations (`n_grad`)
- 最終目的関数値 `f(x*)`
- 勾配無限大ノルム `‖∇f‖∞`
- 参考: SciPyとの差分（ただし後述の理由で **合否判定の一次基準にはしない**）

## 3. 比較方針（SciPyを主参照、必要なら他参照も併用）

- **BFGS**: SciPy `minimize(method='BFGS')` を主参照に比較する。ただし SciPy と `qnm` ではラインサーチ等のヒューリスティクスが一致しないため、最終値/解が異なる場合がある。その場合は **既知解への近さ** と **勾配ノルム**を一次基準にして妥当性を判断する。
- **L-BFGS**: SciPyの `minimize(method='L-BFGS-B')` は存在するが、`qnm.lbfgs`（boundなしL-BFGS）とは同一実装ではない。よって L-BFGS は\n+  - 一次: 性質テスト（降下方向、Wolfe充足、既知解収束）\n+  - 二次: 参考比較（例: SciPy L-BFGS-B を bounds無しで実行した最終値）\n+ という二段構えにする。

## 4. Evidence Status（合否定義）

Status は「この検証が何を主張できるか」を表す。

- `PASSED`: solverが `success=True` で終了し、一次基準（例: 既知解への近さ/勾配ノルム/性質テスト）を満たす。
- `ACCEPTABLE_DIFF`: solver自体は一次基準を満たすが、参照実装（例: SciPy）と結果が大きく異なる。差分の原因（例: 局所解、line search差）を脚注で説明できる。
- `FAILED`: `success=False`、または一次基準を満たさない。

## 5. 再現性

- 壁時計時間は環境依存が大きいため補助指標とする。
- 実行環境（Python/SciPy/NumPyのバージョン）は Evidence に出力して併記する。
