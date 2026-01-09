# 理論（Theoretical Concepts）

本プロジェクトは **Nocedal & Wright『Numerical Optimization』（2006）** の定式化に基づき、準ニュートン法（Quasi-Newton methods）を実装しています。

## 核心となるアルゴリズム

各手法の詳細な理論と実装上のポイントについては、以下の個別記事を参照してください。

- **[BFGS 法 (Standard BFGS)](bfgs.md)**
  - 逆ヘッセ行列を直接更新する標準的な手法。中規模までの問題に適しています。
- **[L-BFGS 法 (Limited-memory BFGS)](lbfgs.md)**
  - 更新履歴のみを保持するメモリ節約型。大規模な問題に不可欠な手法です。
- **[L-BFGS-B 法 (Bound-constrained)](lbfgsb.md)**
  - 変数に範囲制約がある場合の拡張。投影勾配とアクティブセットの概念を用います。

---

## 共通の基盤技術

各アルゴリズムに共通する重要な構成要素です。

### Newton 法と準ニュートン法
Newton 法は 2 次の Taylor 展開を用い、探索方向 $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$ を決定します。しかし、ヘッセ行列 $\nabla^2 f(x_k)$ の計算コスト（$O(n^2)$）とその逆行列計算（$O(n^3)$）がボトルネックとなります。

**準ニュートン法**では、ヘッセ行列の逆行列を近似行列 $H_k$ で置き換え、勾配情報（一次微分）のみを用いて反復的に更新することで、高い収束性と計算効率を両立させています。

### 強 Wolfe ラインサーチ (Strong Wolfe Line Search)
数値的な安定性と収束を確保するため、すべてのアルゴリズムでステップサイズ $\alpha$ は **強 Wolfe 条件** を満たすように決定されます。

1.  **十分減少条件 (Sufficient Decrease)**: 目的関数が現在の勾配から期待される以上に減少すること。
2.  **曲率条件 (Curvature Condition)**: ステップ後の勾配が緩和され、正定値性が維持されること。

本実装では Nocedal & Wright の Alg. 3.5 に基づき、$c_1 = 10^{-4}, c_2 = 0.9$ を既定値として使用しています。

## 参考文献
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
