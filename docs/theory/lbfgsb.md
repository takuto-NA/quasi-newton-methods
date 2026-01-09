# L-BFGS-B (Limited-memory BFGS with Bounds)

L-BFGS-B は、変数が一定の範囲内に収まる必要がある**境界制約付き（Box-constrained）**の最適化問題を解くためのアルゴリズムです。

## 1. 問題の定義

以下の制約条件下での最小化を考えます。

$$\min_{x \in \mathbb{R}^n} f(x)$$
$$\text{subject to } l_i \le x_i \le u_i, \quad i = 1, \dots, n$$

ここで、$l_i$ は下限、$u_i$ は上限です（制限がない場合は $\pm \infty$）。

## 2. アルゴリズムの主要概念

無制約の L-BFGS と異なり、単に探索方向に進むだけでは制約（境界）を越えてしまう可能性があります。L-BFGS-B では以下の2つのステップを組み合わせます。

### 投影勾配 (Projected Gradient)
現在の点 $x$ において、制約を満たす範囲内で最も改善が見込める方向を決定します。収束判定には、通常の勾配ではなく、境界に当たっている成分を考慮した**投影勾配**のノルム（$||g^{pg}||_\infty$）が用いられます。これは SciPy や bgranzow (Matlab) 実装における一次基準（`pgtol`）と一致します。

$$x^{pg} = P_{[l,u]}(x - \nabla f(x))$$
$$g^{pg}(x) = x - x^{pg}$$

ここで $P_{[l,u]}(\cdot)$ は成分ごとのクリップ（投影）です。
### 1. Generalized Cauchy Point (GCP)
現在の L-BFGS 近似二次モデルを用いつつ、勾配方向に沿って制約境界にぶつかるまで（あるいは極小値に達するまで）探索を行います。このプロセスで「どのアクティブな制約（境界）に留まるべきか」を決定します。

### 2. Subspace Minimization
Cauchy Point によって固定された境界以外の「自由な変数」について、L-BFGS の情報を用いてさらに最適化を行います。これにより、制約付き問題でありながら準ニュートン法の高い収束性を維持します。

## 3. 実装の形態 (`qnm.lbfgsb`)

制約付き最適化、特に GCP の計算やアクティブセットの管理は非常に複雑です。本プロジェクトの `qnm.lbfgsb` は、一貫した API を提供しつつ、内部的な計算は **SciPy の `fmin_l_bfgs_b`** （Fortran 実装のラッパー）に委譲しています。

### SciPy への委譲の理由
1. **信頼性**: 元の L-BFGS-B アルゴリズム（Byrd et al., 1995）は実装が非常にデリケートであり、SciPy の実装は長年広く利用され検証されています。
2. **パフォーマンス**: 境界判定や部分空間の最小化などのループ処理が Fortran で高速化されています。

## 4. 停止条件の注意点

L-BFGS-B では、以下のいずれかで終了します。
- **pgtol**: 投影勾配の最大成分がこの値以下になった場合（成功）。
- **factr**: 目的関数の改善率が閾値を下回った場合。
- **maxiter / maxfun**: 反復回数や関数評価回数の上限に達した場合。

## 参考文献
- Byrd, R. H., Lu, P., Nocedal, J., & Zhu, C. (1995). "A Limited Memory Algorithm for Bound Constrained Optimization". *SIAM Journal on Scientific and Statistical Computing*.
- Zhu, C., Byrd, R. H., Lu, P., & Nocedal, J. (1997). "Algorithm 778: L-BFGS-B". *ACM Transactions on Mathematical Software*.
