# 理論（Theoretical Concepts）

本プロジェクトは **Nocedal & Wright『Numerical Optimization』（2006）** の定式化に基づき、準ニュートン法（Quasi-Newton methods）を実装しています。

## Newton 法と準ニュートン法（Newton vs. Quasi-Newton）

Newton 法は 2 次の Taylor 展開を用います。

$$f(x_k + p) \approx f(x_k) + \nabla f(x_k)^T p + \frac{1}{2} p^T \nabla^2 f(x_k) p$$

探索方向は $p_k = -(\nabla^2 f(x_k))^{-1} \nabla f(x_k)$ です。一方で、ヘッセ行列 $\nabla^2 f(x_k)$ の計算・逆行列計算は計算量が大きく（概ね $O(n^3)$）、現実的でない場合があります。

**準ニュートン法（Quasi-Newton methods）**では、ヘッセ行列を近似行列 $B_k$（またはその逆行列近似 $H_k$）で置き換え、勾配情報のみで更新します。

## BFGS（Broyden-Fletcher-Goldfarb-Shanno）

BFGS は代表的な準ニュートン法で、ヘッセ行列の**逆行列**近似 $H_k$ を更新します。

### セカント条件（Secant equation）

更新は次のセカント条件を満たす必要があります。

$$H_{k+1} y_k = s_k$$

ここで $s_k = x_{k+1} - x_k$、$y_k = \nabla f_{k+1} - \nabla f_k$ です。

### 更新式（Eq. 6.17, p. 140）

$$H_{k+1} = (I - \rho_k s_k y_k^T) H_k (I - \rho_k y_k s_k^T) + \rho_k s_k s_k^T$$

ただし $\rho_k = 1 / (y_k^T s_k)$ です。

### 曲率条件（Curvature condition）

$H_{k+1}$ が正定（positive definite）であり続けるために、曲率条件

$$s_k^T y_k > 0$$

が必要です。これはラインサーチが **Wolfe 条件（Wolfe conditions）** を満たす場合に保証されます。

## L-BFGS（Limited-memory BFGS）

L-BFGS は大規模問題向けに、$n \times n$ 行列を保持せず、直近 $m$ ステップの $\{s_i, y_i\}$ のみを保持します。

### Two-loop recursion（Alg. 7.4, p. 178）

探索方向 $p_k = -H_k \nabla f_k$ は、$H_k$ を明示的に構成せずに計算できます。

1. 現在の勾配から $q$ を作る
2. 逆順パス（backward pass）: 保存された $s_i, y_i$ を用いて $q$ を更新
3. 初期スケーリング（Eq. 7.20）を適用
4. 順方向パス（forward pass）: $r$（結果）を計算

## 強 Wolfe ラインサーチ（Strong Wolfe line search）

安定性と収束を確保するため、ステップサイズ $\alpha$ は **強 Wolfe 条件（Strong Wolfe conditions）**（Eq. 3.7, p. 34）を満たす必要があります。

1. **十分減少条件（Sufficient decrease）**: $f(x_k + \alpha p_k) \le f(x_k) + c_1 \alpha \nabla f_k^T p_k$
2. **曲率条件（Curvature condition）**: $|\nabla f(x_k + \alpha p_k)^T p_k| \le c_2 |\nabla f_k^T p_k|$

本実装では Nocedal & Wright の推奨に従い、$c_1 = 10^{-4}$、$c_2 = 0.9$ を用います。

## 参考文献（References）
- Nocedal, J., & Wright, S. J. (2006). *Numerical Optimization*. Springer.
- Liu, D. C., & Nocedal, J. (1989). "On the limited memory BFGS method for large scale optimization". *Mathematical Programming*.

