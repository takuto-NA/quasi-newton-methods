# 実装（Implementations）

外部実装の参照先です。
このリポジトリのコードを読む際に「この処理はどこで一般に実装されているか」を辿る用途を想定しています。

SciPy の公式ドキュメントは [SciPy](/references/scipy) にまとめています。

関連:
- [実装比較（BFGS / L-BFGS）](/references/implementation_comparison)

## SciPy
- [optimize.minimize(method='BFGS')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-bfgs.html)
- [optimize.minimize(method='L-BFGS-B')](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html)

## L-BFGS-B（別実装）
- [bgranzow/L-BFGS-B](https://github.com/bgranzow/L-BFGS-B): L-BFGS-B の独立実装（学習/比較用）

## PyTorch
- [hjmshi/PyTorch-LBFGS](https://github.com/hjmshi/PyTorch-LBFGS): PyTorch 向けの L-BFGS 実装

## C++（CppNumericalSolvers / cppoptlib）
- [PatWie/CppNumericalSolvers](https://github.com/PatWie/CppNumericalSolvers): C++ 最適化ライブラリ（BFGS / L-BFGS / L-BFGS-B を含む）

