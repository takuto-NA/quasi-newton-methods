# quasi-newton-methods

クォジニュートン法（BFGS / L-BFGS / L-BFGS-B）の実装と、根拠やベンチマークを整理するプロジェクトです。

- NumPy 実装の BFGS / L-BFGS（強 Wolfe 条件のラインサーチ付き）
- SciPy の fmin_l_bfgs_b をラップした L-BFGS-B
- Rosenbrock / 対称二次問題のベンチマークと勾配チェック
- 参考文献と実験メモを Evidence として保存

詳細は「References」と「Evidence」セクションを参照してください。
