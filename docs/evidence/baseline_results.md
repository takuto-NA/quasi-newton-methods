# Baseline Results

ローカル環境（NumPy 2.2、SciPy 1.x、強 Wolfe ラインサーチ）での簡易ベンチマーク。

| Problem    | Solver  | Success | f(x\*)          | ‖∇f‖∞      | Iters | Func evals |
|------------|---------|---------|-----------------|------------|-------|------------|
| Quadratic  | BFGS    | ✓       | -9.5878e-01     | 7.6e-07    | 8     | 15         |
| Quadratic  | L-BFGS  | ✓       | -9.5878e-01     | 7.7e-09    | 9     | 12         |
| Rosenbrock | BFGS    | ✓       | 1.89e-18        | 5.5e-08    | 34    | 55         |
| Rosenbrock | L-BFGS  | ✓       | 1.24e-19        | 1.17e-08   | 30    | 87         |

今後は L-BFGS-B（境界付き）のケースや異なる初期値での統計を追加予定。
