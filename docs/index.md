# quasi-newton-methods

クォジニュートン法（BFGS / L-BFGS / L-BFGS-B）の実装と、その理論的根拠・検証結果を整理するプロジェクトです。

## 初見の人へ

このプロジェクトは、単なる実装の提供だけでなく、「正しく動いているか」「なぜこの実装なのか」という**根拠（Evidence）**を重視しています。

- **[理論解説 (Theory)](/theory/concepts)**: Nocedal & Wright のアルゴリズムとコードの対応。
- **[検証結果 (Evidence)](/evidence/baseline_results)**: SciPy との比較結果と、勾配チェックの結果。
- **[出典 (References)](/references/papers)**: 基礎となる論文と教科書のリスト。

## クイックスタート

```bash
# （推奨）インストール
pip install -e ./src/python[dev]

# 実装の正当性検証 (Fact-check)
python src/python/scripts/verify_implementation.py
```

編集インストールを使わずに実行する場合は、`PYTHONPATH` を設定してください。

```bash
# macOS/Linux (bash/zsh) / Git Bash
export PYTHONPATH=src/python
python src/python/scripts/verify_implementation.py
```

```bash
# Windows PowerShell
$env:PYTHONPATH = "src/python"
python src/python/scripts/verify_implementation.py
```

```bash
# Windows cmd.exe
set PYTHONPATH=src/python
python src/python/scripts/verify_implementation.py
```

## 実装されている手法

1.  **[BFGS](/theory/bfgs)**: 逆ヘッセ行列を直接更新する標準的な手法。
2.  **[L-BFGS](/theory/lbfgs)**: メモリ節約型の準ニュートン法（Two-loop recursion）。
3.  **Line Search**: 強 Wolfe 条件を満たすステップサイズ決定アルゴリズム。
4.  **[L-BFGS-B](/theory/lbfgsb)**: 境界制約付き（SciPy ラッパー）。
