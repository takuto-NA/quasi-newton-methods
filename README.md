# quasi-newton-methods

Evidence-first implementations and benchmarks for quasi-Newton methods (BFGS, L-BFGS, L-BFGS-B).

## 0. 目的
* **出典（論文・公式ドキュメント・既存実装）を追跡できること**
* **再現可能な比較（SciPy 等）で“正しい”を検証できること**
* **実装は “読みやすい基準版” と “高速版（Numba等）” を分離し、同じテストで担保すること**
* **docs（VitePress）とCI（GitHub Actions）を常にグリーンに保つこと**

## 1. 開発環境のセットアップ

```bash
# ドキュメントのセットアップ
cd docs
npm install
cd ..

# Python のセットアップ
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ./src/python
pip install -e ./src/python[dev]
```

## 2. テストの実行

```bash
pytest src/python/tests
```

## 3. ドキュメント

本プロジェクトのドキュメント（実装の詳細、出典、ベンチマーク結果）は GitHub Pages で公開されています。

URL: [https://takuto-NA.github.io/quasi-newton-methods/](https://takuto-NA.github.io/quasi-newton-methods/)

## 4. プロジェクト構成

- `src/python/qnm/`: 基準となる Python 実装
- `provenance/`: 各手法の出典と設計方針の YAML 管理
- `docs/`: VitePress によるドキュメントと検証結果
- `.github/workflows/`: CI (テスト、ドキュメント公開)

## 5. ライセンス

MIT License
