# quasi-newton-methods

クォジニュートン法（BFGS / L-BFGS / L-BFGS-B）の実装と、根拠やベンチマークを整理するプロジェクトです。

初見で知りたいこと
- 何があるか: NumPy 実装の BFGS / L-BFGS、SciPy ベースの L-BFGS-B、ベンチマーク問題、勾配チェック。
+- どう動くか: `pip install -e ./src/python[dev]` → `PYTHONPATH=src/python python -m pytest` で動作確認できる。
- どこを見れば根拠があるか: References（出典）と Evidence（検証ログ）に分けて記載。

Sections
- References: 実装・ライブラリ・論文のリンク
- Evidence: 手法・条件・ベースライン結果
