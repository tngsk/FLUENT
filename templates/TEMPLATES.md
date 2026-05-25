# プロジェクトテンプレート

教育目的および独自プロジェクトへのカスタマイズ用に、各機能のコアロジックのみを抽出した独立動作可能なテンプレートです。

## ディレクトリ構成

```text
templates/
├── template_segmenter.py  # 音声分割（librosaの単純なスライス）
├── template_extractor.py  # 特徴量抽出（MFCC + StandardScaler）
├── template_train.py      # モデル学習（scikit-learn MLPRegressor）
├── template_predict.py    # 推論（学習済みモデルによる予測と正規化）
├── template_server.js     # Node.js (Express) バックエンドAPIの最小構成
└── public/
    └── index.html         # Vanilla JSによるフロントエンドUIの最小構成
```

## 使い方

### Python スクリプト

各スクリプトは独立して実行可能です。必要に応じて入出力のファイルパスやパラメータをスクリプト下部の `if __name__ == "__main__":` ブロック内で調整してください。

実行例:
```bash
python template_segmenter.py
python template_extractor.py
python template_train.py
python template_predict.py
```

※実行には `librosa`, `soundfile`, `scikit-learn` 等のパッケージが必要です。

### Node.js サーバーとUI

Node.jsサーバーは、フロントエンド用の静的ファイル配信と、Pythonスクリプトを呼び出すAPIエンドポイントの例を提供します。

1. 依存関係のインストール（必要に応じて）:
   ```bash
   npm install
   ```
2. サーバーの起動:
   ```bash
   node template_server.js
   ```
3. ブラウザでアクセス:
   `http://localhost:3000` にアクセスし、UIのボタンをクリックしてAPI呼び出しをテストします。

## カスタマイズのポイント

- **データ構造**: すべてのテンプレートは、`{"cols": N, "data": {"id": [...]}}` という A-MAP 互換のJSON構造を想定・生成するように作られています。
- **エラーハンドリング**: テンプレートはシンプルさを重視し、高度なエラー処理（リトライや詳細なログ出力）は省かれています。実運用に向けて追加してください。
- **特徴量**: 現在はMFCCのみを抽出していますが、`template_extractor.py` の `librosa` 関数を増やすことで次元を拡張できます。
