# FLUENT

音響物理特徴 X → 主観感性ラベル Y へのマッピングフレームワーク。

## プロジェクトの構成

```text
FLUENT/
├── python/    # バックエンド処理（セグメンタ・特徴量抽出・機械学習モデルの学習と推論、独立したダウンローダーツール群）。依存関係は `uv` によって `.venv` で管理。
├── node/      # フロントエンドUI・APIサーバー（Vanilla JS/HTML + Express、UI/ラベルマネージャー、ツールAPI）。依存関係は `npm` で管理。
├── data/      # 共有データ（入出力WAV、JSONデータセット、モデル、スケーラー、設定ファイル）
├── templates/ # 教育目的・独自カスタマイズ用に独立動作可能な各コアロジックのテンプレート
```

## モジュール構成

プロジェクトは独立したモジュールで構成されており、JSONやWAVファイルを介して疎結合に連携します。

- **Module A (Segmenter)**: 音源の分割処理
- **Module B (Extractor)**: 物理特徴量の抽出・標準化
- **Module C (Label Manager & API)**: WebベースのラベリングUIおよびExpress APIサーバー。さらにダウンローダー専用UI(`public/downloader.html`)も含む
- **Module D (Trainer/Predictor)**: MLPによる機械学習・推論
- **Tools (Audio Downloader Tool)**: YouTubeからのダウンロード、VAD(音声区間検出)、クロッピング・正規化を行う独立したツール群

## 各モジュールの詳細な機能

- **Module A (python/segmenter.py)**
  `data/raw_audio/` にあるWAVファイルから、人間の聴覚特性に近いMFCC（メル周波数ケプストラム係数）を抽出し、凝集型クラスタリング (Agglomerative Clustering) を用いて音響的に類似した部分をグループ化することで、音声を自動的に「自然な切れ目」で分割します。デフォルトでは約2秒粒度で分割され、`data/segments/` に `example-NNN.wav` の形式で出力されます。
- **Module B (python/extractor.py)**
  `data/segments/` のWAVファイルから、機械学習モデルへの入力として使用する26次元の物理特徴量を抽出します。なお、0.1秒未満の音声セグメントは抽出対象外として除外されます。構成は以下の通りです：
  - MFCC (12次元): 第0係数（音量に依存する成分）を除外し、音色の特性に焦点を当てたもの。
  - Spectral Features (4次元): スペクトルの形状（Centroid, Flatness, Rolloff, Bandwidth）。
  - Theoretical Features (3次元): 音楽理論的な特徴（Tempo, Key, Mode）。
  - Chroma Feature (7次元): 12音階の強さの分布（最初の7半音分）。
  抽出された全セグメントの特徴量は `StandardScaler` で一括して標準化（平均0、分散1）され、`data/dataset.json`（A-MAP形式）とスケーラーのパラメータが `data/scaler.pkl` に保存されます。
- **Module C (node/server.js, node/public/)**
  - **Main UI (`public/index.html`)**: `<audio>` タグを用いたオーディオ再生機能と、`data/config.json` に基づき動的生成されるラベル入力フォーム（color, dropdown, slider, checkboxes 等）を提供します。動的フォームの生成時、checkboxesなどのタイプは `0.0` または `1.0` のフラットな浮動小数点配列としてエンコード/デコードされます。DOMの描画最適化のため `DocumentFragment` を使用してリフローを最小限に抑えています。未ラベルセグメントのAIサジェスト機能も統合されています。
  - **API Server (`server.js`)**: イベントループのブロッキングを避けるため `fs/promises` による非同期I/Oと、`child_process.spawn` によるPythonスクリプトの非同期呼び出しでオーケストレーションを行います。`ALLOWED_ORIGINS` 環境変数を用いた制限付きCORSポリシー（デフォルト `http://localhost:5173`）を実装。Pythonプロセスから標準出力（stdout）経由でJSON出力される学習進捗を傍受し、Server-Sent Events (SSE) でリアルタイムにUIへストリーミングします。また、`labelset.json` の自動バックアップ、学習の中断（SIGTERM送信）やモデルのリセット、ツールによる音声追加後の自動的な特徴量抽出パイプラインの連鎖呼び出し、特定被験者ごとのラベルフィルタリング、アンケートの保存・取得（`/api/survey`）などをサポートします。
- **Module D (python/train.py, python/predict.py)**
  - **Trainer**: MLPRegressor等を用い、`dataset.json` (X) と `labelset.json` (Y) をマッピングするモデルを学習します。過学習を防ぐためL2正則化（alpha）を適用し、データサイズに応じて最適化アルゴリズム（lbfgs/adam）を動的に切り替えます。学習途中での安全な中断（SIGTERMによる途中保存）や、既存モデルからの学習再開（resume）に対応しています。C実装ループであっても標準出力を横取り(`VerboseCapture`)してJSON形式に整形し、Nodeサーバーへ進行状況を伝えます。
  - **Predictor**: `extractor.py`の`scaler.pkl`と`train.py`の`model.pkl`を用いて、新たな音声に対するラベルを推論します。`--id` 引数により特定セグメントのみ推論する効率的なフィルタリングが可能。推論結果は、システムの破綻を防ぐため常に `[0.0, 1.0]` の範囲にクリッピング（外挿対策）してJSON形式で出力されます。
- **Tools (Downloader, VAD, Processor)**
  スタンドアロンのYouTubeダウンロードおよびセグメンテーションツール（`/api/tools/*` および `public/downloader.html`）。
  - **`main.py`** (ルートディレクトリ): YouTubeからのダウンロード、楽曲の特徴量（クロマとMFCC）を時間軸でスタックし、過去の文脈を考慮した「自己回帰行列」アプローチによる凝集型クラスタリングで音楽構造（イントロ、Aメロ、サビなど）の境界を自動検出します。また、クロマ特徴量の時間平均とプロファイル照合から調（Key）とコード（Chords）を推定し、構造メタデータを `data/segments/segments.json` や `dataset.json` に出力します。Node.jsサーバーの `/api/youtube` エンドポイントから自動セグメンテーション用としてトリガーされます。
  - **`downloader_tool.py`**: `yt-dlp` を使用しYouTube等から音声を `data/tmp/` にダウンロード。
  - **`vad_tool.py`**: `librosa` を用いて音声区間(VAD)を検出し、無音部分を除外した領域を特定します。
  - **`processor_tool.py`**: `pydub` を用いて指定領域をクロップし、ラウドネス正規化（loudnorm）を適用後、`data/segments/` に `VideoTitle_NNN.wav` の形式で保存します。
  - **`youtube_dl.py`**: YouTube等のURLから直接指定範囲（開始時間・長さ）をダウンロードし、ラウドネス正規化を適用して `data/segments/` に保存します。
  - **`normalize_existing_audio.py`**: `data/segments/` に存在するすべてのWAVファイルに対して一括でラウドネス正規化（loudnorm）を適用し、音量を統一するユーティリティスクリプト。

## プロセスごとのデータ形式

各プロセス間のデータは、FluCoMa 互換の A-MAP JSONフォーマット、またはWAVファイルでやり取りされます。

- **元音源・セグメント (WAV)**:
  - `data/raw_audio/` → 分割 → `data/segments/example-NNN.wav`
  - ツールの場合は `data/tmp/` → 処理 → `data/segments/VideoTitle_NNN.wav`
- **特徴量 X (`data/dataset.json`)**:
  - 形式: `{"cols": 26, "data": {"example-001": [0.12, -0.45, ...]}}`
  - Extractorが出力し、Trainer/Predictorが読み込みます。すべての値は標準化(Standardization)されています。また、`main.py` などの自動処理により、`global_key` および `global_chords` も追加記録されます。
- **主観ラベル Y (`data/labelset.json`)**:
  - 形式: `{"cols": 8, "data": {"example-001": {"subjectId": {"labels": [0.2, 0.6, 1.0, ...], "survey": {...}}}}}`
  - Module C のUIで生成されファイルに保存されます。複数被験者(subjectId)をサポートするため入れ子構造になっています。分析しやすいよう、ラベル配列とアンケート回答をセットで保存します。Trainer等では特定ユーザーのラベルとして利用されます。UIコンポーネント（checkboxes等）の値は `0.0` または `1.0` のフラットな浮動小数点配列としてエンコードされます。
- **学習済みモデル・状態**:
  - `data/scaler.pkl`: StandardScalerのパラメータ。新規推論時の正規化に必要。
  - `data/model.pkl`: MLPの重みデータ。推論や学習の再開(`resume`)に使用。
  - `data/train_meta.json`: 最後に学習された日時とIDのリスト。UIのステータス表示に利用。

## クイックスタート

```bash
cp .env.example .env
bash setup_all.sh
```

`setup_all.sh` はダミー音源生成・セグメント分割・特徴量抽出まで自動で実行します。

### UI 起動

```bash
cd node && npm run dev
```

`http://localhost:3000` にアクセス。

## 手順

1. `data/raw_audio/` に WAV を配置して特徴量抽出（自前素材を使う場合）

```bash
# セグメント分割（--duration 2, 5, 10 などで粒度調整可能）
uv run python python/segmenter.py --duration 2

# 特徴量抽出（26次元物理特徴の生成）
uv run python python/extractor.py
```

2. ブラウザでセグメントを試聴 → ラベル入力 → Save Labels

3. 学習は CLI または API で実行（UI の Train Model ボタンは廃止）

4. 未ラベルセグメントを選択すると AI Suggestion が自動表示される（API 経由）

## 詳細

- 実装仕様: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- モジュールごとの詳細な実装と理論の解説: [docs/](docs/) ディレクトリを参照してください。
- Module C 開発ガイド: [node/DEVELOPMENT.md](node/DEVELOPMENT.md)
- 設計原則: [AGENTS.md](AGENTS.md)
- テスト: Pythonのユニットテストは `pytest` を使用し、テスト用のダミー音声ファイルは静的に保持せず動的に生成されます（実行例: `uv run pytest python/tests/`）。
