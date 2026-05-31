# FLUENT

音響物理特徴 X → 主観感性ラベル Y へのマッピングフレームワーク。

## プロジェクトの構成

```text
FLUENT/
├── main.py    # YouTubeからの自動ダウンロード、楽曲構造検出、調・コード進行分析の統合スクリプト
├── python/    # バックエンド処理（セグメンタ・特徴量抽出・機械学習モデルの学習と推論、独立したダウンローダーツール群）
│   └── tests/ # `pytest`を利用し、`numpy`・`tmp_path`等でダミー音源を動的生成してテストを実行
├── node/      # フロントエンドUI・APIサーバー（Vanilla JS/HTML + Express、UI/ラベルマネージャー、ツールAPI）
├── data/      # 共有データ（入出力WAV、JSONデータセット、モデル、スケーラー、設定ファイル）
├── templates/ # 教育目的・独自カスタマイズ用に独立動作可能な各コアロジックのテンプレート
```
※Pythonの依存関係は `uv` によって `.venv` 仮想環境内で厳格に管理されます。

## モジュール構成

プロジェクトは独立したモジュールで構成されており、JSONやWAVファイルを介して疎結合に連携します。

- **Module A (Segmenter)**: 音源の分割処理
- **Module B (Extractor)**: 物理特徴量の抽出・標準化
- **Module C (Label Manager & API)**: WebベースのラベリングUIおよびExpress APIサーバー。さらにダウンローダー専用UI(`public/downloader.html`)も含む
- **Module D (Trainer/Predictor)**: MLPによる機械学習・推論
- **Tools (Audio Downloader Tool & Music Analyzer)**: YouTubeからのダウンロード、楽曲構造や調・コードの自動解析（`main.py`）、VAD(音声区間検出)、クロッピング・正規化を行う独立したツール群

## 各モジュールの詳細な機能

- **Module A (python/segmenter.py)**
  `data/raw_audio/` にあるWAVファイルを指定した秒数（デフォルト2秒）で分割し、`data/segments/` に `example-NNN.wav` の形式で出力します。
- **Module B (python/extractor.py)**
  `data/segments/` のWAVファイルから26次元の物理特徴量（MFCC[12次元・第0係数除外], Spectral[4次元], Theory[3次元], Chroma[7次元]）を抽出します。抽出された特徴量は `StandardScaler` で一括して標準化（平均0、標準偏差1）され、新規推論時にも再利用できるよう `data/scaler.pkl` にパラメータが保存されます。また、0.1秒未満の短すぎるセグメントは自動的にスキップされます。
- **Module C (node/server.js, node/public/)**
  - **Main UI (`public/index.html`)**: `wavesurfer.js` を用いたオーディオ再生機能と、`data/config.json` に基づき動的生成されるラベル入力フォーム（color, dropdown, slider, checkboxes 等）を提供します。AIサジェスト機能も統合されています。
  - **API Server (`server.js`)**: イベントループをブロックしないよう、完全に `fs/promises` による非同期I/Oを採用しています。また、`ALLOWED_ORIGINS`（デフォルト: `http://localhost:5173`）を利用したCORS制御を備えています。
    - エンドポイントの特徴として、学習の中断（`POST /api/train/stop` / SIGTERMによる安全な途中保存）やモデルの削除・リセット（`DELETE /api/model`）をサポート。
    - `POST /api/labels` でラベルが更新されるたびに、上書き事故を防ぐため `data/backups/` フォルダへタイムスタンプ付きで `labelset.json` の自動バックアップを作成します。
- **Module D (python/train.py, python/predict.py)**
  - Pythonの各バックエンド処理は、`argparse`を用いてファイルパスの引数指定に対応しており、Node.js側からの堅牢な実行・パースを保証するため、結果となるJSONデータは `stdout` へ出力し、ログやエラーは全て `sys.stderr` へリダイレクトする設計となっています。
  - **Trainer**: MLPRegressor等を用い、`dataset.json` (X) と `labelset.json` (Y) からモデルを学習します。学習済みモデルは `model.pkl` に、学習履歴は `train_meta.json` に保存されます。Epoch毎のLossや収束状態をJSON形式で標準出力し、NodeサーバーのSSEストリーミングに渡します。学習途中での安全な中断（SIGTERMによる途中保存）や、既存モデルからの学習再開（resume）にも対応しています。
  - **Predictor**: 指定IDまたは全セグメントの推論を行い、結果を常に [0.0, 1.0] の範囲に正規化して出力します。特に `--id` 引数を渡すことで、特定セグメント単体での効率的な推論が可能です。
- **Tools (Downloader, VAD, Processor)**
  スタンドアロンのYouTubeダウンロードおよびセグメンテーションツール（`/api/tools/*` および `public/downloader.html`）。
  - **`main.py`**: YouTube等のURLから音声をダウンロードし、Librosa等を用いて楽曲構造（イントロ、Aメロ、サビなど）や調・コード進行を自動解析。指定範囲または解析されたセグメントに分割し、`dataset.json` や `segments.json` にメタデータとともに保存します。
  - **`downloader_tool.py`**: `yt-dlp` を使用しYouTube等から音声を `data/tmp/` にダウンロード。
  - **`vad_tool.py`**: `librosa` を用いて音声区間(VAD)を検出し、無音部分を除外した領域を特定します。
  - **`processor_tool.py`**: `pydub` を用いて指定領域をクロップし、ラウドネス正規化（loudnorm）を適用後、`data/segments/` に `VideoTitle_NNN.wav` の形式で保存します。
  - **`youtube_dl.py`**: YouTube等のURLから直接指定範囲（開始時間・長さ）をダウンロードし、ラウドネス正規化を適用して `data/segments/` に保存します（`/api/youtube` エンドポイントが利用）。
  - **`normalize_existing_audio.py`**: `data/segments/` に存在するすべてのWAVファイルに対して一括でラウドネス正規化（loudnorm）を適用し、音量を統一するユーティリティスクリプト。

## プロセスごとのデータ形式

各プロセス間のデータは、FluCoMa 互換の A-MAP JSONフォーマット、またはWAVファイルでやり取りされます。

- **元音源・セグメント (WAV) およびメタデータ**:
  - `data/raw_audio/` → 分割 → `data/segments/example-NNN.wav`
  - ツールの場合は `data/tmp/` → 処理 → `data/segments/VideoTitle_NNN.wav`
  - `main.py` などのツールが出力する楽曲のセグメント情報、調、コード進行などのメタデータは `data/segments/segments.json` に保存されます。また、調とコード進行情報は `dataset.json` の各セグメントの `global_key` および `global_chords` にも記録されます。
- **特徴量 X (`data/dataset.json`)**:
  - 形式: `{"cols": 26, "data": {"example-001": [0.12, -0.45, ...]}}` （抽出された特徴量に加えて、解析された `global_key` および `global_chords` が追加されることがあります）
  - Extractorが出力し、Trainer/Predictorが読み込みます。すべての値は標準化(Standardization)されています。
- **主観ラベル Y (`data/labelset.json`)**:
  - 形式: `{"cols": 10, "data": {"example-001": {"subjectId": [0.2, 0.6, 1.0, ...]}}}`
  - Module C のUIで生成されファイルに保存されます。Trainer等では特定ユーザーのラベルとして利用されます。UIコンポーネント（checkboxes等）の値は `0.0` または `1.0` のフラットな浮動小数点配列としてエンコードされます。
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

3. UI の Train Model ボタンで学習。epoch ごとの loss がリアルタイム表示される

4. 未ラベルセグメントを選択すると AI Suggestion が自動表示される

## 詳細

- 実装仕様: [IMPLEMENTATION.md](IMPLEMENTATION.md)
- Module C 開発ガイド: [node/DEVELOPMENT.md](node/DEVELOPMENT.md)
- 設計原則: [AGENTS.md](AGENTS.md)
