# Main (`main.py`) 解説

## 実装と理論的背景
`main.py` は、YouTube 等の動画 URL から音声をダウンロードし、楽曲の構造（イントロ、Aメロ、サビなど）や、調（Key）・コード進行（Chords）を自動的に解析し、意味のあるセグメントに分割する統合スクリプトです。

### 理論的背景
1. **楽曲構造の自動検出**
   - 楽曲の特徴量（クロマとMFCC）を時間軸でスタックし、過去の文脈を考慮した「自己回帰行列（Recurrence Matrix）」的なアプローチを用いてクラスタリングを行います。
   - クラスタリング結果の境界を、楽曲の展開が変わるポイント（セクションの変わり目）とみなします。
2. **調（Key）とコード（Chords）の推定**
   - **Key**: クロマ特徴量の時間平均から、最も強く鳴っている音階（ルート）と、長短（メジャー/マイナー）のパターンを当てはめて推定します。
   - **Chords**: 時間枠ごとにクロマ特徴量を計算し、事前定義されたメジャー・マイナーコードのテンプレート（プロファイル）との類似度（コサイン類似度など）を比較することで、各瞬間のコードを推定します。

### 実装解説
もとのコードの主要な処理部分を引用しながら解説します。

#### 楽曲構造の検出 (`get_segments_via_librosa`内)
```python
    # 和音成分(Chroma)と音色成分(MFCC)を抽出
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    # 特徴量を結合し、時間的な変化を捉えるためにスタックする
    combined_features = np.vstack([chroma, mfcc])
    stacked_features = librosa.feature.stack_memory(
        combined_features, n_steps=10, delay=3
    )
```
音色（MFCC）と和音（クロマ）の両方の特徴量を結合し、さらに時間的に遅延させた特徴量をスタックすることで、時間的な文脈を持たせた特徴量ベクトルを作成します。

```python
    # sklearn.cluster.AgglomerativeClustering の connectivity 引数に渡すための疎行列
    # 時間的に隣接するフレームのみを接続すると定義します。
    n_frames = stacked_features.shape[1]
    connectivity = lil_matrix((n_frames, n_frames), dtype=int)
    for i in range(n_frames):
        if i > 0:
            connectivity[i, i - 1] = 1
            connectivity[i - 1, i] = 1  # 双方向
    connectivity = connectivity.tocsr()  # CSR形式に変換

    # 3. Agglomerative Clustering を実行
    model = AgglomerativeClustering(
        n_clusters=6, connectivity=connectivity, linkage="ward"
    )
    cluster_labels = model.fit_predict(stacked_features.T)
```
隣接するフレーム間のみを繋ぐ connectivity 行列を作成し、これを用いて Agglomerative Clustering を実行します。これにより、時間的に連続したフレームが一つのクラスタ（セクション）になりやすくなります。

#### 音声の切り出しと正規化 (`download_specific_sections`内)
```python
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(info["start"]),  # 入力の前に置くことで高速に該当箇所へ移動
            "-t",
            str(duration),
            "-i",
            temp_source,
            "-vn",
            "-af",
            "loudnorm=I=-21:TP=-9.0:LRA=7,volume=-3dB",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "44100",
            "-ac",
            "2",
            output_path,
        ]
        subprocess.run(
            ffmpeg_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
```
検出されたセグメントの開始時間（`-ss`）と長さ（`-t`）を指定して `ffmpeg` で音声を切り出します。このとき、`-af` オプションで `loudnorm` フィルタを適用し、音量のラウドネス正規化を行っています。

#### メタデータの保存
```python
    # [STEP 4] dataset.json を更新
    # ...
    for item in downloaded_info:
        seg_id = item["id"]
        if seg_id not in dataset["data"]:
            dataset["data"][seg_id] = {}
        # 調とコード進行を記録
        dataset["data"][seg_id]["global_key"] = key_info
        dataset["data"][seg_id]["global_chords"] = chord_info
```
切り出したセグメントに対して推定した調（`global_key`）とコード進行（`global_chords`）を、システム全体の `dataset.json` にメタデータとして書き込みます。
