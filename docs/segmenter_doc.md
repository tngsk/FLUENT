# Segmenter (`python/segmenter.py`) 解説

## 実装と理論的背景
`python/segmenter.py` は、入力された音声ファイルから MFCC（メル周波数ケプストラム係数）を計算し、凝集型クラスタリング (Agglomerative Clustering) を用いて音響的に類似した部分をグループ化することで、音声を自動的に「自然な切れ目」で分割するモジュールです。

### 理論的背景
1. **MFCC (Mel-Frequency Cepstral Coefficients)**
人間の聴覚特性に近いメル尺度を用いて周波数スペクトルを表現した特徴量です。音声認識や音楽情報検索において、音色の違いを捉えるための標準的な手法として広く用いられています。
2. **凝集型クラスタリング (Agglomerative Clustering)**
各データ点（ここでは各フレームの MFCC）を1つのクラスタとして開始し、類似度が高いクラスタ同士を順次統合していく階層的クラスタリングの一種です。このスクリプトでは `librosa.segment.agglomerative` を用いて、隣接するフレーム間の類似性に基づき、指定された目標クラスタ数（セグメント数）になるまで統合を行います。

### 実装解説
もとのコードの主要な処理部分を引用しながら解説します。

#### 波形の読み込み
```python
        # y: 音声の波形データ（1次元配列。例: 44100Hz × 30秒 = 1,323,000サンプル）
        # sr: サンプリングレート（Hz。通常 44100 または 48000）
        y, sr = librosa.load(file_path, sr=None)
```
`librosa.load` を使用して入力されたWAVファイルを読み込み、波形データ `y` とサンプリングレート `sr` を取得します。

#### MFCCの計算
```python
        # ============================================================
        # Step 4: MFCC（メル周波数ケプストラム係数）を計算
        # ============================================================
        # MFCC は「人間の耳に聞こえやすい周波数特性」を13次元で表現
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        smoothed_features = mfcc
```
`librosa.feature.mfcc` を用いて、波形データから MFCC を抽出します。これがクラスタリングの入力特徴量となります。

#### 目標クラスタ数の決定
```python
        # ============================================================
        # Step 5: セグメント数 k を自動決定
        # ============================================================
        # k: Agglomerative clustering の目標クラスタ数
        #
        # 計算式: max(2, len(y) // (sr * duration_per_segment))
        k = int(max(2, len(y) // (sr * duration_per_segment)))
```
目標とする分割単位時間 (`duration_per_segment`) に基づき、全体の音声長からクラスタ数 `k` を決定します。最小でも2分割されるように `max(2, ...)` が用いられています。

#### クラスタリングと境界の検出
```python
            # ============================================================
            # Step 6: Agglomerative Clustering で分割境界を検出
            # ============================================================
            cluster_labels = librosa.segment.agglomerative(smoothed_features, k)
            boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 1
```
`librosa.segment.agglomerative` に MFCC の時系列データを与え、`k` 個のクラスタに分割します。隣接するフレーム間でクラスタラベルが変化する箇所（`np.diff(cluster_labels) != 0`）を検出し、これをセグメントの境界とします。

#### 音声ファイルの切り出しと保存
```python
            for j in range(len(boundary_samples) - 1):
                start = boundary_samples[j]
                end = boundary_samples[j + 1]

                # 0長セグメント（start == end）は保存しない
                if end - start > 0:
                    # 波形データをスライス
                    segment_y = y[start:end]

                    # ファイル名: example-001.wav, example-002.wav, ...
                    # :03d は 3 桁ゼロパディング（001, 002, ..., 999）
                    out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")

                    # WAV ファイルとして保存
                    sf.write(out_path, segment_y, sr)
```
検出された境界（サンプル位置）に基づいて波形データをスライスし、`soundfile` を用いて個別の WAV ファイルとして保存します。
