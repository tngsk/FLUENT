# Extractor (`python/extractor.py`) 解説

## 実装と理論的背景
`python/extractor.py` は、入力された音声セグメント（WAVファイル）から、機械学習モデルへの入力として使用する26次元の物理特徴量ベクトルを抽出し、全体を標準化（Standardization）して JSON 形式で出力するモジュールです。

### 理論的背景
音声音楽の特性を捉えるために、様々な時間・周波数領域の特徴量を抽出します。
1. **MFCC (12次元)**: 音色（Timbre）を表す特徴量。第0係数（音量に依存する成分）を除外した12次元を使用し、音色そのものの特性に焦点を当てます。
2. **Spectral Features (4次元)**: 周波数スペクトルの形状を表します。
   - **Centroid**: 重心。音の「明るさ」に関連します。
   - **Flatness**: 平坦度。トーンかノイズ（パーカッシブな音など）かを示します。
   - **Rolloff**: 指定した割合のエネルギーが含まれる周波数。高周波エネルギーの指標となります。
   - **Bandwidth**: 帯域幅。音の広がりを示します。
3. **Theoretical Features (3次元)**: 音楽理論的な特徴。
   - **Tempo**: BPM (Beats Per Minute)。楽曲の速さ。
   - **Key**: 調（C, C#, D... の12種類）。
   - **Mode**: 長調 (Major) か短調 (Minor) か。
4. **Chroma Feature (7次元)**: 12音階の強さの分布。ここでは最初の7半音分を使用しています。

### 実装解説
もとのコードの主要な処理部分を引用しながら解説します。

#### 特徴量抽出関数 `extract_features`
```python
def extract_features(file_path: str) -> np.ndarray:
    y, sr = librosa.load(file_path, sr=None)
```
音声を読み込みます。

#### MFCC の抽出
```python
    # ============================================================
    # Step 3: MFCC（メル周波数ケプストラム係数）抽出
    # ============================================================
    # librosa.feature.mfcc の出力は (13, T) の2次元配列
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=n_fft)

    # 時系列（フレーム）方向に平均をとり、(13,) の1次元ベクトルにする
    mfcc_mean = np.mean(mfcc, axis=1)

    # 第0係数（mfcc_mean[0]）は「全体的な音の大きさ（パワー）」を強く反映するため、
    # 音色の特徴に集中するために除外し、残りの 12 次元を使用する
    mfcc_mean = mfcc_mean[1:]
```
MFCCを計算し、時間方向の平均を取った後、音量に依存する第0係数を除外して12次元ベクトルとします。

#### スペクトル特徴量の抽出
```python
    # ============================================================
    # Step 4-a: Spectral Centroid（スペクトラルセントロイド）
    # ============================================================
    # S: 振幅スペクトログラム（複素数スペクトルの絶対値）
    S, _ = librosa.magphase(librosa.stft(y, n_fft=n_fft))
    centroid = np.mean(librosa.feature.spectral_centroid(S=S))

    flatness = np.mean(librosa.feature.spectral_flatness(y=y, n_fft=n_fft))
    rolloff = np.mean(librosa.feature.spectral_rolloff(S=S, sr=sr))
    bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=n_fft))
```
STFT（短時間フーリエ変換）によりスペクトログラム `S` を得て、Centroid, Flatness, Rolloff, Bandwidth を計算し、それぞれの時間平均を取ります。

#### 理論的特徴量・クロマの抽出
```python
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, n_fft=n_fft)
    tempo_tuple = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    tempo = (
        tempo_tuple[0][0] if isinstance(tempo_tuple[0], np.ndarray) else tempo_tuple[0]
    )

    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    key = np.argmax(chroma_mean)
    mode = 1.0 if chroma_mean[(key + 4) % 12] > chroma_mean[(key + 3) % 12] else 0.0
```
Onset強度からテンポを推定し、Constant-Q変換に基づくクロマ特徴量からKeyとModeを推定します。

#### ベクトルの結合
```python
    features = np.concatenate(
        [
            mfcc_mean,
            np.array([centroid, flatness, rolloff, bandwidth]),
            np.array([tempo, key, mode]),
            chroma_mean[:7],
        ]
    )
    return features
```
抽出した特徴量を全て結合し、26次元ベクトルを作成します。

#### データセットの標準化
```python
        # StandardScaler: 各列（特徴量）ごとに平均0、標準偏差1に正規化
        scaler = StandardScaler()
        all_features_scaled = scaler.fit_transform(all_features)

        # scaler.pkl: 後の推論時に、新規セグメントを同じスケール体系で変換するために必要
        scaler_path = os.path.join(os.path.dirname(output_json), "scaler.pkl")
        joblib.dump(scaler, scaler_path)
```
全てのセグメントから抽出した特徴量を行列（`all_features`）とし、`scikit-learn` の `StandardScaler` を用いて標準化します（平均0、分散1）。この `scaler` は推論時にも必要となるため、`scaler.pkl` として保存されます。
