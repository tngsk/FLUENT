import librosa
import numpy as np
import os
import json
import pickle
from sklearn.preprocessing import StandardScaler

def simple_extractor(segment_dir, output_json="features.json", output_scaler="scaler.pkl"):
    """
    ディレクトリ内のWAVファイルからMFCC特徴量を抽出し、
    標準化（Standardization）を行ってJSONに保存する最小構成のエクストラクター。
    """
    file_ids = []
    features_list = []

    # 1. 各ファイルから特徴量（MFCC）を抽出
    for filename in os.listdir(segment_dir):
        if not filename.endswith(".wav"):
            continue

        filepath = os.path.join(segment_dir, filename)
        file_id = os.path.splitext(filename)[0]

        try:
            y, sr = librosa.load(filepath, sr=22050) # サンプリングレートを固定
            if len(y) < 2048: # 短すぎる音声をスキップ
                continue

            # MFCCの計算 (13次元)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            # 時間軸方向の平均をとって1次元の特徴ベクトルにする
            mfcc_mean = np.mean(mfcc, axis=1)

            # (オプション) 第0係数（音量）を除外して12次元にする場合:
            # mfcc_mean = mfcc_mean[1:]

            features_list.append(mfcc_mean)
            file_ids.append(file_id)
            print(f"Extracted: {file_id}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")

    if not features_list:
        print("No valid audio files processed.")
        return

    # 2. 標準化（StandardScaler）
    X = np.array(features_list)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. データの整形 (A-MAP JSON形式を模倣)
    data_dict = {}
    for i, file_id in enumerate(file_ids):
        data_dict[file_id] = X_scaled[i].tolist()

    output_data = {
        "cols": X_scaled.shape[1],
        "data": data_dict
    }

    # 4. JSONとScaler(pkl)の保存
    with open(output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    with open(output_scaler, 'wb') as f:
        pickle.dump(scaler, f)

    print(f"Saved features to {output_json}")
    print(f"Saved scaler to {output_scaler}")

if __name__ == "__main__":
    # 使用例
    # simple_extractor("output_segments", "features.json", "scaler.pkl")
    print("This is a template for extracting audio features.")
