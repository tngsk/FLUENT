import librosa
import soundfile as sf
import os
import numpy as np

def simple_segmenter(input_wav_path, output_dir, duration_sec=2.0):
    """
    指定した秒数で音声を単純に等間隔分割する最小構成のセグメンター。
    librosaの高度な分割機能ではなく、配列のスライスによるシンプルな分割を行います。
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. 音声のロード
    print(f"Loading {input_wav_path}...")
    y, sr = librosa.load(input_wav_path, sr=None)

    # 2. 分割サイズの計算（サンプル数）
    segment_samples = int(duration_sec * sr)
    total_samples = len(y)

    # 3. 分割と保存
    segment_idx = 1
    for start in range(0, total_samples, segment_samples):
        end = min(start + segment_samples, total_samples)
        segment_y = y[start:end]

        # 音声が短すぎる場合はスキップ（例: 0.1秒未満）
        if len(segment_y) < sr * 0.1:
            continue

        output_path = os.path.join(output_dir, f"segment_{segment_idx:03d}.wav")
        sf.write(output_path, segment_y, sr)
        print(f"Saved: {output_path}")
        segment_idx += 1

if __name__ == "__main__":
    # 使用例 (ダミーファイルが必要な場合は用意してください)
    # simple_segmenter("input.wav", "output_segments", duration_sec=2.0)
    print("This is a template for segmenting audio.")
