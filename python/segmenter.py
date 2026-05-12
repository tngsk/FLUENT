import argparse
import glob
import os
import sys

import librosa
import numpy as np
import soundfile as sf


def segment_audio(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    audio_files = glob.glob(os.path.join(input_dir, "*.wav"))
    total = len(audio_files)
    if total == 0:
        print("No WAV files found.", file=sys.stderr)
        return
    segment_id = 1

    for i, file_path in enumerate(audio_files, 1):
        print(
            f"[{i}/{total}] Segmenting: {os.path.basename(file_path)}",
            file=sys.stderr,
            flush=True,
        )
        y, sr = librosa.load(file_path, sr=None)
        hop_length = 512
        if len(y) < sr * 1:
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1
            continue
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=hop_length)
        k = int(max(2, len(y) // (sr * 2)))
        try:
            boundaries = librosa.segment.agglomerative(mfcc, k)
            boundary_samples = librosa.frames_to_samples(
                boundaries, hop_length=hop_length
            )
            boundary_samples = np.unique(
                np.concatenate([[0], boundary_samples, [len(y)]])
            )
            for j in range(len(boundary_samples) - 1):
                start = boundary_samples[j]
                end = boundary_samples[j + 1]
                if end - start > 0:
                    segment_y = y[start:end]
                    out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
                    sf.write(out_path, segment_y, sr)
                    segment_id += 1
        except Exception as e:
            print(f"Error segmenting {file_path}: {e}", file=sys.stderr)
            out_path = os.path.join(output_dir, f"example-{segment_id:03d}.wav")
            sf.write(out_path, y, sr)
            segment_id += 1

    print(f"Done: {segment_id - 1} segments written to {output_dir}", file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment audio files")
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw_audio",
        help="Input directory with WAV files",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/segments",
        help="Output directory for segments",
    )

    args = parser.parse_args()
    segment_audio(args.input, args.output)
