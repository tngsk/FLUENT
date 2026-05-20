import os
import glob
import subprocess
import sys

def normalize_all_segments(segments_dir):
    # data/segments 内の全ての wav ファイルを取得
    wav_files = glob.glob(os.path.join(segments_dir, "*.wav"))
    if not wav_files:
        print(f"No WAV files found in {segments_dir}", file=sys.stderr)
        return

    print(f"Normalizing {len(wav_files)} files in {segments_dir} to -12dB LUFS...", file=sys.stdout)

    for wav_file in wav_files:
        temp_output = wav_file + ".tmp.wav"
        
        # youtube_dl.py と同じラウドネス正規化設定（loudnorm）を適用
        cmd = [
            "ffmpeg", "-y", "-i", wav_file,
            "-af", "loudnorm=I=-21:TP=-9.0:LRA=7,volume=-3dB",
            "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
            temp_output
        ]
        
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            # 変換に成功したら元のファイルを置き換え
            os.replace(temp_output, wav_file)
            print(f"Normalized: {os.path.basename(wav_file)}", file=sys.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {wav_file}: {e}", file=sys.stderr)
            if os.path.exists(temp_output):
                os.remove(temp_output)

    print("\nNormalization complete. Please run 'python python/extractor.py' to update features.", file=sys.stdout)

if __name__ == "__main__":
    normalize_all_segments("data/segments")