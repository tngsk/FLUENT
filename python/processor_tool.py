import argparse
import sys
import os
import json
import glob
import subprocess
from pydub import AudioSegment

def process_regions(input_file, title, regions, out_dir):
    try:
        os.makedirs(out_dir, exist_ok=True)

        safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c in (' ', '-', '_')]).rstrip()
        safe_title = safe_title.replace(" ", "_")
        if not safe_title:
            safe_title = "Audio"

        existing_files = glob.glob(os.path.join(out_dir, f"{safe_title}_*.wav"))
        max_idx = 0
        for f in existing_files:
            base = os.path.basename(f)
            try:
                idx_str = base.replace(f"{safe_title}_", "").replace(".wav", "")
                idx = int(idx_str)
                if idx > max_idx:
                    max_idx = idx
            except ValueError:
                pass

        next_idx = max_idx + 1

        # Load the whole audio file
        audio = AudioSegment.from_file(input_file)

        results = []
        for region in regions:
            start_ms = int(region.get("start", 0) * 1000)
            end_ms = int(region.get("end", 0) * 1000)

            if end_ms <= start_ms:
                continue

            cropped = audio[start_ms:end_ms]

            out_filename = f"{safe_title}_{next_idx:03d}.wav"
            out_filepath = os.path.join(out_dir, out_filename)
            out_filepath = os.path.abspath(out_filepath)

            # 一時的に保存してから ffmpeg を使用してラウドネス正規化（-12dB LUFS）を適用し、
            # YouTubeからのダウンロード結果と音量を統一する
            temp_segment = out_filepath + ".raw.wav"
            cropped.export(temp_segment, format="wav")

            cmd = [
                "ffmpeg", "-y", "-i", temp_segment,
                "-af", "loudnorm=I=-21:TP=-9.0:LRA=7,volume=-3dB",
                "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                out_filepath
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            os.remove(temp_segment)

            results.append({
                "id": out_filename.replace(".wav", ""),
                "file": out_filename
            })
            next_idx += 1

        print(json.dumps({"success": True, "saved_files": results}))
        return 0
    except Exception as e:
        print(json.dumps({"success": False, "error": str(e)}))
        return 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input temporary audio file")
    parser.add_argument("--title", required=True, help="Video title")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--regions", required=True, help="JSON string of regions [{'start': 1.0, 'end': 5.0}]")

    args = parser.parse_args()

    try:
        regions = json.loads(args.regions)
    except json.JSONDecodeError:
        print(json.dumps({"success": False, "error": "Invalid regions JSON"}))
        sys.exit(1)

    sys.exit(process_regions(args.input, args.title, regions, args.out_dir))
