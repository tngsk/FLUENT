#!/bin/bash
set -e

# --- Python 環境 ---
uv sync

# --- Node.js 環境 ---
cd node
npm install
cd ..

# --- 初期データ生成 ---
mkdir -p data/raw_audio data/segments
uv run python python/generate_dummy_audio.py
uv run python python/segmenter.py
uv run python python/extractor.py

echo "Setup complete. Run 'cd node && npm run dev' to start."
