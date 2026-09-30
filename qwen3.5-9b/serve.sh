#!/usr/bin/env bash
# Qwen3.5-9B (Dense) を llama-server で起動する。全レイヤーを GPU に載せる。
# 単体でも使える (OpenAI 互換 API: http://127.0.0.1:${PORT}/v1)。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MMPROJ="${MMPROJ:-models/mmproj-F16.gguf}"
if [[ ! -f "$MMPROJ" ]]; then
    echo "画像用モデルがありません: $MMPROJ。LLM/download-mmproj.sh を実行してください。" >&2
    exit 1
fi
exec ../llama.cpp/llama-server \
    --model models/Qwen3.5-9B-UD-Q5_K_XL.gguf \
    --mmproj "$MMPROJ" --no-mmproj-offload \
    --alias qwen3.5-9b \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --n-gpu-layers all \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit off \
    --jinja --reasoning-format deepseek \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 \
    "$@"
