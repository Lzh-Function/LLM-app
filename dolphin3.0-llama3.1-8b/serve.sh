#!/usr/bin/env bash
# Dolphin 3.0 Llama 3.1 8B を llama-server で起動する。全レイヤーを GPU に載せる。
# 単体でも使える (OpenAI 互換 API: http://127.0.0.1:${PORT}/v1)。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ../llama.cpp/llama-server \
    --model models/Dolphin3.0-Llama3.1-8B-Q6_K.gguf \
    --alias dolphin3.0-llama3.1-8b \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --n-gpu-layers all \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit off \
    --jinja \
    --temp 0.7 --top-p 0.95 --top-k 40 \
    "$@"
