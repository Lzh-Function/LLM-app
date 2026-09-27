#!/usr/bin/env bash
# Qwen3.8-27B (Dense) を llama-server で起動する。
# 単体でも使える (OpenAI 互換 API: http://127.0.0.1:${PORT}/v1)。
# 重みが 12GB VRAM を超えるため --fit で一部レイヤーを CPU 側へ逃がす。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ../llama.cpp/llama-server \
    --model models/Qwen3.8-27B-UD-Q4_K_M.gguf \
    --alias qwen3.8-27b \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit on --fit-target 768 \
    --threads 8 \
    --jinja --reasoning-format deepseek \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    "$@"
