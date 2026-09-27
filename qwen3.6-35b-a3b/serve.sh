#!/usr/bin/env bash
# Qwen3.6-35B-A3B (MoE 35B / active 3B) を llama-server で起動する。
# 単体でも使える (OpenAI 互換 API: http://127.0.0.1:${PORT}/v1)。
# --fit (既定 on) が VRAM に収まるよう MoE expert を自動で CPU 側へ逃がす。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
exec ../llama.cpp/llama-server \
    --model models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf \
    --alias qwen3.6-35b-a3b \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-32768}" \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit on --fit-target 768 \
    --threads 8 \
    --jinja --reasoning-format deepseek \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0 \
    --presence-penalty 1.5 \
    "$@"
