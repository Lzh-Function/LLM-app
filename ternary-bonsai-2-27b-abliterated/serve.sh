#!/usr/bin/env bash
# Hikari07jp の Bonsai 2 PQ2_0 派生モデルを Prism ML fork で起動する。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
MMPROJ="${MMPROJ:-models/Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf}"
if [[ ! -f "$MMPROJ" ]]; then
    echo "画像用モデルがありません: $MMPROJ。LLM/download-mmproj.sh を実行してください。" >&2
    exit 1
fi
exec ../llama-prism/llama-server \
    --model models/Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf \
    --mmproj "$MMPROJ" --no-mmproj-offload \
    --alias ternary-bonsai-2-27b-abliterated \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" \
    --ctx-size "${CTX_SIZE:-16384}" \
    --parallel 1 \
    --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --fit on --fit-target 768 \
    --jinja --reasoning-format deepseek --reasoning-effort medium \
    --temp 1.0 --top-p 0.95 --top-k 20 --min-p 0.05 \
    "$@"
