#!/usr/bin/env bash
# Isolated official runtime; the chat environment does not need PyTorch.
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BACKEND="${IRODORI_BACKEND:-cu128}"
REVISION="61012c760f22f7b4a6c21c5c5f8f9e148120b6f9"
case "$BACKEND" in cpu|cu128|rocm) ;; *) echo 'IRODORI_BACKEND must be cpu, cu128 or rocm' >&2; exit 2;; esac
if [[ ! -d "$TASK_DIR/runtime/.git" ]]; then
  git clone https://github.com/Aratako/Irodori-TTS-Server.git "$TASK_DIR/runtime"
fi
git -C "$TASK_DIR/runtime" checkout "$REVISION"
cd "$TASK_DIR/runtime"
uv sync --locked --extra "$BACKEND"
uv run --no-sync python "$TASK_DIR/download.py" --output "$TASK_DIR/models" "$@"
