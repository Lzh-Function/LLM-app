#!/usr/bin/env bash
set -euo pipefail
unset VIRTUAL_ENV
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REVISION="022e286b98fbec7e1e916cb940cdf532cd9f488e"
if [[ ! -d "$TASK_DIR/runtime/.git" ]]; then
  git clone https://github.com/QwenLM/Qwen3-TTS.git "$TASK_DIR/runtime"
fi
git -C "$TASK_DIR/runtime" checkout "$REVISION"
cd "$TASK_DIR"
uv sync --python 3.12 --locked
uv run --no-sync python download.py "$@"
