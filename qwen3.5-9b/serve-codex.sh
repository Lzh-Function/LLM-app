#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$TASK_DIR/../strata/.venv/bin/python" "$TASK_DIR/../codex/llama_server.py" qwen-9b "$@"
