#!/usr/bin/env bash
set -euo pipefail
unset VIRTUAL_ENV
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$TASK_DIR"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
exec uv run --no-sync python service.py "$@"
