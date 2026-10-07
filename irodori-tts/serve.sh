#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$TASK_DIR/runtime/.venv/bin/python" "$TASK_DIR/launch.py" "$@"
