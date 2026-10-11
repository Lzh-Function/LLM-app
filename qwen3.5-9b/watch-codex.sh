#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TRACE_DIR="${CODEX_TRACE_DIR:-$TASK_DIR/codex-traces}"
exec tail -n 200 -F "$TRACE_DIR/live.log"
