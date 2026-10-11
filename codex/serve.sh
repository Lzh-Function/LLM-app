#!/usr/bin/env bash
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PROFILE="${1:-}"
if [[ $# -gt 0 ]]; then shift; fi
case "$PROFILE" in
  flash-next|qwen) exec bash "$TASK_ROOT/qwen3.8-flash-next/serve-codex.sh" "$@" ;;
  qwen-9b) exec bash "$TASK_ROOT/qwen3.5-9b/serve-codex.sh" "$@" ;;
  bonsai) exec bash "$TASK_ROOT/ternary-bonsai-2-27b-abliterated/serve-codex.sh" "$@" ;;
  *) echo 'usage: bash codex/serve.sh {flash-next|qwen-9b|bonsai} [options]' >&2; exit 2 ;;
esac
