#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec python3 qwen3.8-flash-next/install_sc117.py "$@"
