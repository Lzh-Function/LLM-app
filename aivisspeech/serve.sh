#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export XDG_DATA_HOME="$PWD/data"
export XDG_CACHE_HOME="$PWD/cache"
export VV_CPU_NUM_THREADS="${VV_CPU_NUM_THREADS:-8}"

exec ./Linux-x64/run --host 127.0.0.1 --port 10101 --no-use_gpu --disable_sentry "$@"
