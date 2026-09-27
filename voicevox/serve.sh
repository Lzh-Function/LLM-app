#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
export XDG_DATA_HOME="$PWD/data"
export XDG_CACHE_HOME="$PWD/cache"

exec ./linux-cpu-x64/run --host 127.0.0.1 --port 50021 --cpu_num_threads "${VV_CPU_NUM_THREADS:-8}" "$@"
