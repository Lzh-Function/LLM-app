#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ ! -d strata ]]; then
    git clone --depth 1 https://github.com/Niko1221/Strata.git strata
fi
exec bash strata/setup.sh \
    --family qwen --model IQ2_XS --context 32768 --kv int8 \
    --vision cpu --experimental-speed-projection off --low-ram off \
    --data-dir "$PWD/Strata-data" --port 15080 --host 127.0.0.1 \
    --yes --no-start "$@"
