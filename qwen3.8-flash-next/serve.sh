#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
STRATA_DIR="$(cd ../strata && pwd)"
if [[ ! -f "$STRATA_DIR/strata-sc117-iq3_xxs.json" ]]; then
    echo "Strata のセットアップが必要です: bash setup.sh" >&2
    exit 1
fi
exec "$STRATA_DIR/.venv/bin/python" chat_server.py \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-5071}" "$@"
