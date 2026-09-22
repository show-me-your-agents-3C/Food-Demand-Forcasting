#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
if [ -x .venv/bin/python ]; then
    FRESHFLOW_PYTHON="$PWD/.venv/bin/python"
elif [ -x /private/tmp/freshflow-review-20260921-venv/bin/python ]; then
    FRESHFLOW_PYTHON=/private/tmp/freshflow-review-20260921-venv/bin/python
else
    FRESHFLOW_BASE=""
    for candidate in python3.12 python3.11 python3.10 "$HOME/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3"; do
        if command -v "$candidate" >/dev/null 2>&1 && "$candidate" -c 'import sys; sys.exit(not ((3,10) <= sys.version_info[:2] <= (3,12)))'; then
            FRESHFLOW_BASE="$candidate"
            break
        fi
    done
    if [ -z "$FRESHFLOW_BASE" ]; then
        echo "Please install Python 3.10–3.12 to start FreshFlow."
        exit 1
    fi
    "$FRESHFLOW_BASE" -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    FRESHFLOW_PYTHON="$PWD/.venv/bin/python"
fi
echo "FreshFlow: http://127.0.0.1:8001"
echo "Keep this window open. Press Control+C to stop."
exec "$FRESHFLOW_PYTHON" -B -m uvicorn src.api.main:app --host 127.0.0.1 --port 8001
