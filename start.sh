#!/usr/bin/env bash
set -euo pipefail

# Run from the script's directory so 'core', 'api', 'cli' resolve whether
# the script is invoked from $PWD, a parent dir, or absolute path.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$SCRIPT_DIR"

HOST="${ROITELET_APP_HOST:-0.0.0.0}"
PORT="${ROITELET_APP_PORT:-8000}"

# Single uvicorn process serves both the JSON API and the static web client
# (mounted at '/' — see api/main.py).
exec python -m uvicorn api.main:app --host "$HOST" --port "$PORT"
