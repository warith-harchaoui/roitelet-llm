#!/usr/bin/env bash
set -euo pipefail

HOST="${ROITELET_APP_HOST:-0.0.0.0}"
API_PORT="${ROITELET_APP_PORT:-8000}"
UI_PORT="${ROITELET_STREAMLIT_PORT:-8501}"

python -m uvicorn api.main:app --host "$HOST" --port "$API_PORT" &
API_PID=$!

streamlit run gui/main.py --server.address "$HOST" --server.port "$UI_PORT" --browser.gatherUsageStats false &
UI_PID=$!

cleanup() {
  kill "$API_PID" "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait -n "$API_PID" "$UI_PID"
